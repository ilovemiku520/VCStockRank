# main.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from config import ModelConfig, PortfolioConfig, BacktestConfig
from data.download import load_stock_pool, download_all
from data.clean import DataCleaner
from data.features import FactorBuilder
from model.multitask import MultiTaskVCformerTPA
from training.dataset import TimeSeriesDataset
from training.trainer import Trainer
from portfolio.backtest import Backtester
from evaluation.factor_analysis import FactorAnalyzer


class MultiModalStrategy:
    """多模态选股策略主类（不含宏观数据）"""

    def __init__(self, config):
        self.config = config
        self.factor_builder = FactorBuilder()
        self.cleaner = DataCleaner()
        self.model = None
        self.trainer = None
        self.data = {}
        self.factors = {}
        self.predictions = {}
        self.backtest_results = None
        self.evaluation_results = {}
        self.feature_cols = None

    def load_data(self):
        """加载和预处理数据（核心修正：先算标签，再标准化特征）"""
        print("=" * 60)
        print("1. 数据加载与预处理")
        print("=" * 60)

        print("\n加载股票池...")
        stocks = load_stock_pool("stock_pool.csv", max_stocks=self.config.MAX_STOCKS)
        if not stocks:
            print("错误: 股票池为空")
            return None

        print("\n下载日频数据...")
        daily = download_all(stocks, self.config.DATA_START, self.config.DATA_END)
        if daily.empty:
            print("错误: 未获取到任何日频数据")
            return None

        # ========== 关键修改：在清洗之前保存原始价格 ==========
        self.data['price_raw'] = daily[['close']].copy()
        daily[['close']].to_parquet("data/daily_raw.parquet")
        print("  ✓ 原始价格数据（未清洗）已保存至 data/daily_raw.parquet")
        # ====================================================

        print("\n数据清洗...")
        daily_cleaned = self.cleaner.clean(daily)
        daily_cleaned.to_parquet("data/daily.parquet")
        print("  ✓ 清洗后日频数据已保存至 data/daily.parquet")

        # ========== 关键修正：在标准化之前计算标签 ==========
        print("\n构建标签（基于原始价格，防止畸变）...")
        factors = daily_cleaned.copy()

        if 'ret_1d' not in factors.columns:
            factors['ret_1d'] = factors.groupby('stock')['close'].pct_change()

        factors['future_ret_5d'] = factors.groupby('stock')['close'].transform(
            lambda x: x.shift(-5) / x - 1
        )

        market_ret_daily = factors.groupby('date')['ret_1d'].mean()
        factors['market_ret'] = factors.index.get_level_values('date').map(market_ret_daily.to_dict())

        market_ret_future = market_ret_daily.shift(-5)
        factors['market_ret_future'] = factors.index.get_level_values('date').map(market_ret_future.to_dict())

        factors['excess_ret_5d'] = factors['future_ret_5d'] - factors['market_ret_future']

        try:
            vol_series = factors.groupby('stock')['ret_1d'].transform(
                lambda x: x.rolling(5, min_periods=1).std().shift(-5)
            )
            factors['future_vol_5d'] = vol_series
        except Exception as e:
            print(f"波动率计算失败: {e}")
            factors['future_vol_5d'] = np.nan

        print("\n目标列非NaN统计（基于原始价格）：")
        print(f"  future_ret_5d: {factors['future_ret_5d'].notna().sum()}")
        print(f"  excess_ret_5d: {factors['excess_ret_5d'].notna().sum()}")
        print(f"  future_vol_5d: {factors['future_vol_5d'].notna().sum()}")

        factors.drop(columns=['market_ret_future'], inplace=True, errors='ignore')

        # ========== 现在才构建因子（只标准化特征列）==========
        print("\n构建因子（标准化特征，但保留原始标签）...")
        factors = self.factor_builder.create_factors_panel(factors)
        if factors.empty:
            print("错误: 因子构建失败，无有效数据")
            return None

        # ========== 缺失值填充 ==========
        print("\n填充缺失值...")
        exclude_cols = ['close', 'open', 'high', 'low', 'volume', 'amount', 'turnover', 'stock',
                        'future_ret_5d', 'future_vol_5d', 'market_ret', 'excess_ret_5d', 'ret_1d']
        fill_cols = [col for col in factors.columns if col not in exclude_cols]

        for col in tqdm(fill_cols, desc="  填充列"):
            factors[col] = factors.groupby('stock')[col].transform(
                lambda g: g.ffill().bfill()
            )
        factors[fill_cols] = factors[fill_cols].fillna(0)
        print(f"填充完成，共填充 {len(fill_cols)} 列")

        factors.to_csv("data/factors_filled.csv")

        self.data['daily'] = daily_cleaned
        self.data['factors'] = factors
        self.data['stocks'] = stocks

        print(f"\n数据加载完成:")
        print(f"  日频数据: {len(daily_cleaned)} 行")
        print(f"  因子数据: {len(factors)} 行, {len(factors.columns)} 列")
        print(f"  股票数量: {len(stocks)}")
        return factors

    def train_model(self):
        """训练模型（支持从 checkpoint 自动恢复 INPUT_DIM）"""
        print("\n" + "=" * 60)
        print("2. 模型训练")
        print("=" * 60)

        if self.data.get('factors') is None or self.data['factors'].empty:
            print("错误: 因子数据为空")
            return None

        exclude_cols = ['close', 'open', 'high', 'low', 'volume', 'amount',
                        'turnover', 'stock', 'future_ret_5d', 'future_vol_5d',
                        'market_ret', 'excess_ret_5d', 'ret_1d']
        feature_cols = [col for col in self.data['factors'].columns
                        if col not in exclude_cols and not col.startswith('future_')]

        if not feature_cols:
            print("错误: 没有可用特征列")
            return None

        print(f"特征数量: {len(feature_cols)}")
        self.config.INPUT_DIM = len(feature_cols)
        self.feature_cols = feature_cols

        best_model_path = os.path.join('logs', 'best_model.pt')
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location=self.config.DEVICE, weights_only=False)
                saved_input_dim = checkpoint.get('input_dim')
                if saved_input_dim is not None and saved_input_dim != self.config.INPUT_DIM:
                    print(f"检测到保存模型的 input_dim = {saved_input_dim}，自动更新配置")
                    self.config.INPUT_DIM = saved_input_dim
                self.model = MultiTaskVCformerTPA(self.config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.config.DEVICE)
                print(f"成功加载已有模型 (input_dim={self.config.INPUT_DIM})，跳过训练")
                return self.model
            except Exception as e:
                print(f"加载已有模型失败: {e}，将重新训练")

        dataset = TimeSeriesDataset(
            self.data['factors'],
            feature_cols=feature_cols,
            target_col='excess_ret_5d',
            vol_col='future_vol_5d',
            seq_len=self.config.SEQ_LEN
        )

        if len(dataset) == 0:
            print("错误: 数据集为空")
            return None

        batch_size = min(self.config.BATCH_SIZE, len(dataset))
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        if len(train_loader) == 0:
            print("错误: 数据加载器为空")
            return None

        self.model = MultiTaskVCformerTPA(self.config)
        self.model.to(self.config.DEVICE)
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        self.trainer = Trainer(self.model, self.config, train_loader)
        self.model = self.trainer.train()
        torch.save(self.model.state_dict(), 'model_checkpoint.pt')
        print("\n模型训练完成并保存")
        return self.model

    def generate_predictions(self):
        """生成预测：对每只股票独立提取历史序列"""
        print("\n" + "=" * 60)
        print("3. 生成预测")
        print("=" * 60)

        if self.model is None:
            print("错误: 模型未训练")
            return None

        if self.feature_cols is None:
            print("错误: 未找到特征列，请先训练模型")
            return None

        self.model.eval()
        feature_cols = self.feature_cols
        seq_len = self.config.SEQ_LEN
        factors = self.data['factors']

        predictions = {}
        dates = factors.index.get_level_values('date').unique()

        for date in tqdm(dates, desc="  预测交易日"):
            date_data = factors.xs(date, level='date')
            stocks = date_data.index.get_level_values('stock').unique()
            if len(stocks) < 1:
                continue

            scores = []
            vols = []
            stock_list = []

            for stock in stocks:
                stock_data = factors.xs(stock, level='stock').sort_index()
                try:
                    idx = stock_data.index.get_loc(date)
                except KeyError:
                    continue
                if idx < seq_len:
                    continue
                X = stock_data[feature_cols].iloc[idx - seq_len:idx].values
                if X.shape[0] != seq_len:
                    continue
                X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.config.DEVICE)
                with torch.no_grad():
                    outputs = self.model(X_tensor, return_attention=False)
                score = outputs['rank_score'].detach().cpu().numpy().flatten()[0]
                vol = outputs['vol_pred'].detach().cpu().numpy().flatten()[0]
                scores.append(score)
                vols.append(vol)
                stock_list.append(stock)

            if scores:
                predictions[date] = {
                    'scores': np.array(scores),
                    'vols': np.array(vols),
                    'stocks': np.array(stock_list)
                }

        self.predictions = predictions
        print(
            f"生成 {len(predictions)} 天的预测（每天 {np.mean([len(v['scores']) for v in predictions.values()]) if predictions else 0:.1f} 只股票）")
        return predictions

    def run_backtest(self):
        """运行回测（使用原始价格数据）"""
        print("\n" + "=" * 60)
        print("4. 回测")
        print("=" * 60)

        if not self.predictions:
            print("错误: 没有预测结果")
            return None

        price_data = self.data.get('price_raw')
        if price_data is None:
            price_data = self.data['daily'][['close']].copy()
            print("警告：未找到 price_raw，使用 daily 中的 close（可能已被标准化）")
        else:
            print("使用原始价格数据 (price_raw) 进行回测")

        if price_data.empty:
            print("错误: 价格数据为空，无法回测")
            return None

        backtester = Backtester(PortfolioConfig())

        daily_predictions = {}
        for date, preds in self.predictions.items():
            daily_predictions[date] = {}
            for stock, score, vol in zip(preds['stocks'], preds['scores'], preds['vols']):
                daily_predictions[date][stock] = {'score': float(score), 'vol': float(vol)}

        result = backtester.run(daily_predictions, price_data)
        self.backtest_results = result
        self._print_backtest_results()
        return result

    def _print_backtest_results(self):
        if hasattr(self.backtest_results, 'metrics'):
            m = self.backtest_results.metrics
            if m:
                print("\n回测绩效指标:")
                def fmt(v, d=4):
                    return "N/A" if v is None else f"{v:.{d}f}" if isinstance(v, float) else str(v)
                print(f"  年化收益率: {fmt(m.get('annual_return', 0))}")
                print(f"  夏普比率: {fmt(m.get('sharpe_ratio', 0))}")
                print(f"  最大回撤: {fmt(m.get('max_drawdown', 0))}")
                print(f"  胜率: {fmt(m.get('win_rate', 0))}")
                print(f"  信息系数: {fmt(m.get('ic', 'N/A'))}")
                print(f"  ICIR: {fmt(m.get('icir', 'N/A'))}")
            else:
                print("回测结果为空")

    def evaluate_predictions(self):
        """使用日收益率评估预测效果（IC、分层收益）"""
        print("\n" + "=" * 60)
        print("5. 模型评估（IC & 分层收益）")
        print("=" * 60)

        if not self.predictions:
            print("错误: 没有预测结果")
            return None

        # 准备日收益率
        price_data = self.data.get('price_raw')
        if price_data is None:
            print("错误: 缺少价格数据")
            return None

        # 计算日收益率
        price_data_ret = price_data.groupby('stock')['close'].pct_change().reset_index()
        price_data_ret['date'] = pd.to_datetime(price_data_ret['date'])
        price_data_ret.set_index(['date', 'stock'], inplace=True)
        price_data_ret.rename(columns={'close': 'daily_ret'}, inplace=True)

        # 构建评估数据框
        temp_factors = self.data['factors'][['excess_ret_5d']].copy()
        temp_factors = temp_factors.join(price_data_ret['daily_ret'], how='inner')

        # 添加预测得分（只保留有预测的日期和股票）
        for date, preds in self.predictions.items():
            if date not in temp_factors.index.get_level_values('date'):
                continue
            for stock, score in zip(preds['stocks'], preds['scores']):
                try:
                    temp_factors.loc[(date, stock), 'pred_score'] = score
                except:
                    pass

        temp_factors = temp_factors.dropna(subset=['pred_score', 'daily_ret'])

        if temp_factors.empty:
            print("无有效数据用于评估")
            return

        # 使用 FactorAnalyzer
        fa = FactorAnalyzer(temp_factors)

        # IC
        print("计算信息系数 (IC)...")
        ic_series = fa.compute_ic(factor_col='pred_score', ret_col='daily_ret', method='spearman')
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else np.nan
        ic_pos = (ic_series > 0).mean()
        print(f"  IC 均值: {ic_mean:.4f}, ICIR: {icir:.4f}, 正收益比率: {ic_pos:.2%}")

        # 分层收益
        print("计算分层收益（十分位，基于日收益）...")
        decile_returns, long_short = fa.compute_decile_returns(
            factor_col='pred_score',
            ret_col='daily_ret',
            n_groups=10,
            period=1
        )
        decile_metrics = {}
        for col in decile_returns.columns:
            rets = decile_returns[col].dropna()
            if len(rets) > 0:
                annual_ret = (1 + rets).prod() ** (252 / len(rets)) - 1
                sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan
                decile_metrics[col] = {'annual_return': annual_ret, 'sharpe': sharpe}
        long_short_rets = long_short.dropna()
        if len(long_short_rets) > 0:
            ls_annual = (1 + long_short_rets).prod() ** (252 / len(long_short_rets)) - 1
            ls_sharpe = long_short_rets.mean() / long_short_rets.std() * np.sqrt(252) if long_short_rets.std() > 0 else np.nan
            decile_metrics['long_short'] = {'annual_return': ls_annual, 'sharpe': ls_sharpe}

        print("\n分层组合年化收益与夏普（基于日收益）:")
        for g, m in decile_metrics.items():
            print(f"  {g:12s}: 年化收益 {m.get('annual_return', np.nan):.2%}, 夏普 {m.get('sharpe', np.nan):.3f}")

        self.evaluation_results = {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'ic_positive_ratio': ic_pos,
            'decile_metrics': decile_metrics,
            'long_short_returns': long_short,
            'daily_ic': ic_series,
        }
        # 保存日收益和分层收益
        daily_ic = self.evaluation_results['daily_ic']
        if not daily_ic.empty:
            daily_ic.to_csv('daily_ic.csv', header=True)
        if 'long_short_returns' in self.evaluation_results and not self.evaluation_results['long_short_returns'].empty:
            ls_df = pd.DataFrame({'date': self.evaluation_results['long_short_returns'].index,
                                  'long_short_return': self.evaluation_results['long_short_returns'].values})
            ls_df.to_csv('long_short_returns.csv', index=False)
        print("\n评估完成，IC 和分层收益已保存。")

    def run_full_pipeline(self):
        print("=" * 60)
        print("多模态选股策略 - 完整流程（仅量价因子）")
        print("=" * 60)
        start_time = datetime.now()

        if self.load_data() is None:
            print("数据加载失败，终止流程")
            return
        if self.train_model() is None:
            print("模型训练失败，终止流程")
            return
        if self.generate_predictions() is None:
            print("预测生成失败，终止流程")
            return
        self.run_backtest()
        self.evaluate_predictions()

        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"\n总耗时: {elapsed:.2f} 分钟")


def main():
    config = ModelConfig()
    strategy = MultiModalStrategy(config)
    strategy.run_full_pipeline()


if __name__ == "__main__":
    main()