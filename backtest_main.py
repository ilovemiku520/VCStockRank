# backtest_main.py
"""
独立回测与评估脚本
功能：加载已训练模型和因子数据，执行回测并输出绩效指标、IC 分析、分层收益。
依赖：logs/best_model.pt, data/factors_filled.csv, data/daily_raw.parquet
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import ModelConfig, PortfolioConfig
from model.multitask import MultiTaskVCformerTPA
from portfolio.backtest import Backtester
from evaluation.factor_analysis import FactorAnalyzer


def load_model_and_data(model_path='logs/best_model.pt', factor_path='data/factors_filled.csv'):
    print("\n[1/4] 加载模型与因子数据...")
    if not os.path.exists(factor_path):
        raise FileNotFoundError(f"因子数据文件不存在: {factor_path}")
    factors = pd.read_csv(factor_path, index_col=[0, 1], parse_dates=[0])
    print(f"  ✓ 因子数据加载成功，形状: {factors.shape}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    config = ModelConfig()
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    saved_input_dim = checkpoint.get('input_dim')
    if saved_input_dim is not None:
        config.INPUT_DIM = saved_input_dim
    print(f"  ✓ 从 checkpoint 读取 input_dim = {config.INPUT_DIM}")

    model = MultiTaskVCformerTPA(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    print("  ✓ 模型加载成功")
    return model, config, factors


def generate_predictions(model, config, factors):
    print("\n[2/4] 生成预测...")
    model.eval()
    exclude_cols = ['close', 'open', 'high', 'low', 'volume', 'amount',
                    'turnover', 'stock', 'future_ret_5d', 'future_vol_5d',
                    'market_ret', 'excess_ret_5d', 'ret_1d']
    feature_cols = [col for col in factors.columns
                    if col not in exclude_cols and not col.startswith('future_')]
    seq_len = config.SEQ_LEN
    device = config.DEVICE

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
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(X_tensor, return_attention=False)
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

    print(f"  ✓ 预测完成，共 {len(predictions)} 个交易日")
    return predictions


def get_price_data(factors):
    raw_path = "data/daily_raw.parquet"
    if os.path.exists(raw_path):
        price_data = pd.read_parquet(raw_path)[['close']].copy()
        print("  ✓ 使用原始价格数据 'data/daily_raw.parquet' (未经任何处理)")
        print(f"    close 均值={price_data['close'].mean():.4f}, 标准差={price_data['close'].std():.4f}")
        return price_data

    daily_path = "data/daily.parquet"
    if os.path.exists(daily_path):
        price_data = pd.read_parquet(daily_path)[['close']].copy()
        print("  ✓ 使用日频数据 'data/daily.parquet'")
        return price_data

    if 'close' in factors.columns:
        print("  ⚠ 使用因子数据中的 close 列（可能被标准化）")
        return factors[['close']].copy()
    else:
        raise FileNotFoundError("未找到价格数据，请运行 save_daily_only.py 生成。")


def run_backtest_and_evaluate(predictions, price_data, factors, portfolio_config):
    print("\n[3/4] 执行回测...")

    # ====== 过滤：只保留 price_data 中存在的股票 ======
    price_stocks = set(price_data.index.get_level_values('stock').unique())
    filtered_predictions = {}
    for date, preds in predictions.items():
        filtered = {}
        for stock, score, vol in zip(preds['stocks'], preds['scores'], preds['vols']):
            if stock in price_stocks:
                filtered[stock] = {'score': float(score), 'vol': float(vol)}
        if len(filtered) >= 5:
            filtered_predictions[date] = filtered
        else:
            print(f"  日期 {date} 有效股票数 {len(filtered)} < 5，跳过")

    print(f"  过滤后交易日数: {len(filtered_predictions)}")
    if len(filtered_predictions) == 0:
        raise ValueError("过滤后无有效交易日，请检查股票代码是否与价格数据匹配。")

    # ====== 执行回测 ======
    backtester = Backtester(portfolio_config, verbose=True)
    result = backtester.run(filtered_predictions, price_data)
    metrics = result.metrics

    print("\n[4/4] 计算评估指标...")

    # ----- 生成日收益率（用于 IC 和分层收益）-----
    # 按股票分组计算日收益率，对齐到因子数据的索引
    price_data_ret = price_data.groupby('stock')['close'].pct_change().reset_index()
    price_data_ret['date'] = pd.to_datetime(price_data_ret['date'])
    price_data_ret.set_index(['date', 'stock'], inplace=True)
    price_data_ret.rename(columns={'close': 'daily_ret'}, inplace=True)

    # 构建评估用的数据框，包含预测得分和日收益率
    temp_factors = factors[['excess_ret_5d']].copy()  # 保留原列，但以下评估将使用 daily_ret
    temp_factors = temp_factors.join(price_data_ret['daily_ret'], how='inner')  # 只保留有价格的交易日

    # 添加预测得分
    for date, preds in filtered_predictions.items():
        if date not in temp_factors.index.get_level_values('date'):
            continue
        for stock, info in preds.items():
            try:
                temp_factors.loc[(date, stock), 'pred_score'] = info['score']
            except:
                pass

    temp_factors = temp_factors.dropna(subset=['pred_score', 'daily_ret'])

    # ---- IC 分析 ----
    print("  计算信息系数 (IC)...")
    if not temp_factors.empty:
        fa = FactorAnalyzer(temp_factors)
        # 使用日收益率，period=1
        ic_series = fa.compute_ic(factor_col='pred_score', ret_col='daily_ret', method='spearman')
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else np.nan
        ic_positive_ratio = (ic_series > 0).mean()
        print(f"    ✓ IC 均值: {ic_mean:.4f}, ICIR: {icir:.4f}")
    else:
        ic_series = pd.Series(dtype=float)
        ic_mean = ic_std = icir = ic_positive_ratio = np.nan
        print("    ⚠ IC 分析失败：无有效数据")

    # ---- 分层收益 ----
    print("  计算分层收益（基于日收益率）...")
    if not temp_factors.empty:
        try:
            # 使用日收益率，period=1
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
            print(f"    ✓ 分层收益计算完成")
        except Exception as e:
            print(f"    ⚠ 分层收益计算失败: {e}")
            decile_metrics = {}
            long_short = pd.Series(dtype=float)
    else:
        decile_metrics = {}
        long_short = pd.Series(dtype=float)

    results = {
        'backtest_metrics': metrics,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': ic_positive_ratio,
        'daily_ic': ic_series,
        'long_short_returns': long_short,
        'decile_metrics': decile_metrics,
        'returns': result.returns,
    }
    return results


def main():
    print("=" * 60)
    print("回测与评估（独立脚本）")
    print("=" * 60)

    try:
        model, config, factors = load_model_and_data()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    predictions = generate_predictions(model, config, factors)
    if not predictions:
        print("无有效预测结果，终止。")
        return

    print("\n准备价格数据...")
    price_data = get_price_data(factors)
    print(f"  ✓ 价格数据形状: {price_data.shape}")

    portfolio_config = PortfolioConfig()
    results = run_backtest_and_evaluate(predictions, price_data, factors, portfolio_config)

    metrics = results['backtest_metrics']
    print("\n" + "=" * 60)
    print("回测绩效指标")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    print("\n" + "=" * 60)
    print("信息系数 (IC) 分析")
    print("=" * 60)
    print(f"IC 均值     : {results['ic_mean']:.4f}")
    print(f"IC 标准差   : {results['ic_std']:.4f}")
    print(f"ICIR        : {results['icir']:.4f}")
    print(f"IC 正收益比率: {results['ic_positive_ratio']:.2%}")

    if results['decile_metrics']:
        print("\n" + "=" * 60)
        print("分层组合（十分位）年化收益与夏普（基于日收益）")
        print("=" * 60)
        for group, m in results['decile_metrics'].items():
            print(f"{group:12s}: 年化收益 {m.get('annual_return', np.nan):.2%}, 夏普 {m.get('sharpe', np.nan):.3f}")

    # 保存结果
    returns = results['returns']
    if not returns.empty:
        returns_df = pd.DataFrame({'date': returns.index, 'return': returns.values})
        returns_df.to_csv('backtest_returns.csv', index=False)
        print("\n回测收益序列已保存至 backtest_returns.csv")

    daily_ic = results['daily_ic']
    if not daily_ic.empty:
        daily_ic.to_csv('daily_ic.csv', header=True)
        print("每日 IC 已保存至 daily_ic.csv")

    if 'long_short_returns' in results and not results['long_short_returns'].empty:
        ls_df = pd.DataFrame({'date': results['long_short_returns'].index,
                              'long_short_return': results['long_short_returns'].values})
        ls_df.to_csv('long_short_returns.csv', index=False)
        print("多空组合收益已保存至 long_short_returns.csv")

    print("\n✅ 评估完成。")


if __name__ == "__main__":
    main()