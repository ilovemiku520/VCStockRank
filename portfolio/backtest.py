# portfolio/backtest.py
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BacktestResult:
    def __init__(self, returns, positions, weights, metrics=None):
        self.returns = returns
        self.positions = positions
        self.weights = weights
        self.metrics = metrics or {}

    def summary(self):
        print("\n" + "=" * 60)
        print("Backtest Results Summary")
        print("=" * 60)
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60)
        return self.metrics

    def to_dict(self):
        return {
            'returns': self.returns,
            'positions': self.positions,
            'weights': self.weights,
            'metrics': self.metrics
        }


class Backtester:
    def __init__(self, config, benchmark_returns=None, verbose=True):
        self.config = config
        self.benchmark_returns = benchmark_returns
        self.verbose = verbose
        self.transaction_cost = config.TRANSACTION_COST
        self.slippage = config.SLIPPAGE

        from .optimizer import RiskParityOptimizer
        self.optimizer = RiskParityOptimizer(
            max_weight=config.MAX_WEIGHT,
            min_weight=0.001
        )

    def run(self, predictions, price_data):
        # ---------- 强制转换日期索引为 Timestamp ----------
        # 1. 转换价格数据的日期索引
        old_dates = price_data.index.get_level_values('date')
        new_dates = pd.to_datetime(old_dates).normalize()
        # 重建 MultiIndex
        new_index = pd.MultiIndex.from_arrays([new_dates, price_data.index.get_level_values('stock')])
        price_data.index = new_index
        price_dates = set(price_data.index.get_level_values('date'))

        # 2. 统一预测日期
        all_dates_raw = sorted(predictions.keys())
        all_dates = [pd.Timestamp(d).normalize() for d in all_dates_raw]

        # 3. 找出共有日期
        valid_dates = [d for d in all_dates if d in price_dates]

        if not valid_dates:
            print("错误: 没有价格数据与预测日期匹配")
            print(f"  预测日期范围: {all_dates[0]} ~ {all_dates[-1]}")
            print(f"  价格日期范围: {min(price_dates)} ~ {max(price_dates)}")
            print(f"  预测日期示例: {all_dates[:5]}")
            print(f"  价格日期示例: {list(price_dates)[:5]}")
            return BacktestResult(pd.Series(), {}, {})

        if self.verbose:
            print(f"  回测日期范围: {valid_dates[0]} ~ {valid_dates[-1]}, 共 {len(valid_dates)} 天")
            price_stocks = price_data.index.get_level_values('stock').unique()
            print(f"  价格数据中的股票数: {len(price_stocks)}")
            print(f"  价格数据索引示例 (转换后): {price_data.index[:3].tolist()}")

        portfolio_returns = []
        portfolio_weights = []
        portfolio_positions = []
        trade_log = []
        current_weights = None
        current_positions = None

        all_stocks = price_data.index.get_level_values('stock').unique().tolist()

        # 追踪上一个有效交易日
        last_valid_date = None

        iterator = tqdm(valid_dates, desc="回测进度") if self.verbose else valid_dates

        total_attempts = 0
        price_missing_count = 0

        for i, date in enumerate(iterator):
            # 获取该日期的预测
            preds = predictions.get(date)
            if preds is None:
                # 尝试用字符串格式获取
                alt_date = date.strftime('%Y-%m-%d')
                preds = predictions.get(alt_date)
                if preds is None:
                    continue

            stocks_today = list(preds.keys())

            if len(stocks_today) < 5:
                if self.verbose and i % 50 == 0:
                    print(f"  警告: {date} 只有 {len(stocks_today)} 只股票，跳过")
                continue

            scores = np.array([preds[s]['score'] for s in stocks_today])
            vols = np.array([preds[s]['vol'] for s in stocks_today])

            top_k = min(self.config.TOP_K, len(stocks_today))
            top_indices = np.argsort(scores)[-top_k:][::-1]

            selected_stocks = [stocks_today[i] for i in top_indices]
            selected_scores = scores[top_indices]
            selected_vols = vols[top_indices]

            target_weights = self.optimizer.optimize(
                selected_scores,
                selected_vols,
                cov_matrix=None
            )

            cost = 0.0
            if current_weights is not None:
                full_weights = np.zeros(len(all_stocks))
                for stock, w in zip(selected_stocks, target_weights):
                    try:
                        idx = all_stocks.index(stock)
                        full_weights[idx] = w
                    except ValueError:
                        continue
                turnover = np.sum(np.abs(full_weights - current_weights)) / 2
                cost = turnover * (self.transaction_cost + self.slippage)

            # 计算收益
            if i > 0 and current_positions is not None and last_valid_date is not None:
                prev_date = last_valid_date
                today_returns = []
                for stock, weight in current_positions.items():
                    total_attempts += 1
                    try:
                        prev_price = price_data.loc[(prev_date, stock), 'close']
                        curr_price = price_data.loc[(date, stock), 'close']
                        ret = curr_price / prev_price - 1
                        today_returns.append(ret * weight)
                    except KeyError as e:
                        price_missing_count += 1
                        if self.verbose and price_missing_count <= 10:
                            print(f"   ⚠️ 价格缺失: {stock} @ {date} - {e}")
                        continue
                    except Exception as e:
                        if self.verbose:
                            print(f"   ⚠️ 其他异常: {stock} @ {date} - {e}")
                        continue

                if today_returns:
                    daily_return = np.sum(today_returns) - cost
                    portfolio_returns.append(daily_return)
                    if self.verbose and i % 50 == 0:
                        print(f"  日期 {date}: 持仓 {len(current_positions)} 只, 日收益 {daily_return:.6f}")
                else:
                    if self.verbose:
                        print(f"   ⚠️ 日期 {date} 无有效收益, 设为 0")
                    portfolio_returns.append(0.0)

            # 更新持仓
            current_positions = {stock: w for stock, w in zip(selected_stocks, target_weights)}
            current_weights = np.zeros(len(all_stocks))
            for stock, w in zip(selected_stocks, target_weights):
                try:
                    idx = all_stocks.index(stock)
                    current_weights[idx] = w
                except ValueError:
                    continue

            last_valid_date = date

            if i % 50 == 0 and self.verbose:
                print(f"  已处理 {i+1}/{len(valid_dates)} 天，当前持仓 {len(current_positions)} 只")

        if self.verbose and total_attempts > 0:
            missing_rate = price_missing_count / total_attempts * 100
            print(f"  价格缺失率: {missing_rate:.1f}% ({price_missing_count}/{total_attempts})")
            if missing_rate > 50:
                print("   ⚠️ 价格缺失率过高，请检查股票代码是否与价格数据中的一致。")

        if portfolio_returns:
            returns_series = pd.Series(
                portfolio_returns,
                index=valid_dates[1:len(portfolio_returns) + 1]
            )
        else:
            returns_series = pd.Series()

        metrics = self._compute_metrics(returns_series)
        return BacktestResult(
            returns=returns_series,
            positions=portfolio_positions,
            weights=portfolio_weights,
            metrics=metrics
        )

    def _compute_metrics(self, returns):
        if returns.empty:
            return {}

        n_days = len(returns)
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252) if daily_vol is not None else 0

        rf_rate = 0.02
        excess_return = returns - rf_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_return.mean() / (returns.std() + 1e-6)

        cumsum = (1 + returns).cumprod()
        running_max = cumsum.expanding().max()
        drawdown = (cumsum - running_max) / running_max
        max_drawdown = drawdown.min()

        win_rate = (returns > 0).mean()
        positive = returns[returns > 0]
        negative = returns[returns < 0]
        profit_factor = positive.sum() / abs(negative.sum()) if len(negative) > 0 else np.inf

        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        ic = None
        icir = None
        if self.benchmark_returns is not None:
            common_idx = returns.index.intersection(self.benchmark_returns.index)
            if len(common_idx) > 0:
                ic = returns.loc[common_idx].corr(self.benchmark_returns.loc[common_idx])
                icir = ic / (returns.loc[common_idx].std() + 1e-6) if ic else 0

        max_consecutive_loss = 0
        max_consecutive_gain = 0
        current_loss = 0
        current_gain = 0
        for r in returns:
            if r < 0:
                current_loss += 1
                current_gain = 0
                max_consecutive_loss = max(max_consecutive_loss, current_loss)
            else:
                current_gain += 1
                current_loss = 0
                max_consecutive_gain = max(max_consecutive_gain, current_gain)

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_consecutive_loss': max_consecutive_loss,
            'max_consecutive_gain': max_consecutive_gain,
            'n_days': n_days,
            'ic': ic,
            'icir': icir
        }
        return metrics

    def compute_factor_performance(self, returns, factor_scores):
        long_short = []
        for date in sorted(factor_scores.keys()):
            if date not in returns:
                continue
            scores = factor_scores[date]
            rets = returns[date]
            df = pd.DataFrame({'score': scores, 'ret': rets}).dropna()
            if len(df) < 10:
                continue
            df['decile'] = pd.qcut(df['score'], 10, labels=False, duplicates='drop')
            group_returns = df.groupby('decile')['ret'].mean()
            if len(group_returns) >= 2:
                top = group_returns.max()
                bottom = group_returns.min()
                long_short.append(top - bottom)
        return pd.Series(long_short, index=sorted(factor_scores.keys())[:len(long_short)])

    def generate_report(self, result):
        metrics = result.metrics
        report = f"""
        ================ 回测报告 ================
        回测期间: {result.returns.index[0]} 至 {result.returns.index[-1]}
        交易日数: {metrics.get('n_days', 0)}

        绩效指标:
        总收益率: {metrics.get('total_return', 0):.2%}
        年化收益率: {metrics.get('annual_return', 0):.2%}
        年化波动率: {metrics.get('annual_volatility', 0):.2%}
        夏普比率: {metrics.get('sharpe_ratio', 0):.3f}
        最大回撤: {metrics.get('max_drawdown', 0):.2%}
        卡玛比率: {metrics.get('calmar_ratio', 0):.3f}
        胜率: {metrics.get('win_rate', 0):.2%}
        盈亏比: {metrics.get('profit_factor', 0):.3f}
        信息系数: {metrics.get('ic', 0):.4f}
        ICIR: {metrics.get('icir', 0):.4f}
        ===========================================
        """
        print(report)
        return report


def prepare_backtest_data(factors_df, predictions_df):
    returns_dict = {}
    scores_dict = {}

    for date in factors_df.index.get_level_values('date').unique():
        date_data = factors_df.xs(date, level='date')
        rets = date_data['close'].pct_change().dropna()
        returns_dict[date] = rets.to_dict()

        if date in predictions_df.index.get_level_values('date'):
            preds = predictions_df.xs(date, level='date')
            scores_dict[date] = preds['score'].to_dict()

    return returns_dict, scores_dict