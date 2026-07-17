# evaluation/factor_analysis.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')


class FactorAnalyzer:
    """
    因子分析类
    计算IC、IR、十分位组合、分层收益等
    """

    def __init__(self, factor_data, returns_data=None):
        self.factor_data = factor_data
        self.returns_data = returns_data
        self.results = {}

    def compute_ic(self, factor_col, ret_col='excess_ret_5d', method='spearman'):
        """计算信息系数（IC）"""
        ic_list = []
        dates = self.factor_data.index.get_level_values('date').unique()

        for date in dates:
            date_factors = self.factor_data.xs(date, level='date')
            if ret_col not in date_factors.columns:
                continue
            df = date_factors[[factor_col, ret_col]].dropna()
            if len(df) < 10:
                continue
            if method == 'pearson':
                ic = df[factor_col].corr(df[ret_col])
            else:
                ic = df[factor_col].corr(df[ret_col], method='spearman')
            ic_list.append({'date': date, 'ic': ic})

        ic_df = pd.DataFrame(ic_list).set_index('date')
        return ic_df['ic']

    def compute_icir(self, ic_series):
        return ic_series.mean() / (ic_series.std() + 1e-6)

    def compute_decile_returns(self, factor_col, ret_col='excess_ret_5d', n_groups=10, period=1):
        """
        计算十分位组合收益
        period : int, 收益周期（天数），默认1表示日收益。若 ret_col 为5日收益，则 period=5
        """
        decile_returns = []
        dates = self.factor_data.index.get_level_values('date').unique()

        for date in dates:
            date_data = self.factor_data.xs(date, level='date')
            if ret_col not in date_data.columns:
                continue
            df = date_data[[factor_col, ret_col]].dropna()
            if len(df) < n_groups:
                continue
            df['group'] = pd.qcut(df[factor_col], n_groups, labels=False, duplicates='drop')
            group_ret = df.groupby('group')[ret_col].mean()
            for g in range(n_groups):
                if g not in group_ret.index:
                    group_ret[g] = np.nan
            group_ret = group_ret.reindex(range(n_groups))
            group_ret['date'] = date
            decile_returns.append(group_ret)

        decile_df = pd.DataFrame(decile_returns).set_index('date')
        decile_df.columns = [f'group_{i}' for i in range(n_groups)]

        long_short = decile_df['group_9'] - decile_df['group_0']

        self.results['decile_returns'] = decile_df
        self.results['long_short'] = long_short
        self.results['cum_returns'] = (1 + decile_df).cumprod()

        return decile_df, long_short

    def compute_grouped_metrics(self, factor_col, ret_col='excess_ret_5d', n_groups=10, period=1):
        """计算各分组的绩效指标，年化时考虑周期 period"""
        decile_df, _ = self.compute_decile_returns(factor_col, ret_col, n_groups, period=period)

        metrics = {}
        for col in decile_df.columns:
            rets = decile_df[col].dropna()
            if len(rets) < 10:
                continue
            # 调整年化天数
            annual_factor = 252 / period
            metrics[col] = {
                'mean_return': rets.mean(),
                'std_return': rets.std(),
                'sharpe': rets.mean() / (rets.std() + 1e-6) * np.sqrt(annual_factor),
                'cum_return': (1 + rets).prod() - 1,
                'win_rate': (rets > 0).mean()
            }

        long_short = self.results.get('long_short')
        if long_short is not None and not long_short.empty:
            rets = long_short.dropna()
            annual_factor = 252 / period
            metrics['long_short'] = {
                'mean_return': rets.mean(),
                'std_return': rets.std(),
                'sharpe': rets.mean() / (rets.std() + 1e-6) * np.sqrt(annual_factor),
                'cum_return': (1 + rets).prod() - 1,
                'win_rate': (rets > 0).mean()
            }

        return pd.DataFrame(metrics).T

    def compute_factor_turnover(self, factor_col, period=20):
        dates = self.factor_data.index.get_level_values('date').unique()
        turnovers = []
        for i in range(1, len(dates)):
            date_prev = dates[i - 1]
            date_curr = dates[i]
            prev_data = self.factor_data.xs(date_prev, level='date')
            curr_data = self.factor_data.xs(date_curr, level='date')
            common_stocks = set(prev_data.index.get_level_values('stock')) & set(curr_data.index.get_level_values('stock'))
            if len(common_stocks) < 10:
                continue
            prev_rank = prev_data.loc[common_stocks, factor_col].rank()
            curr_rank = curr_data.loc[common_stocks, factor_col].rank()
            turnover = np.abs(prev_rank - curr_rank).sum() / (2 * len(common_stocks))
            turnovers.append({'date': date_curr, 'turnover': turnover})
        turnover_series = pd.DataFrame(turnovers).set_index('date')['turnover']
        return turnover_series

    def compute_factor_autocorrelation(self, factor_col, lags=10):
        dates = self.factor_data.index.get_level_values('date').unique()
        autocorr = []
        for i in range(1, len(dates)):
            date_t = dates[i]
            date_t_1 = dates[i - 1]
            data_t = self.factor_data.xs(date_t, level='date')
            data_t_1 = self.factor_data.xs(date_t_1, level='date')
            common = set(data_t.index.get_level_values('stock')) & set(data_t_1.index.get_level_values('stock'))
            if len(common) < 10:
                continue
            merged = pd.DataFrame({
                'f_t': data_t.loc[common, factor_col],
                'f_t_1': data_t_1.loc[common, factor_col]
            }).dropna()
            corr = merged['f_t'].corr(merged['f_t_1'])
            autocorr.append({'date': date_t, 'autocorr': corr})
        return pd.DataFrame(autocorr).set_index('date')['autocorr']

    def analyze(self, factor_cols=None, ret_col='excess_ret_5d', period=1):
        if factor_cols is None:
            exclude_cols = ['close', 'open', 'high', 'low', 'volume', 'amount',
                            'turnover', 'stock', 'date', ret_col, 'future_ret_5d', 'future_vol_5d']
            factor_cols = [col for col in self.factor_data.columns if col not in exclude_cols]

        results = {}
        for factor in factor_cols:
            if factor not in self.factor_data.columns:
                continue
            print(f"Analyzing factor: {factor}")
            ic = self.compute_ic(factor, ret_col)
            icir = self.compute_icir(ic)
            decile, ls = self.compute_decile_returns(factor, ret_col, period=period)
            turnover = self.compute_factor_turnover(factor)

            # 手动计算年化收益和夏普（考虑 period）
            annual_factor = 252 / period
            results[factor] = {
                'ic_mean': ic.mean(),
                'ic_std': ic.std(),
                'icir': icir,
                'ic_positive_ratio': (ic > 0).mean(),
                'long_short_mean': ls.mean(),
                'long_short_sharpe': ls.mean() / (ls.std() + 1e-6) * np.sqrt(annual_factor),
                'turnover_mean': turnover.mean(),
                'n_days': len(ic)
            }

        self.results['summary'] = pd.DataFrame(results).T
        return self.results

    def plot_decile_performance(self, factor_col, period=1, save_path=None):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            decile_df = self.results.get('decile_returns')
            if decile_df is None:
                _, _ = self.compute_decile_returns(factor_col, period=period)
                decile_df = self.results['decile_returns']

            cum_ret = (1 + decile_df).cumprod()
            plt.figure(figsize=(12, 6))
            for col in cum_ret.columns:
                plt.plot(cum_ret.index, cum_ret[col], label=col, alpha=0.7)
            if 'long_short' in self.results:
                ls_cum = (1 + self.results['long_short']).cumprod()
                plt.plot(ls_cum.index, ls_cum, label='Long-Short', linewidth=2, color='black')
            plt.title(f'Factor {factor_col} Decile Portfolio Performance')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        except ImportError:
            print("matplotlib not installed, skipping plot")