# evaluation/ff_regression.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class FamaFrenchRegression:
    """
    Fama-French因子回归
    支持三因子和五因子模型
    """

    def __init__(self, model_type='five_factor'):
        """
        Parameters:
        -----------
        model_type : str, 'three_factor' or 'five_factor'
        """
        self.model_type = model_type
        self.alpha = None
        self.beta = None
        self.r_squared = None
        self.t_stats = None
        self.results = {}

    def fit(self, portfolio_returns, factor_data):
        """
        拟合Fama-French回归

        Parameters:
        -----------
        portfolio_returns : Series, 策略日收益率（索引为日期）
        factor_data : DataFrame, 因子数据，列包含:
            - 'Mkt-RF': 市场超额收益
            - 'SMB': 规模因子
            - 'HML': 价值因子
            - 'RMW': 盈利因子 (五因子)
            - 'CMA': 投资因子 (五因子)
            - 'RF': 无风险利率 (可选)

        Returns:
        --------
        results : dict, 包含alpha, beta, R2, t-stat等
        """
        # 对齐索引
        common_idx = portfolio_returns.index.intersection(factor_data.index)
        if len(common_idx) < 20:
            print("Warning: Insufficient overlapping observations")
            return {}

        y = portfolio_returns.loc[common_idx].values

        # 构建X
        if self.model_type == 'three_factor':
            required_cols = ['Mkt-RF', 'SMB', 'HML']
        else:
            required_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        # 检查是否存在
        available_cols = [c for c in required_cols if c in factor_data.columns]
        if len(available_cols) < len(required_cols):
            print(f"Missing factor columns: {set(required_cols) - set(available_cols)}")
            return {}

        X = factor_data.loc[common_idx, available_cols].values

        # 添加常数项
        X = sm.add_constant(X)

        # OLS回归
        model = sm.OLS(y, X)
        results = model.fit()

        # 提取结果
        self.alpha = results.params[0]
        self.beta = results.params[1:]
        self.r_squared = results.rsquared
        self.t_stats = results.tvalues
        self.p_values = results.pvalues
        self.resid = results.resid
        self.results = {
            'alpha': self.alpha,
            'beta': self.beta.to_dict() if hasattr(self.beta, 'to_dict') else dict(zip(available_cols, self.beta)),
            'r_squared': self.r_squared,
            'adj_r_squared': results.rsquared_adj,
            't_stats': self.t_stats.to_dict() if hasattr(self.t_stats, 'to_dict') else dict(
                zip(['const'] + available_cols, self.t_stats)),
            'p_values': self.p_values.to_dict() if hasattr(self.p_values, 'to_dict') else dict(
                zip(['const'] + available_cols, self.p_values)),
            'f_stat': results.fvalue,
            'f_p_value': results.f_pvalue,
            'n_obs': len(y),
            'model_type': self.model_type
        }

        # 计算信息比率
        self.information_ratio = self.alpha / (self.resid.std() + 1e-6) * np.sqrt(252)
        self.results['information_ratio'] = self.information_ratio

        # 计算年化alpha
        self.results['annualized_alpha'] = self.alpha * 252
        self.results['alpha_t_stat'] = self.t_stats[0] if len(self.t_stats) > 0 else None

        return self.results

    def fit_from_ff_data(self, portfolio_returns, ff_data_path=None):
        """
        从Fama-French数据文件加载因子并拟合

        Parameters:
        -----------
        portfolio_returns : Series
        ff_data_path : str, CSV文件路径（可选）
        """
        # 如果未提供路径，使用内置数据（示例）
        if ff_data_path is None:
            # 这里需要实际数据，我们构造一个示例
            # 实际使用时，可以从Kenneth French网站下载
            print("Please provide FF data file path")
            return {}

        ff_data = pd.read_csv(ff_data_path, index_col=0, parse_dates=True)
        return self.fit(portfolio_returns, ff_data)

    def summary(self):
        """打印回归结果摘要"""
        if not self.results:
            print("No results available, run fit() first")
            return

        print("\n" + "=" * 60)
        print(f"Fama-French {self.model_type} Regression Results")
        print("=" * 60)
        print(f"Alpha (daily): {self.alpha:.6f}")
        print(
            f"Annualized Alpha: {self.results['annualized_alpha']:.4f} ({self.results['annualized_alpha'] * 100:.2f}%)")
        print(f"Alpha t-stat: {self.results['alpha_t_stat']:.3f}")
        print(f"Information Ratio: {self.results['information_ratio']:.3f}")
        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Adjusted R-squared: {self.results['adj_r_squared']:.4f}")
        print(f"Number of observations: {self.results['n_obs']}")
        print("\nFactor Loadings:")
        for factor, beta in self.results['beta'].items():
            t_val = self.results['t_stats'].get(factor, 0)
            p_val = self.results['p_values'].get(factor, 0)
            print(f"  {factor}: {beta:.4f} (t={t_val:.3f}, p={p_val:.4f})")
        print(f"\nF-statistic: {self.results['f_stat']:.3f} (p={self.results['f_p_value']:.4f})")
        print("=" * 60)

    def plot_residuals(self, save_path=None):
        """绘制残差图"""
        try:
            import matplotlib.pyplot as plt
            if not hasattr(self, 'resid'):
                print("No residuals available")
                return

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 残差时序图
            axes[0, 0].plot(self.resid)
            axes[0, 0].axhline(y=0, color='r', linestyle='--')
            axes[0, 0].set_title('Residuals over Time')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Residual')

            # 残差直方图
            axes[0, 1].hist(self.resid, bins=30, edgecolor='black')
            axes[0, 1].set_title('Residual Distribution')
            axes[0, 1].set_xlabel('Residual')

            # Q-Q图
            stats.probplot(self.resid, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')

            # 残差自相关
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(self.resid, ax=axes[1, 1], lags=20)
            axes[1, 1].set_title('Autocorrelation of Residuals')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300)
            plt.show()

        except ImportError:
            print("matplotlib or statsmodels not installed")