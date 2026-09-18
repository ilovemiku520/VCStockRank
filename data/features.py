# data/features.py
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class FactorBuilder:
    def __init__(self, winsorize=True, standardize=True):
        self.winsorize = winsorize
        self.standardize = standardize

    def build_all_factors(self, df):
        df = df.copy()
        df = self._add_price_factors(df)
        df = self._add_volume_factors(df)
        df = self._add_technical_factors(df)
        df = self._add_decomposition_factors(df)
        df = self._add_dynamic_volatility_factors(df)
        df = self._add_statistical_factors(df)
        return df

    def _add_price_factors(self, df):
        for period in [1, 5, 10, 20]:
            df[f'ret_{period}d'] = df['close'].pct_change(period)
        df['ret_excess_20d'] = df['close'] / df['close'].rolling(20).mean() - 1
        df['price_position_20d'] = (df['close'] - df['close'].rolling(20).min()) / \
                                   (df['close'].rolling(20).max() - df['close'].rolling(20).min() + 1e-6)
        df['max_drawdown_20d'] = df['close'].rolling(20).apply(
            lambda x: (x.max() - x.iloc[-1]) / x.max() if x.max() > 0 else 0
        )
        return df

    def _add_volume_factors(self, df):
        df['turnover_ma5'] = df['turnover'].rolling(5).mean()
        df['turnover_ratio'] = df['turnover'] / df['turnover_ma5']
        df['amihud'] = (df['close'].pct_change().abs() * 1e6) / (df['volume'] + 1e-6)
        df['amihud_ma20'] = df['amihud'].rolling(20).mean()
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume_ma5'] / df['volume_ma20']
        df['money_flow'] = df['volume'] * (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
        df['money_flow_ma5'] = df['money_flow'].rolling(5).mean()
        return df

    def _add_technical_factors(self, df):
        df['rsi_14'] = self._compute_rsi(df['close'], 14)
        df['rsi_28'] = self._compute_rsi(df['close'], 28)
        df['macd'], df['macd_signal'], df['macd_hist'] = self._compute_macd(df['close'])
        df['bb_position'] = self._compute_bollinger_bands(df['close'])
        df['atr_14'] = self._compute_atr(df, 14)
        df['atr_ratio'] = df['atr_14'] / df['close']
        df['kdj_k'], df['kdj_d'], df['kdj_j'] = self._compute_kdj(df)
        return df

    def _add_decomposition_factors(self, df):

        close = df['close'].ffill()
        trend = close.rolling(20, min_periods=5).mean()
        residual = close - trend
        rolling_scale = close.rolling(60, min_periods=20).std()
        df['trend_strength'] = trend / (close + 1e-6)
        df['seasonal_strength'] = residual.rolling(7, min_periods=5).std() / (rolling_scale + 1e-6)
        df['residual_strength'] = residual.rolling(20, min_periods=10).std() / (rolling_scale + 1e-6)
        df['trend_change'] = trend.pct_change()
        df['cycle_ratio'] = (residual.abs() / (close.abs() + 1e-6)).rolling(20, min_periods=5).mean()
        df['forecast_residual'] = close - close.rolling(5, min_periods=3).mean().shift(1)
        return df

    def _add_dynamic_volatility_factors(self, df):
        returns = df['close'].pct_change()
        for period in [5, 10, 20, 60]:
            df[f'rv_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
        df['vol_term_structure'] = df['rv_20d'] / (df['rv_5d'] + 1e-6)
        df['vol_term_slope'] = df['rv_60d'] - df['rv_20d']
        df['vol_change_5d'] = df['rv_20d'] / df['rv_20d'].shift(5) - 1
        df['vol_change_20d'] = df['rv_20d'] / df['rv_20d'].shift(20) - 1
        df['vol_clustering'] = returns.rolling(20).apply(
            lambda x: (x.abs() > x.abs().mean() + x.abs().std()).mean() if len(x) == 20 else np.nan
        )
        pos_vol = returns.where(returns > 0).rolling(20, min_periods=5).std()
        neg_vol = returns.where(returns < 0).rolling(20, min_periods=5).std()
        df['vol_asymmetry'] = pos_vol / (neg_vol + 1e-6)
        df['var_95'] = returns.rolling(20).quantile(0.05)
        df['cvar_95'] = returns.rolling(20).apply(
            lambda x: x[x < x.quantile(0.05)].mean() if len(x[x < x.quantile(0.05)]) > 0 else np.nan
        )
        df['hurst'] = df['close'].rolling(100, min_periods=100).apply(
            lambda x: self._compute_hurst(np.asarray(x), max_lag=50), raw=False
        )
        return df

    def _add_statistical_factors(self, df):
        df['skew_20d'] = df['close'].rolling(20).apply(lambda x: stats.skew(x) if len(x) == 20 else np.nan)
        df['kurt_20d'] = df['close'].rolling(20).apply(lambda x: stats.kurtosis(x) if len(x) == 20 else np.nan)
        df['autocorr_1'] = df['close'].rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) == 20 else np.nan
        )
        df['autocorr_5'] = df['close'].rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) == 20 else np.nan
        )
        df['rolling_sharpe_20d'] = df['ret_1d'].rolling(20).mean() / (df['ret_1d'].rolling(20).std() + 1e-6)
        return df

    @staticmethod
    def _compute_rsi(price, period):
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_macd(price, fast=12, slow=26, signal=9):
        exp1 = price.ewm(span=fast, adjust=False).mean()
        exp2 = price.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    @staticmethod
    def _compute_bollinger_bands(price, period=20, std_dev=2):
        ma = price.rolling(period).mean()
        std = price.rolling(period).std()
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        position = (price - lower) / (upper - lower + 1e-6)
        return position

    @staticmethod
    def _compute_atr(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    @staticmethod
    def _compute_kdj(df, n=9, m1=3, m2=3):
        low_min = df['low'].rolling(n).min()
        high_max = df['high'].rolling(n).max()
        rsv = (df['close'] - low_min) / (high_max - low_min + 1e-6) * 100
        k = rsv.ewm(span=m1, adjust=False).mean()
        d = k.ewm(span=m2, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    @staticmethod
    def _compute_hurst(series, max_lag=100):
        if len(series) < max_lag:
            return 0.5
        lags = range(10, min(max_lag, len(series) // 2), 5)
        tau = []
        for lag in lags:
            diff = np.log(series[lag:]) - np.log(series[:-lag])
            var = np.var(diff)
            tau.append(var)
        if len(tau) < 2:
            return 0.5
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0] / 2
        return np.clip(hurst, 0, 1)

    def process_factors(self, df, winsorize=True, standardize=True):
        """Clip and scale features without modifying raw prices or forward labels."""
        df = df.copy()

        protected_cols = [
            'close', 'open', 'high', 'low', 'volume', 'amount', 'turnover',
            'ret_1d', 'future_ret_5d', 'future_vol_5d', 'market_ret', 'excess_ret_5d'
        ]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        factor_cols = [col for col in numeric_cols if col not in protected_cols]

        values = df[factor_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        if 'date' in df.index.names:
            groups = values.groupby(level='date')
            if winsorize:
                values = values.clip(groups.transform('quantile', q=.01),
                                     groups.transform('quantile', q=.99))
            if standardize:
                groups = values.groupby(level='date')
                std = groups.transform('std').replace(0, np.nan)
                values = (values - groups.transform('mean')) / std
        else:
            if winsorize:
                lower = values.expanding(min_periods=20).quantile(.01)
                upper = values.expanding(min_periods=20).quantile(.99)
                values = values.clip(lower, upper)
            if standardize:
                mean = values.expanding(min_periods=2).mean()
                std = values.expanding(min_periods=2).std().replace(0, np.nan)
                values = (values - mean) / std
        df[factor_cols] = values
        return df

    def create_factors_panel(self, daily_data):
        """Build per-stock features, then normalize within each trading date."""
        all_factors = []
        stocks = daily_data.index.get_level_values('stock').unique()

        for stock in tqdm(stocks, desc='Building causal features'):
            stock_data = daily_data.xs(stock, level='stock').sort_index()
            factors = self.build_all_factors(stock_data)
            factors['stock'] = stock
            factors = factors.reset_index()
            all_factors.append(factors)

        panel = pd.concat(all_factors, ignore_index=True)
        panel = panel.set_index(['date', 'stock'])
        panel = panel.sort_index()
        panel = self.process_factors(panel, winsorize=self.winsorize, standardize=self.standardize)
        return panel
