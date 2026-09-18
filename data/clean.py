# data/clean.py
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class DataCleaner:
    """Clean each stock causally; never fill an observation from the future."""

    def __init__(self, winsorize_limits=None, fillna_method='ffill'):
        self.winsorize_limits = winsorize_limits
        self.fillna_method = fillna_method

    def clean(self, df, price_cols=None):
        """Remove duplicate observations and apply the configured causal missing-value policy."""
        df = df.copy()

        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]

        df = self._handle_missing(df)

        macro_cols = ['bond_10y', 'social_financing', 'pmi', 'cpi', 'm2']
        if self.winsorize_limits is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in macro_cols]
            for col in numeric_cols:
                df[col] = self._winsorize(df[col])

        if price_cols is not None:
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].clip(lower=0)

        return df

    def _handle_missing(self, df):
        """Forward-fill within each stock in chronological order."""
        if self.fillna_method == 'drop':
            return df.dropna()
        elif self.fillna_method == 'ffill':
            if 'stock' in df.index.names:
                return df.groupby(level='stock', group_keys=False).apply(
                    lambda group: group.sort_index().ffill()
                )
            else:
                return df.sort_index().ffill()
        elif self.fillna_method == 'interpolate':

            if 'stock' in df.index.names:
                return df.groupby(level='stock', group_keys=False).apply(
                    lambda group: group.sort_index().ffill()
                )
            else:
                return df.sort_index().ffill()
        else:
            return df

    def _winsorize(self, series):
        """Clip using expanding quantiles computed only from available history."""
        if 'stock' in series.index.names:
            return series.groupby(level='stock', group_keys=False).apply(
                self._winsorize_single_series
            )
        return self._winsorize_single_series(series.sort_index())

    def _winsorize_single_series(self, series):
        lower = series.expanding(min_periods=20).quantile(self.winsorize_limits[0])
        upper = series.expanding(min_periods=20).quantile(self.winsorize_limits[1])
        clipped = series.clip(lower=lower, upper=upper)
        return clipped.where(lower.notna() & upper.notna(), series)

    def remove_outliers_zscore(self, df, threshold=3):
        """Filter z-score outliers, then apply the configured missing-value policy."""
        macro_cols = ['bond_10y', 'social_financing', 'pmi', 'cpi', 'm2']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in macro_cols]

        for col in numeric_cols:
            zscore = np.abs(stats.zscore(df[col].dropna()))
            mask = zscore < threshold
            df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].where(
                zscore < threshold, other=np.nan
            )
        return self._handle_missing(df)
