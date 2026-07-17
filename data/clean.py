# data/clean.py
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class DataCleaner:
    """
    数据清洗类：
    - 去重
    - 缺失值处理（向前填充/插值）
    - 异常值检测与处理
    - 去极值（Winsorize）
    """

    def __init__(self, winsorize_limits=(0.01, 0.99), fillna_method='ffill'):
        self.winsorize_limits = winsorize_limits
        self.fillna_method = fillna_method

    def clean(self, df, price_cols=None):
        """
        主清洗函数
        """
        df = df.copy()

        # 1. 去重
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]

        # 2. 处理缺失值
        df = self._handle_missing(df)

        # 3. 去极值（对数值列，排除宏观列）
        macro_cols = ['bond_10y', 'social_financing', 'pmi', 'cpi', 'm2']
        if self.winsorize_limits is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in macro_cols]
            for col in numeric_cols:
                df[col] = self._winsorize(df[col])

        # 4. 检查价格列异常
        if price_cols is not None:
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].clip(lower=0)

        return df

    def _handle_missing(self, df):
        """处理缺失值"""
        if self.fillna_method == 'drop':
            return df.dropna()
        elif self.fillna_method == 'ffill':
            if 'stock' in df.index.names:
                return df.groupby(level='stock', group_keys=False).apply(
                    lambda group: group.ffill().bfill()  # 关键修改
                )
            else:
                return df.ffill().bfill()
        elif self.fillna_method == 'interpolate':
            if 'stock' in df.index.names:
                return df.groupby(level='stock', group_keys=False).apply(
                    lambda group: group.interpolate(method='time', limit_area='inside')
                )
            else:
                return df.interpolate(method='time', limit_area='inside')
        else:
            return df

    def _winsorize(self, series):
        """去极值"""
        lower = series.quantile(self.winsorize_limits[0])
        upper = series.quantile(self.winsorize_limits[1])
        return series.clip(lower=lower, upper=upper)

    def remove_outliers_zscore(self, df, threshold=3):
        """基于Z-score剔除异常值"""
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