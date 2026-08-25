# training/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集
    用于多任务学习：排序得分 + 波动率预测
    """

    def __init__(self, df, feature_cols, target_col='excess_ret_5d',
                 vol_col='future_vol_5d', seq_len=60,
                 mode='train', normalize=True, scaler=None,
                 sample_start=None, sample_end=None):
        """
        Parameters:
        -----------
        df : DataFrame with MultiIndex (date, stock)
        feature_cols : list, 特征列名
        target_col : str, 目标列（超额收益）
        vol_col : str, 波动率列
        seq_len : int, 序列长度
        mode : str, 'train', 'val', 'test'
        normalize : bool, 是否标准化
        """
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.vol_col = vol_col
        self.seq_len = seq_len
        self.mode = mode
        self.normalize = normalize
        self.sample_start = pd.Timestamp(sample_start) if sample_start is not None else None
        self.sample_end = pd.Timestamp(sample_end) if sample_end is not None else None

        # 标准化器
        self.scaler = scaler or StandardScaler()

        # 准备样本
        self.samples = self._prepare_samples()

        # 如果标准化，拟合scaler
        if normalize and len(self.samples) > 0:
            if mode == 'train':
                self._fit_scaler()
            elif scaler is None:
                raise ValueError("验证/测试集启用标准化时必须传入训练集 scaler")

    def _prepare_samples(self):
        """准备样本（带调试日志）"""
        samples = []
        stocks = self.df.index.get_level_values('stock').unique()

        print(f"  总股票数: {len(stocks)}")

        for stock in stocks:
            stock_data = self.df.xs(stock, level='stock').sort_index()
            print(f"    股票 {stock}: 数据长度 {len(stock_data)}")

            if len(stock_data) < self.seq_len + 5:
                print(f"      跳过: 数据长度 {len(stock_data)} < {self.seq_len + 5}")
                continue

            features = stock_data[self.feature_cols].values
            targets = stock_data[self.target_col].values if self.target_col in stock_data else None
            vols = stock_data[self.vol_col].values if self.vol_col in stock_data else None

            # 检查是否全是 NaN
            if np.isnan(features).all():
                print(f"      跳过: 所有特征全为 NaN")
                continue

            sample_count = 0
            for i in range(len(stock_data) - self.seq_len):
                X = features[i:i + self.seq_len]
                y_rank = targets[i + self.seq_len] if targets is not None else 0
                y_vol = vols[i + self.seq_len] if vols is not None else 0
                sample_date = pd.Timestamp(stock_data.index[i + self.seq_len])

                if self.sample_start is not None and sample_date < self.sample_start:
                    continue
                if self.sample_end is not None and sample_date > self.sample_end:
                    continue

                if np.isnan(y_rank) or np.isnan(y_vol):
                    continue
                if not np.isfinite(X).all():
                    continue

                samples.append({
                    'X': X,
                    'y_rank': y_rank,
                    'y_vol': y_vol,
                    'stock': stock,
                    'date': sample_date
                })
                sample_count += 1

            print(f"      生成样本数: {sample_count}")

        print(f"  总样本数: {len(samples)}")
        return samples

    def _fit_scaler(self):
        """拟合标准化器"""
        if len(self.samples) == 0:
            return

        # 收集所有特征
        all_features = np.vstack([s['X'] for s in self.samples])
        self.scaler.fit(all_features)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        X = sample['X'].astype(np.float32)

        # 标准化
        if self.normalize:
            X = self.scaler.transform(X)

        y_rank = np.array([sample['y_rank']], dtype=np.float32)
        y_vol = np.array([sample['y_vol']], dtype=np.float32)

        return {
            'x': torch.FloatTensor(X),
            'rank_target': torch.FloatTensor(y_rank),
            'vol_target': torch.FloatTensor(y_vol),
            'stock': sample['stock'],
            'date': sample['date']
        }


class PairwiseDataset(Dataset):
    """
    配对数据集（用于排序学习）
    从同一交易日采样股票对
    """

    def __init__(self, df, feature_cols, target_col='excess_ret_5d',
                 seq_len=60, max_pairs_per_day=1000):
        """
        Parameters:
        -----------
        df : DataFrame with MultiIndex (date, stock)
        feature_cols : list, 特征列名
        target_col : str, 目标列
        seq_len : int, 序列长度
        max_pairs_per_day : int, 每天最多采样对数
        """
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.max_pairs_per_day = max_pairs_per_day

        self.pairs = self._prepare_pairs()

        # 标准化器
        self.scaler = StandardScaler()
        self._fit_scaler()

    def _prepare_pairs(self):
        """准备配对样本"""
        pairs = []

        # 按日期分组
        dates = self.df.index.get_level_values('date').unique()

        for date in tqdm(dates, desc="Creating pairs"):
            # 获取当日所有股票
            date_data = self.df.xs(date, level='date')

            if len(date_data) < 2:
                continue

            # 获取每个股票的特征和目标
            stocks = date_data.index.get_level_values('stock').unique()
            targets = date_data[self.target_col].values

            # 只取有足够历史数据的股票
            valid_stocks = []
            for stock in stocks:
                stock_data = self.df.xs(stock, level='stock').sort_index()
                if len(stock_data) >= self.seq_len:
                    valid_stocks.append(stock)

            if len(valid_stocks) < 2:
                continue

            # 采样配对
            n_pairs = min(self.max_pairs_per_day, len(valid_stocks) * 10)

            for _ in range(n_pairs):
                # 随机选择两只股票
                idx1, idx2 = np.random.choice(len(valid_stocks), 2, replace=False)
                stock1 = valid_stocks[idx1]
                stock2 = valid_stocks[idx2]

                # 获取目标值
                target1 = date_data.xs(stock1, level='stock')[self.target_col].values[0]
                target2 = date_data.xs(stock2, level='stock')[self.target_col].values[0]

                # 确定配对标签（1表示股票1优于股票2）
                label = 1 if target1 > target2 else 0

                # 获取历史序列
                seq1 = self._get_sequence(stock1, date, self.seq_len)
                seq2 = self._get_sequence(stock2, date, self.seq_len)

                if seq1 is not None and seq2 is not None:
                    pairs.append({
                        'X1': seq1,
                        'X2': seq2,
                        'y': label,
                        'date': date
                    })

        return pairs

    def _get_sequence(self, stock, end_date, seq_len):
        """获取股票的历史序列"""
        stock_data = self.df.xs(stock, level='stock').sort_index()

        # 找到end_date的位置
        idx = stock_data.index.get_loc(end_date)

        if idx < seq_len:
            return None

        # 取前seq_len天的特征
        seq = stock_data[self.feature_cols].iloc[idx - seq_len:idx].values

        if np.isnan(seq).any():
            return None

        return seq.astype(np.float32)

    def _fit_scaler(self):
        """拟合标准化器"""
        if len(self.pairs) == 0:
            return

        all_features = []
        for pair in self.pairs:
            all_features.append(pair['X1'])
            all_features.append(pair['X2'])

        all_features = np.vstack(all_features)
        self.scaler.fit(all_features)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        X1 = self.scaler.transform(pair['X1'])
        X2 = self.scaler.transform(pair['X2'])
        y = np.array([pair['y']], dtype=np.float32)

        return {
            'x1': torch.FloatTensor(X1),
            'x2': torch.FloatTensor(X2),
            'y': torch.FloatTensor(y),
            'date': pair['date']
        }


def create_pairwise_sequences(df, feature_cols, target_col='excess_ret_5d',
                              seq_len=60, max_pairs_per_day=1000):
    """
    创建配对序列数据（便捷函数）

    Returns:
    --------
    PairwiseDataset
    """
    return PairwiseDataset(
        df, feature_cols, target_col, seq_len, max_pairs_per_day
    )


def create_train_val_test_datasets(df, feature_cols, target_col='excess_ret_5d',
                                   vol_col='future_vol_5d', seq_len=60,
                                   train_ratio=0.7, val_ratio=0.15,
                                   embargo_days=5, normalize=False):
    """
    创建训练/验证/测试数据集

    Parameters:
    -----------
    df : DataFrame
    feature_cols : list
    target_col : str
    vol_col : str
    seq_len : int
    train_ratio : float
    val_ratio : float

    Returns:
    --------
    train_dataset, val_dataset, test_dataset
    """
    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(df.index.get_level_values('date').unique()))
    )
    n_dates = len(dates)

    if n_dates < seq_len + embargo_days * 2 + 20:
        raise ValueError("可用交易日不足，无法创建可靠的时间顺序训练/验证/测试集")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio 与 val_ratio 必须为正，且两者之和小于 1")

    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    train_sample_end_idx = train_end - embargo_days - 1
    val_sample_end_idx = val_end - embargo_days - 1
    if train_sample_end_idx < 0 or val_sample_end_idx < train_end:
        raise ValueError("embargo_days 过大，时间切分后没有足够样本")

    train_dataset = TimeSeriesDataset(
        df, feature_cols, target_col, vol_col, seq_len, mode='train',
        normalize=normalize, sample_end=dates[train_sample_end_idx]
    )

    val_dataset = TimeSeriesDataset(
        df, feature_cols, target_col, vol_col, seq_len, mode='val',
        normalize=normalize, scaler=train_dataset.scaler if normalize else None,
        sample_start=dates[train_end], sample_end=dates[val_sample_end_idx]
    )

    test_dataset = TimeSeriesDataset(
        df, feature_cols, target_col, vol_col, seq_len, mode='test',
        normalize=normalize, scaler=train_dataset.scaler if normalize else None,
        sample_start=dates[val_end]
    )

    split_metadata = {
        'train_end': dates[train_sample_end_idx],
        'validation_start': dates[train_end],
        'validation_end': dates[val_sample_end_idx],
        'test_start': dates[val_end],
        'embargo_days': embargo_days,
    }
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.split_metadata = split_metadata

    return train_dataset, val_dataset, test_dataset
