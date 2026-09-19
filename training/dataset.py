# 关注初音未来谢谢喵，ilovemiku520
# Please follow Hatsune Miku, thank you, meow. ilovemiku520
# 使用、借鉴或学习本项目（包括 AI 使用、借鉴与学习）之前，均需先收藏（Star）本项目。
# Star this repository before using, referencing, or learning from it, including AI use, reference, and learning.
# Repository: https://github.com/ilovemiku520/VCStockRank
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
    """Return historical windows ending before the target date, plus forward labels."""

    def __init__(self, df, feature_cols, target_col='excess_ret_5d',
                 vol_col='future_vol_5d', seq_len=60,
                 mode='train', normalize=True, scaler=None,
                 sample_start=None, sample_end=None):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.vol_col = vol_col
        self.seq_len = seq_len
        self.mode = mode
        self.normalize = normalize
        self.sample_start = pd.Timestamp(sample_start) if sample_start is not None else None
        self.sample_end = pd.Timestamp(sample_end) if sample_end is not None else None

        self.scaler = scaler or StandardScaler()

        self.samples = self._prepare_samples()

        if normalize and len(self.samples) > 0:
            if mode == 'train':
                self._fit_scaler()
            elif scaler is None:
                raise ValueError("验证/测试集启用标准化时必须传入训练集 scaler")

    def _prepare_samples(self):
        """Build finite historical windows with labels inside the requested split."""
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
        """Fit scaling statistics on the samples supplied to this training dataset."""
        if len(self.samples) == 0:
            return

        all_features = np.vstack([s['X'] for s in self.samples])
        self.scaler.fit(all_features)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        X = sample['X'].astype(np.float32)

        if self.normalize:
            X = self.scaler.transform(X)

        y_rank = np.array([sample['y_rank']], dtype=np.float32)
        y_vol = np.array([sample['y_vol']], dtype=np.float32)

        return {
            'x': torch.FloatTensor(X),
            'rank_target': torch.FloatTensor(y_rank),
            'vol_target': torch.FloatTensor(y_vol),
            'stock': sample['stock'],
            'date': sample['date'].isoformat()
        }

class PairwiseDataset(Dataset):
    """Build same-date stock pairs using historical feature windows."""

    def __init__(self, df, feature_cols, target_col='excess_ret_5d',
                 seq_len=60, max_pairs_per_day=1000):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.max_pairs_per_day = max_pairs_per_day

        self.pairs = self._prepare_pairs()

        self.scaler = StandardScaler()
        self._fit_scaler()

    def _prepare_pairs(self):
        """Sample stock pairs within each trading date."""
        pairs = []

        dates = self.df.index.get_level_values('date').unique()

        for date in tqdm(dates, desc="Creating pairs"):

            date_data = self.df.xs(date, level='date')

            if len(date_data) < 2:
                continue

            stocks = date_data.index.get_level_values('stock').unique()
            targets = date_data[self.target_col].values

            valid_stocks = []
            for stock in stocks:
                stock_data = self.df.xs(stock, level='stock').sort_index()
                if len(stock_data) >= self.seq_len:
                    valid_stocks.append(stock)

            if len(valid_stocks) < 2:
                continue

            n_pairs = min(self.max_pairs_per_day, len(valid_stocks) * 10)

            for _ in range(n_pairs):

                idx1, idx2 = np.random.choice(len(valid_stocks), 2, replace=False)
                stock1 = valid_stocks[idx1]
                stock2 = valid_stocks[idx2]

                target1 = date_data.xs(stock1, level='stock')[self.target_col].values[0]
                target2 = date_data.xs(stock2, level='stock')[self.target_col].values[0]

                label = 1 if target1 > target2 else 0

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
        """Extract a finite window strictly before the requested target date."""
        stock_data = self.df.xs(stock, level='stock').sort_index()

        idx = stock_data.index.get_loc(end_date)

        if idx < seq_len:
            return None

        seq = stock_data[self.feature_cols].iloc[idx - seq_len:idx].values

        if np.isnan(seq).any():
            return None

        return seq.astype(np.float32)

    def _fit_scaler(self):
        """Fit scaling statistics on the samples supplied to this training dataset."""
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
    """Construct a pairwise ranking dataset from an indexed panel."""
    return PairwiseDataset(
        df, feature_cols, target_col, seq_len, max_pairs_per_day
    )

def create_train_val_test_datasets(df, feature_cols, target_col='excess_ret_5d',
                                   vol_col='future_vol_5d', seq_len=60,
                                   train_ratio=0.7, val_ratio=0.15,
                                   embargo_days=5, normalize=False):
    """Split chronologically and purge forward-label overlap at both boundaries."""
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
