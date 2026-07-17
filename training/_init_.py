# training/__init__.py
"""
训练模块：数据集、损失函数、训练器（含滚动窗口）
"""
from .dataset import (
    TimeSeriesDataset,
    PairwiseDataset,
    create_pairwise_sequences,
    create_train_val_test_datasets
)
from .loss import (
    RankingLoss,
    MultiTaskLoss,
    AdaptiveLossWeighting
)
from .trainer import (
    Trainer,
    RollingWindowTrainer
)

__all__ = [
    'TimeSeriesDataset',
    'PairwiseDataset',
    'create_pairwise_sequences',
    'create_train_val_test_datasets',
    'RankingLoss',
    'MultiTaskLoss',
    'AdaptiveLossWeighting',
    'Trainer',
    'RollingWindowTrainer'
]