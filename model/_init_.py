# model/__init__.py
"""
模型模块：VCformer、TPA、时序分解、市场状态检测、多任务模型
"""
from .vcformer import (
    PositionalEncoding,
    VariableCentricAttention,
    VariableCentricTransformer
)
from .tpa_module import (
    MultiScaleCNN,
    TPAAttention,
    TPAModule
)
from .decomposition import (
    TimeSeriesDecomposition,
    WaveletDecomposition,
    SpectralAttention
)
from .regime_detection import (
    MarketRegimeDetector,
    AdaptiveWeighting
)
from .multitask import MultiTaskVCformerTPA

__all__ = [
    'PositionalEncoding',
    'VariableCentricAttention',
    'VariableCentricTransformer',
    'MultiScaleCNN',
    'TPAAttention',
    'TPAModule',
    'TimeSeriesDecomposition',
    'WaveletDecomposition',
    'SpectralAttention',
    'MarketRegimeDetector',
    'AdaptiveWeighting',
    'MultiTaskVCformerTPA'
]