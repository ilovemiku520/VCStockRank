# evaluation/__init__.py
"""
评估模块：因子分析、FF回归、SHAP解释、注意力可视化
"""
from .factor_analysis import FactorAnalyzer
from .ff_regression import FamaFrenchRegression
from .shap_analysis import SHAPAnalyzer
from .attention_vis import AttentionVisualizer

__all__ = [
    'FactorAnalyzer',
    'FamaFrenchRegression',
    'SHAPAnalyzer',
    'AttentionVisualizer'
]