import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture


class MarketRegimeDetector(nn.Module):
    """
    市场状态检测器（创新）
    使用隐马尔可夫模型 + 深度学习的混合方法
    """

    def __init__(self, feature_dim, hidden_dim=64, n_regimes=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes

        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 状态转移网络
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes * n_regimes)
        )

        # 发射网络
        self.emission_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes)
        )

        # 状态分类器
        self.classifier = nn.Linear(hidden_dim, n_regimes)

        # 每个状态的特征分布参数
        self.regime_params = nn.ParameterDict({
            'means': nn.Parameter(torch.randn(n_regimes, feature_dim)),
            'vars': nn.Parameter(torch.ones(n_regimes, feature_dim))
        })

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        Returns:
            regime_probs: 状态概率 (batch, n_regimes)
            transition_matrix: 转移矩阵 (n_regimes, n_regimes)
            regime_features: 每个状态的特征
        """
        # 提取特征
        features = self.feature_extractor(x.mean(dim=1))  # (batch, hidden)

        # 计算状态概率
        regime_logits = self.classifier(features)
        regime_probs = F.softmax(regime_logits, dim=-1)

        # 计算转移矩阵
        transition_logits = self.transition_net(features)
        transition_matrix = F.softmax(
            transition_logits.view(-1, self.n_regimes, self.n_regimes),
            dim=-1
        )

        # 计算每个状态的期望特征
        regime_features = []
        for i in range(self.n_regimes):
            weight = regime_probs[:, i:i + 1]
            weighted_feature = weight * x.mean(dim=1)
            regime_features.append(weighted_feature)

        return {
            'regime_probs': regime_probs,
            'transition_matrix': transition_matrix,
            'regime_features': torch.stack(regime_features, dim=1),
            'regime_labels': regime_logits.argmax(dim=-1)
        }

    def compute_regime_loss(self, returns, regime_probs):
        """计算状态损失"""
        # 状态收益分布
        weighted_returns = returns.unsqueeze(1) * regime_probs.unsqueeze(2)

        # 每个状态的收益统计
        regime_means = weighted_returns.mean(dim=0, keepdim=True)
        regime_vars = weighted_returns.var(dim=0, keepdim=True)

        # 最大化状态间的区分度
        between_regime_var = regime_means.var(dim=1)

        # 最小化状态内的噪声
        within_regime_var = regime_vars.mean(dim=1)

        # 返回正则化损失
        return -between_regime_var + within_regime_var


class AdaptiveWeighting(nn.Module):
    """
    自适应加权模块（创新）
    根据市场状态动态调整特征权重
    """

    def __init__(self, feature_dim, n_regimes=3):
        super().__init__()
        self.n_regimes = n_regimes

        # 每个状态的权重网络
        self.regime_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(n_regimes)
        ])

    def forward(self, features, regime_probs):
        """
        features: (batch, seq_len, feature_dim)
        regime_probs: (batch, n_regimes)
        Returns:
            weighted_features: 加权后的特征
            attention_weights: 特征权重
        """
        batch_size = features.shape[0]
        seq_len = features.shape[1]
        feat_dim = features.shape[2]

        # 计算每个状态的特征权重
        weights = torch.zeros(batch_size, seq_len, feat_dim).to(features.device)

        for i, weight_net in enumerate(self.regime_weights):
            # 计算该状态下的权重
            w = weight_net(features)  # (batch, seq_len, 1)
            # 用状态概率加权
            regime_weight = regime_probs[:, i:i + 1, None]  # (batch, 1, 1)
            weights += w * regime_weight

        # 应用权重
        weighted_features = features * weights

        # 注意力权重（用于可视化）
        attention_weights = weights.mean(dim=-1)  # (batch, seq_len)

        return weighted_features, attention_weights