import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleCNN(nn.Module):
    """
    多尺度卷积模块（TPA风格）
    使用不同大小卷积核提取多尺度时序模式
    """

    def __init__(self, input_dim, filters=[3, 6, 12, 24], channels=32):
        super().__init__()
        self.input_dim = input_dim
        self.filters = filters
        self.channels = channels

        # 为每个特征创建多尺度卷积
        self.convs = nn.ModuleList()
        for filter_size in filters:
            conv = nn.Sequential(
                nn.Conv1d(input_dim, channels, kernel_size=filter_size, padding=filter_size // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=filter_size, padding=filter_size // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU()
            )
            self.convs.append(conv)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(channels * len(filters), channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        Returns: (batch, input_dim, channels * len(filters))
        """
        # 转置为 (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)

        # 多尺度卷积
        multi_scale_features = []
        for conv in self.convs:
            conv_out = conv(x)  # (batch, channels, seq_len)
            # 全局池化
            pooled = conv_out.mean(dim=-1)  # (batch, channels)
            multi_scale_features.append(pooled)

        # 融合特征
        fused = torch.cat(multi_scale_features, dim=-1)  # (batch, channels * len(filters))
        fused = self.fusion(fused)  # (batch, channels)

        return fused


class TPAAttention(nn.Module):
    """
    TPA风格的注意力机制
    使用Sigmoid激活进行特征选择
    """

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 注意力网络
        self.attn_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, features, context=None):
        """
        features: (batch, feature_dim)
        context: (batch, feature_dim) 可选上下文
        Returns:
            weighted_features: (batch, feature_dim)
            attention_weights: (batch, feature_dim)
        """
        if context is not None:
            # 使用上下文计算注意力权重
            combined = features * context
        else:
            combined = features

        # 计算注意力权重
        attention_weights = self.attn_net(combined)  # (batch, feature_dim)

        # 加权特征
        weighted_features = features * attention_weights

        return weighted_features, attention_weights


class TPAModule(nn.Module):
    """
    完整的TPA模块
    多尺度卷积 + 注意力聚合
    """

    def __init__(self, input_dim, filters=[3, 6, 12, 24], channels=32):
        super().__init__()
        self.input_dim = input_dim
        self.channels = channels

        # 多尺度CNN
        self.multi_scale_cnn = MultiScaleCNN(input_dim, filters, channels)

        # TPA注意力
        self.tpa_attention = TPAAttention(channels, channels * 2)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, input_dim)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        Returns:
            aggregated: (batch, input_dim)
            attention_weights: (batch, channels)
        """
        # 多尺度特征提取
        features = self.multi_scale_cnn(x)  # (batch, channels)

        # TPA注意力
        weighted_features, attention_weights = self.tpa_attention(features)

        # 投影到原始维度
        aggregated = self.output_proj(weighted_features)  # (batch, input_dim)

        return aggregated, attention_weights

    def get_filter_responses(self, x):
        """获取各滤波器的响应（用于可视化）"""
        x = x.permute(0, 2, 1)

        responses = []
        for conv in self.multi_scale_cnn.convs:
            conv_out = conv(x)
            responses.append(conv_out.mean(dim=-1))

        return torch.stack(responses, dim=-1)  # (batch, channels, n_filters)