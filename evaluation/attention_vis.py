# evaluation/attention_vis.py
import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings('ignore')


class AttentionVisualizer:
    """
    注意力可视化工具
    用于可视化VCformer的注意力权重
    """

    def __init__(self, model, seq_len=60, feature_names=None):
        """
        Parameters:
        -----------
        model : MultiTaskVCformerTPA model
        seq_len : int, 序列长度
        feature_names : list, 特征名称
        """
        self.model = model
        self.seq_len = seq_len
        self.feature_names = feature_names
        self.attention_weights = None

    def extract_attention(self, X, layer_idx=-1):
        """
        提取注意力权重

        Parameters:
        -----------
        X : tensor or array, (batch, seq_len, feat_dim)
        layer_idx : int, 指定提取哪一层的注意力（-1表示最后一层）

        Returns:
        --------
        attention_weights : list of tensors
        """
        self.model.eval()

        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)

        # 前向传播，开启return_attention
        if hasattr(self.model, 'forward'):
            outputs = self.model(X, return_attention=True)
            if 'vc_attention' in outputs:
                attention = outputs['vc_attention']
                if isinstance(attention, list) and len(attention) > 0:
                    # 取指定层
                    if layer_idx == -1:
                        layer_idx = len(attention) - 1
                    self.attention_weights = attention[layer_idx]
                    return self.attention_weights

        # 如果没有vc_attention，尝试从vcformer获取
        if hasattr(self.model, 'vcformer'):
            # 单独执行vcformer
            with torch.no_grad():
                # 先经过嵌入和预处理（需匹配模型流程）
                # 简化：直接调用vcformer
                x_embedded = self.model.embed(X)
                # 进行分解等其他步骤（这里简化）
                x = self.model.vcformer(x_embedded)
                attn = self.model.vcformer.get_attention_weights()
                if attn:
                    self.attention_weights = attn[layer_idx] if layer_idx != -1 else attn[-1]
                    return self.attention_weights

        print("Could not extract attention weights")
        return None

    def visualize_attention_matrix(self, attn_weights, save_path=None):
        """
        可视化注意力矩阵（变量-变量）

        Parameters:
        -----------
        attn_weights : tensor, (n_heads, seq_len, feat_dim) or (seq_len, feat_dim)
        save_path : str, 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 转换为numpy
            if torch.is_tensor(attn_weights):
                attn_weights = attn_weights.cpu().detach().numpy()

            # 如果有多个头，取平均
            if attn_weights.ndim == 4:  # (batch, n_heads, seq_len, feat_dim)
                attn_weights = attn_weights.mean(axis=0)  # 平均batch和头
            elif attn_weights.ndim == 3:  # (n_heads, seq_len, feat_dim)
                attn_weights = attn_weights.mean(axis=0)  # 平均头

            # 现在应该是 (seq_len, feat_dim) 或 (feat_dim, feat_dim)
            if attn_weights.shape[0] == attn_weights.shape[1]:
                # 对称矩阵，可能是变量-变量
                matrix = attn_weights
                labels = self.feature_names[:matrix.shape[0]] if self.feature_names else None
                title = "Variable-Variable Attention"
            else:
                # 可能是序列-变量
                # 取平均序列维度
                matrix = attn_weights.mean(axis=0, keepdims=True)
                # 转换为方阵近似
                if matrix.shape[0] < matrix.shape[1]:
                    matrix = matrix.T @ matrix
                else:
                    matrix = matrix @ matrix.T
                labels = None
                title = "Attention Matrix"

            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, annot=False, cmap='RdBu_r',
                        xticklabels=labels, yticklabels=labels,
                        center=0, square=True)
            plt.title(title)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            print("matplotlib not installed")

    def visualize_feature_attention(self, attn_weights, save_path=None):
        """
        可视化特征级别的注意力权重（条形图）
        """
        try:
            import matplotlib.pyplot as plt

            if torch.is_tensor(attn_weights):
                attn_weights = attn_weights.cpu().detach().numpy()

            # 如果有多个头，取平均
            if attn_weights.ndim >= 3:
                attn_weights = attn_weights.mean(axis=(0, 1)) if attn_weights.ndim == 4 else attn_weights.mean(axis=0)

            # 假设最后一维是特征
            if attn_weights.ndim == 1:
                importance = attn_weights
            elif attn_weights.ndim == 2:
                importance = attn_weights.mean(axis=0)
            else:
                importance = attn_weights.flatten()

            # 截取特征数
            n_features = len(self.feature_names) if self.feature_names else len(importance)
            importance = importance[:n_features]

            plt.figure(figsize=(12, 6))
            plt.bar(range(len(importance)), importance)
            if self.feature_names:
                plt.xticks(range(len(importance)), self.feature_names[:len(importance)], rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Attention Weight')
            plt.title('Feature Attention Weights')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            print("matplotlib not installed")

    def visualize_temporal_attention(self, attn_weights, save_path=None):
        """
        可视化时间维度的注意力
        """
        try:
            import matplotlib.pyplot as plt

            if torch.is_tensor(attn_weights):
                attn_weights = attn_weights.cpu().detach().numpy()

            # 简化：取平均
            if attn_weights.ndim >= 3:
                # 取序列维度平均
                if attn_weights.shape[1] == self.seq_len:
                    temporal = attn_weights.mean(axis=(0, 2))  # 平均head和feature
                else:
                    temporal = attn_weights.mean(axis=(0, 1))
            else:
                temporal = attn_weights

            # 确保长度为seq_len
            if len(temporal) > self.seq_len:
                temporal = temporal[:self.seq_len]
            elif len(temporal) < self.seq_len:
                temporal = np.pad(temporal, (0, self.seq_len - len(temporal)), 'constant')

            plt.figure(figsize=(12, 4))
            plt.plot(range(len(temporal)), temporal, marker='o')
            plt.xlabel('Time Steps')
            plt.ylabel('Attention Weight')
            plt.title('Temporal Attention Distribution')
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            print("matplotlib not installed")

    def visualize_cnn_filters(self, save_path=None):
        """
        可视化TPA的CNN滤波器（DFT）
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.fft import fft

            # 获取TPA模块
            if not hasattr(self.model, 'tpa'):
                print("Model does not have TPA module")
                return

            tpa = self.model.tpa
            if not hasattr(tpa, 'multi_scale_cnn'):
                print("TPA does not have multi_scale_cnn")
                return

            # 获取卷积核权重
            convs = tpa.multi_scale_cnn.convs
            fig, axes = plt.subplots(len(convs), 1, figsize=(12, 4 * len(convs)))
            if len(convs) == 1:
                axes = [axes]

            for i, conv in enumerate(convs):
                # 获取第一个卷积层的权重
                weight = conv[0].weight.data.cpu().numpy()  # (out_channels, in_channels, kernel_size)
                # 取第一个输出通道和输入通道
                kernel = weight[0, 0, :].flatten()

                # 计算DFT
                fft_vals = np.abs(fft(kernel))
                freqs = np.fft.fftfreq(len(kernel))

                axes[i].plot(freqs[:len(freqs) // 2], fft_vals[:len(fft_vals) // 2])
                axes[i].set_title(f'Filter {i + 1} (kernel size {conv[0].kernel_size[0]})')
                axes[i].set_xlabel('Frequency')
                axes[i].set_ylabel('Magnitude')
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            print("matplotlib not installed")