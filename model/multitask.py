import torch
import torch.nn as nn
import torch.nn.functional as F
from .vcformer import VariableCentricTransformer
from .tpa_module import TPAModule
from .decomposition import TimeSeriesDecomposition, SpectralAttention
from .regime_detection import MarketRegimeDetector, AdaptiveWeighting


class MultiTaskVCformerTPA(nn.Module):
    """
    多任务 VCformer + TPA 模型
    包含时序分解、市场状态检测、排序学习、波动率预测
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. 特征嵌入
        self.embed = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT)
        )

        # 2. 时序分解（创新）
        self.decomposition = TimeSeriesDecomposition(
            config.SEQ_LEN,
            config.HIDDEN_DIM,
            period=20
        )

        # 3. 频谱注意力（创新）
        self.spectral_attn = SpectralAttention(config.HIDDEN_DIM, config.NUM_HEADS)

        # 4. 市场状态检测（创新）
        self.regime_detector = MarketRegimeDetector(
            config.HIDDEN_DIM,
            hidden_dim=64,
            n_regimes=3
        )

        # 5. 自适应加权（创新）
        self.adaptive_weighting = AdaptiveWeighting(config.HIDDEN_DIM, n_regimes=3)

        # 6. VCformer
        self.vcformer = VariableCentricTransformer(
            config.HIDDEN_DIM,
            config.NUM_HEADS,
            config.NUM_LAYERS,
            config.DROPOUT
        )

        # 7. TPA模块
        self.tpa = TPAModule(
            config.HIDDEN_DIM,
            config.CNN_FILTERS,
            config.CNN_CHANNELS
        )

        # 8. 多任务输出
        # 排序得分
        self.rank_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

        # 波动率预测
        self.vol_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1),
            nn.Softplus()  # 确保波动率为正
        )

        # 市场状态分类（辅助任务）
        self.regime_head = nn.Linear(config.HIDDEN_DIM, 3)

    def forward(self, x, return_attention=False):
        """
        x: (batch, seq_len, input_dim)
        Returns:
            rank_score: (batch, 1) 排序得分
            vol_pred: (batch, 1) 波动率预测
            regime_probs: (batch, 3) 市场状态概率
            aux: 辅助输出（用于解释）
        """
        # 1. 嵌入
        x_embedded = self.embed(x)  # (batch, seq_len, hidden)

        # 2. 时序分解
        decomposition_results = self.decomposition(x_embedded)
        x_decomp = decomposition_results['reconstructed']

        # 3. 频谱注意力
        x_spectral, spectral_weights = self.spectral_attn(x_decomp)

        # 4. 市场状态检测
        regime_results = self.regime_detector(x_spectral)

        # 5. 自适应加权
        x_weighted, attention_weights = self.adaptive_weighting(
            x_spectral,
            regime_results['regime_probs']
        )

        # 6. VCformer
        x_vc = self.vcformer(x_weighted)

        # 7. TPA
        x_tpa, tpa_weights = self.tpa(x_vc)  # (batch, hidden)

        # 8. 多任务输出
        rank_score = self.rank_head(x_tpa)
        vol_pred = self.vol_head(x_tpa)
        regime_logits = self.regime_head(x_tpa)
        regime_probs = F.softmax(regime_logits, dim=-1)

        # 9. 返回结果
        result = {
            'rank_score': rank_score,
            'vol_pred': vol_pred,
            'regime_probs': regime_probs,
            'regime_labels': regime_results['regime_labels'],
            'attention_weights': attention_weights,
            'tpa_weights': tpa_weights,
            'decomposition': decomposition_results,
            'spectral_weights': spectral_weights,
            'vc_attention': self.vcformer.get_attention_weights() if return_attention else None
        }

        return result

    def compute_loss(self, batch, config):
        """
        计算多任务损失

        Parameters:
        -----------
        batch: dict containing 'x', 'rank_target', 'vol_target', 'regime_target'
        config: 配置对象
        """
        x = batch['x']
        rank_target = batch.get('rank_target')
        vol_target = batch.get('vol_target')
        regime_target = batch.get('regime_target')

        # 前向传播
        outputs = self.forward(x)

        # 1. 排序损失
        loss_rank = 0
        if rank_target is not None:
            loss_rank = self._compute_ranking_loss(
                outputs['rank_score'].squeeze(-1),
                rank_target
            )

        # 2. 波动率损失
        loss_vol = 0
        if vol_target is not None:
            loss_vol = F.mse_loss(outputs['vol_pred'].squeeze(-1), vol_target)

        # 3. 市场状态损失
        loss_regime = 0
        if regime_target is not None:
            loss_regime = F.cross_entropy(
                outputs['regime_probs'],
                regime_target
            )

        # 4. 时序分解损失
        loss_decomp = outputs['decomposition']['decomposition_loss']

        # 5. 总损失
        total_loss = (loss_rank +
                      config.LAMBDA_VOL * loss_vol +
                      0.1 * loss_regime +
                      config.LAMBDA_DECOMP * loss_decomp)

        return {
            'total_loss': total_loss,
            'loss_rank': loss_rank,
            'loss_vol': loss_vol,
            'loss_regime': loss_regime,
            'loss_decomp': loss_decomp
        }

    def _compute_ranking_loss(self, scores, targets):
        """
        计算排序损失（Pairwise Hinge Loss）

        Parameters:
        -----------
        scores: (batch,) 预测得分
        targets: (batch,) 真实标签（超额收益）
        """
        # 对每个batch中的样本两两配对
        batch_size = scores.size(0)

        # 生成配对
        i_idx, j_idx = torch.triu_indices(batch_size, batch_size, offset=1)

        if i_idx.size(0) == 0:
            return torch.tensor(0.0, device=scores.device)

        # 获取配对的得分和标签
        scores_i = scores[i_idx]
        scores_j = scores[j_idx]
        targets_i = targets[i_idx]
        targets_j = targets[j_idx]

        # 判断排序是否正确
        correct_order = (targets_i > targets_j).float()

        # Hinge Loss
        margin = 0.1
        loss = torch.max(
            torch.zeros_like(scores_i),
            margin - (scores_i - scores_j) * (2 * correct_order - 1)
        )

        return loss.mean()