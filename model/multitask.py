import torch
import torch.nn as nn
import torch.nn.functional as F
from .vcformer import VariableCentricTransformer
from .tpa_module import TPAModule
from .decomposition import TimeSeriesDecomposition, SpectralAttention
from .regime_detection import MarketRegimeDetector, AdaptiveWeighting

class MultiTaskVCformerTPA(nn.Module):
    """Fuse sequence decomposition, attention and temporal convolutions for two prediction heads."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT)
        )

        self.decomposition = TimeSeriesDecomposition(
            config.SEQ_LEN,
            config.HIDDEN_DIM,
            period=20
        )

        self.spectral_attn = SpectralAttention(config.HIDDEN_DIM, config.NUM_HEADS)

        self.regime_detector = MarketRegimeDetector(
            config.HIDDEN_DIM,
            hidden_dim=64,
            n_regimes=3
        )

        self.adaptive_weighting = AdaptiveWeighting(config.HIDDEN_DIM, n_regimes=3)

        # 6. VCformer
        self.vcformer = VariableCentricTransformer(
            config.HIDDEN_DIM,
            config.NUM_HEADS,
            config.NUM_LAYERS,
            config.DROPOUT
        )

        self.tpa = TPAModule(
            config.HIDDEN_DIM,
            config.CNN_FILTERS,
            config.CNN_CHANNELS
        )

        self.rank_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

        self.vol_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1),
            nn.Softplus()
        )

        self.regime_head = nn.Linear(config.HIDDEN_DIM, 3)

    def forward(self, x, return_attention=False):
        """Transform input tensors and return the module outputs."""

        x_embedded = self.embed(x)  # (batch, seq_len, hidden)

        decomposition_results = self.decomposition(x_embedded)
        x_decomp = decomposition_results['reconstructed']

        x_spectral, spectral_weights = self.spectral_attn(x_decomp)

        regime_results = self.regime_detector(x_spectral)

        x_weighted, attention_weights = self.adaptive_weighting(
            x_spectral,
            regime_results['regime_probs']
        )

        # 6. VCformer
        x_vc = self.vcformer(x_weighted)

        # 7. TPA
        x_tpa, tpa_weights = self.tpa(x_vc)  # (batch, hidden)

        rank_score = self.rank_head(x_tpa)
        vol_pred = self.vol_head(x_tpa)
        regime_logits = self.regime_head(x_tpa)
        regime_probs = F.softmax(regime_logits, dim=-1)

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
        """Combine the model prediction losses and decomposition penalty."""
        x = batch['x']
        rank_target = batch.get('rank_target')
        vol_target = batch.get('vol_target')
        regime_target = batch.get('regime_target')

        outputs = self.forward(x)

        loss_rank = 0
        if rank_target is not None:
            loss_rank = self._compute_ranking_loss(
                outputs['rank_score'].squeeze(-1),
                rank_target
            )

        loss_vol = 0
        if vol_target is not None:
            loss_vol = F.mse_loss(outputs['vol_pred'].squeeze(-1), vol_target)

        loss_regime = 0
        if regime_target is not None:
            loss_regime = F.cross_entropy(
                outputs['regime_probs'],
                regime_target
            )

        loss_decomp = outputs['decomposition']['decomposition_loss']

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
        """Compute pairwise hinge loss for a single cross-section."""

        batch_size = scores.size(0)

        i_idx, j_idx = torch.triu_indices(batch_size, batch_size, offset=1)

        if i_idx.size(0) == 0:
            return torch.tensor(0.0, device=scores.device)

        scores_i = scores[i_idx]
        scores_j = scores[j_idx]
        targets_i = targets[i_idx]
        targets_j = targets[j_idx]

        correct_order = (targets_i > targets_j).float()

        # Hinge Loss
        margin = 0.1
        loss = torch.max(
            torch.zeros_like(scores_i),
            margin - (scores_i - scores_j) * (2 * correct_order - 1)
        )

        return loss.mean()
