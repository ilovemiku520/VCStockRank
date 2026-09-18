import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture

class MarketRegimeDetector(nn.Module):
    """Estimate latent market regimes from sequence summaries."""

    def __init__(self, feature_dim, hidden_dim=64, n_regimes=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes

        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes * n_regimes)
        )

        self.emission_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes)
        )

        self.classifier = nn.Linear(hidden_dim, n_regimes)

        self.regime_params = nn.ParameterDict({
            'means': nn.Parameter(torch.randn(n_regimes, feature_dim)),
            'vars': nn.Parameter(torch.ones(n_regimes, feature_dim))
        })

    def forward(self, x):
        """Transform input tensors and return the module outputs."""

        features = self.feature_extractor(x.mean(dim=1))  # (batch, hidden)

        regime_logits = self.classifier(features)
        regime_probs = F.softmax(regime_logits, dim=-1)

        transition_logits = self.transition_net(features)
        transition_matrix = F.softmax(
            transition_logits.view(-1, self.n_regimes, self.n_regimes),
            dim=-1
        )

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
        """Calculate the auxiliary regime classification objective."""

        weighted_returns = returns.unsqueeze(1) * regime_probs.unsqueeze(2)

        regime_means = weighted_returns.mean(dim=0, keepdim=True)
        regime_vars = weighted_returns.var(dim=0, keepdim=True)

        between_regime_var = regime_means.var(dim=1)

        within_regime_var = regime_vars.mean(dim=1)

        return -between_regime_var + within_regime_var

class AdaptiveWeighting(nn.Module):
    """Condition feature weights on estimated regime probabilities."""

    def __init__(self, feature_dim, n_regimes=3):
        super().__init__()
        self.n_regimes = n_regimes

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
        """Transform input tensors and return the module outputs."""
        batch_size = features.shape[0]
        seq_len = features.shape[1]
        feat_dim = features.shape[2]

        weights = torch.zeros(batch_size, seq_len, feat_dim).to(features.device)

        for i, weight_net in enumerate(self.regime_weights):

            w = weight_net(features)  # (batch, seq_len, 1)

            regime_weight = regime_probs[:, i:i + 1, None]  # (batch, 1, 1)
            weights += w * regime_weight

        weighted_features = features * weights

        attention_weights = weights.mean(dim=-1)  # (batch, seq_len)

        return weighted_features, attention_weights
