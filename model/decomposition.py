# model/decomposition.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeSeriesDecomposition(nn.Module):
    """Learn trend, seasonal and residual components with autocorrelation regularization."""

    def __init__(self, seq_len, feature_dim, period=20):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.period = period

        self.trend_net = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2),
        )

        freq_len = seq_len // 2 + 1
        self.fft_weights = nn.Parameter(torch.randn(feature_dim, freq_len))

        self.change_detector = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, seq_len),
            nn.Sigmoid()
        )

        self.reconstruction = nn.Linear(feature_dim * 3, feature_dim)

    def forward(self, x):
        """Transform input tensors and return the module outputs."""

        x_perm = x.permute(0, 2, 1)

        trend = self.trend_net(x_perm)

        x_fft = torch.fft.rfft(x_perm, dim=-1)  # (batch, feature, freq_len)
        season = torch.fft.irfft(x_fft * self.fft_weights.unsqueeze(0), dim=-1)  # (batch, feature, seq_len)

        residual = x_perm - trend - season

        change_weights = self.change_detector(x_perm.mean(dim=1))
        change_points = change_weights.unsqueeze(1) * x_perm

        combined = torch.cat([trend, season, residual], dim=1)
        combined = combined.permute(0, 2, 1)  # (batch, seq, feature*3)
        reconstructed = self.reconstruction(combined)

        return {
            'trend': trend.permute(0, 2, 1),
            'seasonal': season.permute(0, 2, 1),
            'residual': residual.permute(0, 2, 1),
            'change_points': change_points.permute(0, 2, 1),
            'reconstructed': reconstructed,
            'decomposition_loss': self._compute_decomposition_loss(x, trend, season, residual)
        }

    def _compute_decomposition_loss(self, x, trend, seasonal, residual):
        """Penalize trend roughness, seasonal mismatch and residual autocorrelation."""

        trend_smooth = F.mse_loss(
            trend[:, :, 1:] - trend[:, :, :-1],
            torch.zeros_like(trend[:, :, 1:])
        )

        seasonal_auto = F.mse_loss(
            seasonal[:, :, self.period:],
            seasonal[:, :, :-self.period]
        )

        residual_flat = residual.reshape(-1, residual.shape[-1])  # (batch*feature, seq_len)
        mean = residual_flat.mean(dim=-1, keepdim=True)
        resid_centered = residual_flat - mean

        auto_corr_loss = 0.0
        n_lags = min(5, residual.shape[-1] - 1)
        for lag in range(1, n_lags + 1):
            cov_lag0 = torch.mean(resid_centered ** 2, dim=-1)  # (N,)
            cov_lagk = torch.mean(resid_centered[:, lag:] * resid_centered[:, :-lag], dim=-1)
            corr_lagk = cov_lagk / (cov_lag0 + 1e-8)
            auto_corr_loss += torch.mean(corr_lagk ** 2)

        return trend_smooth + seasonal_auto + 0.1 * auto_corr_loss

class WaveletDecomposition(nn.Module):
    """Approximate multiscale decomposition using strided convolutions."""

    def __init__(self, seq_len, feature_dim, wavelet_level=3):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.wavelet_level = wavelet_level

        self.decomp_filters = nn.ModuleList([
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, stride=2)
            for _ in range(wavelet_level)
        ])

        self.recon_filters = nn.ModuleList([
            nn.ConvTranspose1d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
            for _ in range(wavelet_level)
        ])

    def forward(self, x):
        """Transform input tensors and return the module outputs."""
        x_perm = x.permute(0, 2, 1)  # (batch, feature, seq)

        multi_scale = []
        current = x_perm

        for i, filter_layer in enumerate(self.decomp_filters):

            coeffs = filter_layer(current)
            multi_scale.append(coeffs)

            if i < len(self.decomp_filters) - 1:
                current = self.recon_filters[i](coeffs)
                current = F.leaky_relu(current)

        scaled_features = []
        for i, coeff in enumerate(multi_scale):
            if coeff.shape[-1] < self.seq_len:
                coeff = F.interpolate(coeff, size=self.seq_len, mode='linear', align_corners=False)
            scaled_features.append(coeff.permute(0, 2, 1))

        return scaled_features

class SpectralAttention(nn.Module):
    """Apply multihead attention to frequency-domain features."""

    def __init__(self, feature_dim, n_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads

        self.fft_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1)

        self.attention = nn.MultiheadAttention(feature_dim, n_heads, batch_first=True)

        self.output = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        """

        x_fft = torch.fft.rfft(x, dim=1).abs()
        x_fft = self.fft_conv(x_fft.permute(0, 2, 1)).permute(0, 2, 1)

        attn_out, attn_weights = self.attention(x_fft, x_fft, x_fft)

        output = self.output(attn_out)

        return output, attn_weights
