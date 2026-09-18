# portfolio/optimizer.py
import numpy as np
from scipy.optimize import minimize, Bounds
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, max_weight=0.1, min_weight=0.001):
        self.max_weight = max_weight
        self.min_weight = min_weight

    def optimize(self, scores, volatilities, cov_matrix=None):
        raise NotImplementedError

class RiskParityOptimizer(PortfolioOptimizer):
    def __init__(self, max_weight=0.1, min_weight=0.001, risk_aversion=0.5):
        super().__init__(max_weight, min_weight)
        self.risk_aversion = risk_aversion

    def optimize(self, scores, volatilities, cov_matrix=None):
        volatilities = np.asarray(volatilities).flatten()

        volatilities = np.maximum(volatilities, 1e-8)
        raw_weights = 1.0 / (volatilities ** self.risk_aversion)
        weights = self._apply_constraints(raw_weights)
        return weights

    def _apply_constraints(self, raw_weights):

        weights = raw_weights / (raw_weights.sum() + 1e-12)

        n = len(weights)
        if n == 0:
            return weights
        if n * self.max_weight <= 1.0:
            # Keep the unallocated fraction in cash when the cap is infeasible.
            return np.full(n, self.max_weight)

        while True:
            over_weighted = weights > self.max_weight
            if not over_weighted.any():
                break

            if over_weighted.all():
                weights = np.ones(n) / n
                break
            excess = (weights[over_weighted] - self.max_weight).sum()
            weights[over_weighted] = self.max_weight
            under_weighted = ~over_weighted
            sum_under = weights[under_weighted].sum()
            if sum_under > 0:
                weights[under_weighted] += excess * weights[under_weighted] / sum_under
            else:

                weights = np.ones(n) / n
                break

        weights = np.maximum(weights, self.min_weight)
        weights = weights / (weights.sum() + 1e-12)
        return weights

class MeanVarianceOptimizer(PortfolioOptimizer):
    def __init__(self, max_weight=0.1, min_weight=0.001, risk_aversion=0.5):
        super().__init__(max_weight, min_weight)
        self.risk_aversion = risk_aversion

    def optimize(self, scores, volatilities, cov_matrix=None):
        n = len(scores)
        scores = np.asarray(scores).flatten()
        volatilities = np.asarray(volatilities).flatten()
        volatilities = np.maximum(volatilities, 1e-8)

        if cov_matrix is None:
            cov_matrix = np.diag(volatilities ** 2)
        else:
            cov_matrix = np.asarray(cov_matrix)
            if cov_matrix.shape != (n, n):
                cov_matrix = np.diag(volatilities ** 2)

        def objective(w):
            w = np.asarray(w)
            port_return = w @ scores
            port_risk = w @ cov_matrix @ w
            utility = port_return - self.risk_aversion * 0.5 * port_risk
            return -utility

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = Bounds(self.min_weight, self.max_weight)
        w0 = np.ones(n) / n

        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6, 'maxiter': 1000}
        )

        if result.success:
            weights = result.x
        else:
            print("Mean-variance optimization failed, falling back to risk parity")
            weights = RiskParityOptimizer().optimize(scores, volatilities)
        return weights

class AdaptiveWeightingOptimizer(PortfolioOptimizer):
    def __init__(self, max_weight=0.1, min_weight=0.001,
                 bull_risk_aversion=0.3, bear_risk_aversion=1.0):
        super().__init__(max_weight, min_weight)
        self.bull_risk_aversion = bull_risk_aversion
        self.bear_risk_aversion = bear_risk_aversion

    def optimize(self, scores, volatilities, cov_matrix=None, regime_probs=None):
        if regime_probs is None:
            regime_probs = np.array([0.3, 0.3, 0.4])
        bull_prob, bear_prob, neutral_prob = regime_probs
        risk_aversion = (bull_prob * self.bull_risk_aversion +
                         bear_prob * self.bear_risk_aversion +
                         neutral_prob * 0.5)
        optimizer = MeanVarianceOptimizer(
            max_weight=self.max_weight,
            min_weight=self.min_weight,
            risk_aversion=risk_aversion
        )
        weights = optimizer.optimize(scores, volatilities, cov_matrix)
        return weights
