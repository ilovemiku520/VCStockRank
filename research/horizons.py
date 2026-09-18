"""Shared experiment horizons and portfolio settings."""
HORIZONS = (1, 3, 5)


def validate_horizon(value):
    if isinstance(value, bool) or value not in HORIZONS:
        raise ValueError('Prediction horizon must be 1, 3 or 5 trading days.')
    return int(value)


def portfolio_for(config):
    from config import PortfolioConfig
    portfolio = PortfolioConfig()
    portfolio.REBALANCE_FREQ = validate_horizon(getattr(config, 'LABEL_HORIZON', 5))
    return portfolio
