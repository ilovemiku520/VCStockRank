"""Shared experiment horizons and portfolio settings."""
HORIZONS = (1, 3, 5)
LEGACY_PORTFOLIO = {'TOP_K': 10, 'HOLD_BUFFER': 0, 'MAX_WEIGHT': .1, 'RISK_AVERSION': .5,
                    'TRANSACTION_COST': .001, 'SLIPPAGE': .0005}


def validate_horizon(value):
    if isinstance(value, bool) or value not in HORIZONS:
        raise ValueError('Prediction horizon must be 1, 3 or 5 trading days.')
    return int(value)


def portfolio_for(config):
    from config import PortfolioConfig
    portfolio = PortfolioConfig()
    # Old v3/v4 reports used these fixed values. Do not inherit changed defaults.
    for key, value in getattr(config, 'PORTFOLIO_SETTINGS', LEGACY_PORTFOLIO).items():
        if key in LEGACY_PORTFOLIO:
            setattr(portfolio, key, value)
    portfolio.REBALANCE_FREQ = validate_horizon(getattr(config, 'LABEL_HORIZON', 5))
    return portfolio
