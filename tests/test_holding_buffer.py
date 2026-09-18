from types import SimpleNamespace
import numpy as np
import pandas as pd

from portfolio.backtest import Backtester


def test_rank_buffer_retains_incumbents_and_reduces_recorded_costs():
    dates = pd.bdate_range('2025-01-01', periods=3)
    index = pd.MultiIndex.from_product([dates, list('abcd')], names=['date', 'stock'])
    prices = pd.DataFrame({'close': 100.}, index=index)
    signals = {dates[0]: {s: {'score': score, 'vol': .02} for s, score in zip('abcd', [4, 3, 2, 1])},
               dates[1]: {s: {'score': score, 'vol': .02} for s, score in zip('abcd', [3, 2, 4, 1])},
               dates[2]: {s: {'score': score, 'vol': .02} for s, score in zip('abcd', [3, 2, 4, 1])}}
    config = dict(TOP_K=2, REBALANCE_FREQ=1, MAX_WEIGHT=.5, TRANSACTION_COST=.001, SLIPPAGE=.0005)
    plain = Backtester(SimpleNamespace(**config, HOLD_BUFFER=0)).run(signals, prices)
    buffered = Backtester(SimpleNamespace(**config, HOLD_BUFFER=1)).run(signals, prices)
    assert set(buffered.positions[1]) == {'date', 'a', 'b'}
    assert set(plain.positions[1]) == {'date', 'a', 'c'}
    assert buffered.trades[1]['turnover'] == 0
    assert plain.trades[1]['turnover'] == 1
    assert np.prod(1 + buffered.returns) > np.prod(1 + plain.returns)
    # Legacy configurations without the new field retain their exact behavior.
    legacy = Backtester(SimpleNamespace(**config)).run(signals, prices)
    pd.testing.assert_series_equal(legacy.returns, plain.returns)
