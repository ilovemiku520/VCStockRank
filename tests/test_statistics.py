import numpy as np
import pandas as pd
import pytest

from research.sampling import sample_plan, draw_pool
from research.statistical import PCARidge, historical_design, block_bootstrap_mean


def test_finite_population_sampling_target_and_shortfall():
    plan = sample_plan(4992, 227)
    assert plan['required_stocks'] == 357
    assert plan['selected_stocks'] == 227
    assert plan['planned_margin'] == pytest.approx(.063554, abs=1e-6)
    assert not plan['target_met']
    assert sample_plan(4992, 500)['target_met']
    assert sample_plan(1, 1)['planned_margin'] == 0
    pool = pd.DataFrame({'code': [str(i) for i in range(100)]})
    pd.testing.assert_frame_equal(draw_pool(pool, 30), draw_pool(pool.iloc[::-1], 30))
    assert draw_pool(pool, 30)['code'].nunique() == 30


def test_historical_design_is_invariant_to_signal_date_and_future_changes():
    dates = pd.bdate_range('2025-01-01', periods=60)
    index = pd.MultiIndex.from_product([dates, ['a']], names=['date', 'stock'])
    frame = pd.DataFrame({'feature': np.arange(60.), 'target_excess': .01}, index=index)
    prices = pd.DataFrame({'close': 100 * 1.01 ** np.arange(60)}, index=index)
    before, columns = historical_design(frame, ['feature'], prices)
    frame.loc[dates[40]:, 'feature'] = 9999
    prices.loc[dates[40]:, 'close'] *= 9
    after, _ = historical_design(frame, ['feature'], prices)
    pd.testing.assert_frame_equal(before.loc[:dates[40], columns + ['risk']], after.loc[:dates[40], columns + ['risk']])


def test_svd_ridge_handles_collinearity_and_roundtrips_without_pickle(tmp_path):
    rng = np.random.default_rng(42)
    index = pd.MultiIndex.from_product([pd.bdate_range('2025-01-01', periods=30), list('abcd')], names=['date', 'stock'])
    x = rng.normal(size=len(index))
    frame = pd.DataFrame({'x': x, 'duplicate': x, 'constant': 1., 'target': 2 * x + rng.normal(0, .01, len(x))}, index=index)
    columns = ['x', 'duplicate', 'constant']
    model = PCARidge().fit(frame.iloc[:80], frame.iloc[80:], columns)
    assert np.isfinite(model.predict(frame)).all()
    assert model.diagnostics['components'] == 1
    np.testing.assert_allclose(model.mean, frame.iloc[:80][columns].mean())
    model.save(tmp_path / 'model.npz')
    np.testing.assert_allclose(PCARidge.load(tmp_path / 'model.npz').predict(frame), model.predict(frame))
    with pytest.raises(ValueError, match='no usable variation'):
        PCARidge().fit(frame.assign(x=1, duplicate=1), frame, columns)


def test_block_interval_preserves_constants_and_is_reproducible():
    interval = block_bootstrap_mean(np.full(50, .01))
    assert interval['lower'] == pytest.approx(.01)
    assert interval['upper'] == pytest.approx(.01)
    assert interval == block_bootstrap_mean(np.full(50, .01))
    assert block_bootstrap_mean([.01, .02]) is None
