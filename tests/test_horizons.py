import numpy as np
import pandas as pd
import pytest

from data.labels import forward_labels
from research.analysis import analyze_experiment
from research.inference import feature_columns


@pytest.mark.parametrize('horizon', [1, 3, 5])
def test_forward_labels_match_future_closes_and_preserve_unknown_tail(horizon):
    dates = pd.bdate_range('2025-01-01', periods=12)
    index = pd.MultiIndex.from_product([dates, ['a', 'b']], names=['date', 'stock'])
    panel = pd.DataFrame({'close': np.repeat(100 * 1.01 ** np.arange(12), 2)}, index=index)
    labels = forward_labels(panel, horizon)
    assert labels.loc[(dates[0], 'a'), 'future_return'] == pytest.approx(1.01 ** horizon - 1)
    assert labels.loc[(dates[0], 'a'), 'future_risk'] == pytest.approx(.01)
    assert labels.loc[(dates[0], 'a'), 'target_excess'] == pytest.approx(0)
    assert labels.loc[dates[-horizon]:].isna().all().all()
    assert not feature_columns(labels)


def test_cost_diagnostic_uses_next_return_and_compounding(tmp_path):
    pd.DataFrame({'date': ['2025-01-02', '2025-01-03'], 'return': [-.01, .02]}).to_csv(tmp_path / 'backtest_returns.csv', index=False)
    pd.DataFrame({'date': ['2025-01-01'], 'cost': [.01], 'turnover': [1]}).to_csv(tmp_path / 'trades.csv', index=False)
    result = analyze_experiment(tmp_path)
    assert result['costs']['gross_return_same_holdings'] == pytest.approx(.02)
    assert result['costs']['terminal_drag_pp'] == pytest.approx(1.02)
    assert result['drawdown_peak'] == 'initial capital'
    assert result['drawdown_recovery'] == '2025-01-03'


def test_analysis_rejects_missing_reference_dates(tmp_path):
    pd.DataFrame({'date': ['2025-01-02', '2025-01-03'], 'return': [.01, .02]}).to_csv(tmp_path / 'backtest_returns.csv', index=False)
    pd.DataFrame({'date': ['2025-01-02'], 'return': [.01]}).to_csv(tmp_path / 'benchmark_returns.csv', index=False)
    with pytest.raises(ValueError, match='missing dates'):
        analyze_experiment(tmp_path)


def test_feature_scaling_never_changes_forward_labels():
    from data.features import FactorBuilder
    index = pd.MultiIndex.from_product([pd.bdate_range('2025-01-01', periods=3), ['a', 'b']], names=['date', 'stock'])
    frame = pd.DataFrame({'target_excess': np.arange(6) / 100,
                          'future_return': np.arange(6) / 50,
                          'future_risk': np.arange(6) / 30,
                          'feature': np.arange(6)}, index=index)
    transformed = FactorBuilder().process_factors(frame)
    pd.testing.assert_frame_equal(frame.iloc[:, :3], transformed.iloc[:, :3])
    assert feature_columns(transformed) == ['feature']


def test_two_versions_switch_in_both_languages():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
    app.radio[0].set_value('demo').run()
    app.selectbox(key='research_version').set_value('overnight').run()
    assert not app.exception
    assert '演示数据' in app.warning[0].value
    app.sidebar.selectbox[0].set_value('English').run()
    assert not app.exception
    app.selectbox(key='research_version').set_value('swing').run()
    app.selectbox(key='swing_horizon').set_value(3).run()
    assert not app.exception
    app.sidebar.radio[0].set_value('comparison').run()
    assert not app.exception
    assert 'real comparison' in app.title[0].value
