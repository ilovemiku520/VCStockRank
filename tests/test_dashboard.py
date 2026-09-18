from io import StringIO
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard_data import performance, read_returns


def test_returns_sorted_and_first_loss_is_drawdown():
    returns = read_returns(StringIO('date,return\n2025-01-03,0.1\n2025-01-02,-0.2\n'))
    metrics, _ = performance(returns)
    assert returns.index.is_monotonic_increasing
    assert metrics['最大回撤'] == pytest.approx(-.2)
    assert metrics['累计收益'] == pytest.approx(-.12)


@pytest.mark.parametrize('csv', [
    'date,value\n2025-01-01,1', 'date,return\n',
    'date,return\nbad,0.1', 'date,return\n2025-01-01,inf',
    'date,return\n2025-01-01,-2',
    'date,return\n2025-01-01,0.1\n2025-01-01,0.2',
])
def test_invalid_returns_are_rejected(csv):
    with pytest.raises(ValueError):
        read_returns(StringIO(csv))


def test_single_day_has_no_sharpe():
    returns = pd.Series([.01], index=pd.to_datetime(['2025-01-01']))
    assert performance(returns)[0]['夏普比率'] is None


def test_dashboard_navigation_and_stock_search():
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert '演示数据' in app.warning[0].value
    app.sidebar.radio[0].set_value('股票池').run()
    assert not app.exception
    app.text_input[0].set_value('000001').run()
    assert app.dataframe[0].value.iloc[0]['股票代码'] == '000001'
    app.text_input[0].set_value('不存在的股票').run()
    assert app.dataframe[0].value.empty
    app.sidebar.radio[0].set_value('开始研究').run()
    assert not app.exception
    assert app.date_input[0].label == '数据开始日期'


def test_upload_and_local_empty_states():
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
    app.radio[0].set_value('上传 CSV').run()
    assert not app.exception
    assert len(app.metric) == 0
    app.radio[0].set_value('本地实验').run()
    assert not app.exception
    assert len(app.info) > 0


@pytest.mark.parametrize('fail', [False, True])
def test_worker_records_failure_or_exports_result(tmp_path, monkeypatch, fail):
    from dashboard_worker import run

    class Strategy:
        def __init__(self, config):
            assert config.EPOCHS == 3

        def load_data(self):
            return None if fail else pd.DataFrame({'x': [1]})

        def train_model(self):
            return object()

        def generate_predictions(self):
            return {'predictions': True}

        def run_backtest(self):
            return SimpleNamespace(returns=pd.Series([.01, -.02], index=pd.bdate_range('2025-01-01', periods=2)))

        def evaluate_predictions(self):
            pass

    monkeypatch.setitem(sys.modules, 'config', SimpleNamespace(ModelConfig=SimpleNamespace))
    monkeypatch.setitem(sys.modules, 'main', SimpleNamespace(MultiModalStrategy=Strategy))
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'settings.json').write_text('{"EPOCHS": 3}', encoding='utf-8')
    assert run(tmp_path) == (1 if fail else 0)
    status = json.loads((tmp_path / 'status.json').read_text(encoding='utf-8'))
    assert status['state'] == ('failed' if fail else 'complete')
    if not fail:
        assert len(read_returns(tmp_path / 'backtest_returns.csv')) == 2
    else:
        assert not (tmp_path / 'backtest_returns.csv').exists()
