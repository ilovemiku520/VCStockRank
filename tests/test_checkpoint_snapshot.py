from types import SimpleNamespace
import pytest

from research.checkpoints import snapshot_config, restore_config
from research.horizons import portfolio_for


def test_checkpoint_survives_changed_class_defaults_and_mutable_config():
    class Config:
        SEQ_LEN = 30
        FEATURE_COLS = ['original']
        DEVICE = 'cpu'
    original = Config()
    checkpoint = {'config': snapshot_config(original), 'protocol_version': 4, 'feature_cols': ['original']}
    Config.SEQ_LEN = 999
    Config.FEATURE_COLS.append('future')
    restored = restore_config(Config(), checkpoint)
    assert restored.SEQ_LEN == 30
    assert restored.FEATURE_COLS == ['original']
    assert checkpoint['config']['FEATURE_COLS'] == ['original']


def test_legacy_checkpoint_requires_and_restores_original_settings():
    old = SimpleNamespace(SEQ_LEN=999, DEVICE='old-device')
    checkpoint = {'config': old, 'protocol_version': 3, 'feature_cols': ['factor']}
    with pytest.raises(ValueError, match='original summary'):
        restore_config(SimpleNamespace(), checkpoint)
    current = SimpleNamespace(DEVICE='current-device')
    restore_config(current, checkpoint, {'SEQ_LEN': 30, 'DEVICE': 'recorded-device'})
    assert current.SEQ_LEN == 30
    assert current.DEVICE == 'current-device'
    assert current.PROTOCOL_VERSION == 3


def test_portfolio_replay_ignores_future_defaults(monkeypatch):
    from config import PortfolioConfig
    monkeypatch.setattr(PortfolioConfig, 'TRANSACTION_COST', .99)
    assert portfolio_for(SimpleNamespace(LABEL_HORIZON=1)).TRANSACTION_COST == .001
    frozen = SimpleNamespace(LABEL_HORIZON=3, PORTFOLIO_SETTINGS={'TRANSACTION_COST': .002})
    assert portfolio_for(frozen).TRANSACTION_COST == .002
    assert portfolio_for(frozen).REBALANCE_FREQ == 3
