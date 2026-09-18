"""Freeze class defaults so saved models do not depend on future source defaults."""
from copy import deepcopy


def snapshot_config(config):
    return {key: deepcopy(getattr(config, key)) for key in dir(config) if key.isupper()}


def restore_config(config, checkpoint, legacy_settings=None):
    saved = checkpoint['config']
    if isinstance(saved, dict):
        values = saved
    else:
        # Pickle stores instance attributes, not the class defaults used at fitting.
        if not legacy_settings:
            raise ValueError('Legacy checkpoints require the original summary.json settings for reliable replay.')
        values = {**snapshot_config(saved), **legacy_settings}
    for key, value in values.items():
        if key.isupper() and key != 'DEVICE':
            setattr(config, key, deepcopy(value))
    config.PROTOCOL_VERSION = checkpoint['protocol_version']
    config.FEATURE_COLS = checkpoint.get('feature_cols')
    if not config.FEATURE_COLS:
        raise ValueError('Checkpoint is missing its feature schema; retrain the model.')
    return config
