"""Persist auditable results and provenance after a completed experiment."""
import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
import subprocess
import numpy as np
import pandas as pd
import torch
from research.experiments import read_json, write_json, ROOT

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, (pd.Timestamp, torch.device, Path)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def export_results(strategy, directory, elapsed):
    directory = Path(directory)
    result = strategy.backtest_results
    result.returns.rename_axis('date').rename('return').to_csv(directory / 'backtest_returns.csv')
    pd.DataFrame(result.positions).to_csv(directory / 'positions.csv', index=False)
    pd.DataFrame(getattr(result, 'trades', [])).to_csv(directory / 'trades.csv', index=False)
    predictions = [{'date': date, 'stock': stock, 'score': score, 'vol': vol}
                   for date, pred in strategy.predictions.items()
                   for stock, score, vol in zip(pred['stocks'], pred['scores'], pred['vols'])]
    pd.DataFrame(predictions).to_csv(directory / 'predictions.csv', index=False)
    prices = strategy.data['price_raw']['close'].unstack('stock').sort_index()
    benchmark = prices.pct_change(fill_method=None).mean(axis=1).reindex(result.returns.index)
    benchmark.rename_axis('date').rename('return').to_csv(directory / 'benchmark_returns.csv')
    evaluation = {key: value for key, value in strategy.evaluation_results.items()
                  if not isinstance(value, (pd.Series, pd.DataFrame))}
    config = {key: getattr(strategy.config, key) for key in dir(strategy.config) if key.isupper()}
    from research.provenance import capture_source
    provenance = read_json(directory / 'source_snapshot.json')
    if provenance is None:
        provenance = {**capture_source(ROOT), 'timing': 'report time; legacy run without a start snapshot'}
    history = pd.read_csv(directory / 'logs/training_history.csv')
    summary = {
        'protocol_version': strategy.config.PROTOCOL_VERSION,
        'sampling_plan': read_json(directory / 'sampling_plan.json', None),
        'data_source': read_json(directory / 'data/source.json', {}),
        'data_rows': len(strategy.data['daily']), 'stock_count': prices.shape[1],
        'feature_count': len(strategy.feature_cols), 'feature_columns': strategy.feature_cols,
        'sample_counts': strategy.sample_counts, 'split': strategy.split_metadata, 'settings': config,
        'device': str(strategy.config.DEVICE),
        'gpu': torch.cuda.get_device_name(0) if strategy.config.DEVICE.type == 'cuda' else None,
        'peak_gpu_memory_mb': torch.cuda.max_memory_allocated() / 1024 ** 2 if strategy.config.DEVICE.type == 'cuda' else None,
        'epochs_completed': len(history), 'best_epoch': int(history['val_loss'].idxmin()) + 1,
        'best_validation_loss': history['val_loss'].min(), 'elapsed_seconds': elapsed,
        'test_first_return': result.returns.index.min(), 'test_last_return': result.returns.index.max(),
        'metrics': result.metrics, 'evaluation': evaluation,
        'benchmark': 'Daily equal-weight of the same stock pool, before transaction costs; not CSI 300.',
        'base_git_revision': provenance['git_revision'],
        'source_snapshot_timing': provenance['timing'],
        'source_working_tree_dirty': provenance.get('working_tree_dirty'),
        'environment': {'python': platform.python_version(), **{name: importlib.metadata.version(name)
                         for name in ['torch', 'numpy', 'pandas', 'scipy', 'baostock']}},
        'artifact_hashes': {name: file_hash(directory / name) for name in
                           ['stock_pool.csv', 'data/market.parquet', 'logs/best_model.pt', 'backtest_returns.csv']},
        'source_hashes': provenance['source_hashes'],
    }
    write_json(directory / 'summary.json', serializable(summary))
    from research.analysis import analyze_experiment
    write_json(directory / 'analysis.json', serializable(analyze_experiment(directory)))
    return summary
