"""Execute a complete experiment with durable stage updates."""
import os
from pathlib import Path
import time
import traceback
from research.experiments import ROOT, read_json, update_status

def run(directory):
    directory = Path(directory).resolve()
    previous = Path.cwd()
    started = time.monotonic()
    stage = 'initializing'
    try:
        os.chdir(directory)
        from config import ModelConfig
        from research.pipeline import MultiModalStrategy
        from research.reporting import export_results
        config = ModelConfig()
        settings = read_json(directory / 'settings.json', {})
        allowed = {'DATA_START', 'DATA_END', 'MAX_STOCKS', 'EPOCHS', 'PATIENCE', 'SEED', 'CPU_THREADS', 'BATCH_SIZE'}
        unknown = settings.keys() - allowed
        if unknown:
            raise ValueError(f'Unknown settings: {sorted(unknown)}')
        for key, value in settings.items():
            setattr(config, key, value)
        config.CACHE_DIR = str(ROOT / 'cache/market')
        strategy = MultiModalStrategy(config)
        steps = [('download', strategy.load_data), ('training', strategy.train_model),
                 ('prediction', strategy.generate_predictions), ('backtest', strategy.run_backtest),
                 ('evaluation', strategy.evaluate_predictions)]
        for stage, action in steps:
            update_status(directory, 'running', stage)
            outcome = action()
            if stage in {'download', 'training', 'backtest'} and outcome is None:
                raise RuntimeError(f'{stage} returned no result; inspect run.log.')
            if stage == 'prediction' and not outcome:
                raise RuntimeError('No test predictions were generated.')
        if strategy.backtest_results.returns.empty:
            raise RuntimeError('Backtest produced no daily returns.')
        stage = 'reporting'
        update_status(directory, 'running', stage)
        export_results(strategy, directory, time.monotonic() - started)
        update_status(directory, 'complete', 'complete', elapsed_seconds=time.monotonic() - started)
        return 0
    except Exception as error:
        traceback.print_exc()
        update_status(directory, 'failed', stage, str(error))
        return 1
    finally:
        os.chdir(previous)
