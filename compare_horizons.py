"""Run matched 1/3/5-day experiments; results are exploratory, not model selection."""
import argparse
import subprocess
import sys
from pathlib import Path
from research.experiments import ROOT, create_experiment, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--stocks', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', default='horizon-comparison')
    args = parser.parse_args()
    results = []
    for horizon in (1, 3, 5):
        directory = create_experiment({'DATA_START': args.start, 'DATA_END': args.end,
            'EPOCHS': args.epochs, 'MAX_STOCKS': args.stocks, 'SEED': args.seed,
            'LABEL_HORIZON': horizon}, name=f'{args.name}-{horizon}d')
        with (directory / 'run.log').open('wb') as log:
            subprocess.run([sys.executable, '-u', '-X', 'utf8', str(ROOT / 'dashboard_worker.py'), str(directory)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'verify_experiment.py'), str(directory)], cwd=ROOT, check=True)
        analysis = read_json(directory / 'analysis.json', {})
        summary = read_json(directory / 'summary.json', {})
        results.append({'id': directory.name, 'horizon': horizon, 'analysis': analysis,
                        'best_epoch': summary['best_epoch'], 'protocol': summary['protocol_version']})
        write_json(ROOT / 'runs' / f'{args.name}.json', {'comparisons': results,
            'note': 'Exploratory reuse of the held-out period. Confirm any selected version on new unseen dates.'})
        print(f'Completed {horizon}-day experiment: {directory}', flush=True)
    return 0


if __name__ == '__main__':
    from research.console import configure_console
    configure_console()
    raise SystemExit(main())
