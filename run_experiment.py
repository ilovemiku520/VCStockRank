"""Run the isolated dashboard pipeline from the command line."""
import argparse
from research.experiments import create_experiment
from research.runner import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--stocks', type=int, default=None, help='Capacity budget; defaults to a conservative available-memory estimate')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, choices=[1, 3, 5], default=5)
    parser.add_argument('--sampling', choices=['random', 'legacy'], default='random')
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--margin', type=float, default=.05)
    args = parser.parse_args()
    from research.sampling import capacity_estimate
    capacity = args.stocks if args.stocks is not None else capacity_estimate()['estimated_capacity']
    directory = create_experiment({'DATA_START': args.start, 'DATA_END': args.end,
                                   'MAX_STOCKS': capacity, 'EPOCHS': args.epochs, 'SEED': args.seed, 'LABEL_HORIZON': args.horizon,
                                   'SAMPLING_METHOD': args.sampling, 'CONFIDENCE': args.confidence, 'MARGIN': args.margin})
    print(directory)
    return run(directory)

if __name__ == '__main__':
    from research.console import configure_console
    configure_console()
    raise SystemExit(main())
