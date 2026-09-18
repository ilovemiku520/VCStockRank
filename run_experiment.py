"""Run the isolated dashboard pipeline from the command line."""
import argparse
from research.experiments import create_experiment
from research.runner import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--stocks', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    directory = create_experiment({'DATA_START': args.start, 'DATA_END': args.end,
                                   'MAX_STOCKS': args.stocks, 'EPOCHS': args.epochs, 'SEED': args.seed})
    print(directory)
    return run(directory)

if __name__ == '__main__':
    from research.console import configure_console
    configure_console()
    raise SystemExit(main())
