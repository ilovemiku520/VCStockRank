"""Compatibility process entry point for a single experiment."""
import sys
from research.runner import run

if __name__ == '__main__':
    from research.console import configure_console
    configure_console()
    sys.exit(run(sys.argv[1]))
