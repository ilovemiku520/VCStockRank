"""Compatibility CLI; orchestration lives in research.pipeline."""
from research.pipeline import MultiModalStrategy, main

if __name__ == '__main__':
    from research.console import configure_console
    configure_console()
    main()
