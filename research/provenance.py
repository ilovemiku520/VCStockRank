"""Capture source identity before imports and training can outlive workspace edits."""
import hashlib
from pathlib import Path
import subprocess


def capture_source(root):
    root = Path(root)
    paths = [root / 'config.py', root / 'backtest_main.py', root / 'compare_models.py']
    paths += [path for folder in ['research', 'data', 'training', 'model', 'portfolio', 'evaluation']
              for path in (root / folder).glob('*.py')]
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths if path.exists()}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = 'unknown', None
    return {'git_revision': revision, 'working_tree_dirty': dirty, 'source_hashes': hashes}
