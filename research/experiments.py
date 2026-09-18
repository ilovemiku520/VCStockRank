"""Experiment persistence and process management without UI or ML imports."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'

def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Windows antivirus and file watchers can briefly hold the destination open.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.1 * (attempt + 1))

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default

def update_status(directory, state, stage, message='', **extra):
    write_json(Path(directory) / 'status.json', {
        'state': state, 'stage': stage, 'message': message,
        'updated_at': datetime.now(timezone.utc).isoformat(), **extra,
    })

def create_experiment(settings, root=ROOT, name=None):
    run_id = name or datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    directory = Path(root) / 'runs' / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'data').mkdir()
    if settings.get('SAMPLING_METHOD') == 'random':
        from research.sampling import eligible_pool, sample_plan, draw_pool
        pool = eligible_pool(Path(root) / 'stock_pool.csv')
        plan = sample_plan(len(pool), settings['MAX_STOCKS'], settings.get('CONFIDENCE', .95), settings.get('MARGIN', .05))
        selected = draw_pool(pool, plan['selected_stocks'], settings.get('SEED', 42))
        selected.to_csv(directory / 'stock_pool.csv', index=False)
        write_json(directory / 'sampling_plan.json', {**plan, 'seed': settings.get('SEED', 42),
                    'frame': 'Deduplicated Shanghai/Shenzhen A shares, excluding ST and delisting names.'})
        settings = {**settings, 'MAX_STOCKS': len(selected)}
    else:
        shutil.copy2(Path(root) / 'stock_pool.csv', directory / 'stock_pool.csv')
    write_json(directory / 'settings.json', settings)
    update_status(directory, 'created', 'pending')
    return directory

def launch_experiment(directory):
    directory = Path(directory).resolve()
    with (directory / 'run.log').open('ab') as log:
        process = subprocess.Popen(
            [sys.executable, '-u', '-X', 'utf8', str(ROOT / 'dashboard_worker.py'), str(directory)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    return {'directory': str(directory), 'process': process}

def list_experiments(root=ROOT):
    result = []
    for directory in sorted((Path(root) / 'runs').glob('*'), reverse=True):
        if directory.is_dir():
            result.append({'id': directory.name, 'directory': directory,
                           **read_json(directory / 'status.json', {})})
    return result
