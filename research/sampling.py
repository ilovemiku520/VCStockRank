"""Finite-population planning and reproducible simple random stock sampling."""
import math
from statistics import NormalDist
import subprocess

import pandas as pd


def eligible_pool(path):
    """Define the supported Shanghai/Shenzhen A-share sampling frame before drawing."""
    pool = pd.read_csv(path, dtype=str)
    if 'code' not in pool:
        raise ValueError('Stock pool requires a code column.')
    pool['code'] = pool['code'].str.strip().str.zfill(6)
    pool = pool[pool['code'].str.fullmatch(r'[036]\d{5}', na=False)]
    if 'name' in pool:
        pool = pool[~pool['name'].str.contains('ST|退', case=False, na=False)]
    return pool.drop_duplicates('code').sort_values('code').reset_index(drop=True)


def sample_plan(population, capacity, confidence=.95, margin=.05):
    """Plan a proportion estimate under SRS without replacement, using worst-case p=.5."""
    if population < 1 or capacity < 1 or int(population) != population or int(capacity) != capacity:
        raise ValueError('Population and capacity must be positive integers.')
    if not 0 < confidence < 1 or not 0 < margin < 1:
        raise ValueError('Confidence and margin must lie strictly between zero and one.')
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    a = z * z * .25
    required = min(population, math.ceil(population * a / (margin * margin * (population - 1) + a)))
    selected = min(required, capacity, population)
    achieved = z * math.sqrt(.25 / selected * (population - selected) / (population - 1)) if population > 1 else 0.0
    return {'population': population, 'confidence': confidence, 'target_margin': margin,
            'planning_proportion': .5, 'z_statistic': z, 'required_stocks': required,
            'capacity_stocks': capacity, 'selected_stocks': selected,
            'planned_margin': achieved, 'target_met': selected >= required,
            'method': 'simple random sampling without replacement; normal-approximation planning',
            'scope': 'Finite stock-pool binary proportion, not model-return confidence or power.'}


def capacity_estimate(trading_days=750, features=51, sequence=30):
    """Conservative memory estimate; not a measured maximum or a runtime guarantee."""
    try:
        import psutil
        ram_mb = psutil.virtual_memory().available / 1024 ** 2
    except ImportError:
        return {'estimated_capacity': 25, 'basis': 'fallback; available RAM unavailable', 'measured': False}
    per_stock_mb = trading_days * sequence * features * 4 * 4 / 1024 ** 2
    ram_capacity = max(1, int(ram_mb * .30 / per_stock_mb))
    gpu_mb = None
    try:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                      text=True, timeout=3, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        gpu_mb = float(raw.splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    # A conservative allowance relative to the observed 25-stock GPU run.
    gpu_capacity = max(1, int(max(0, gpu_mb - 1024) / 20)) if gpu_mb is not None else ram_capacity
    return {'estimated_capacity': min(ram_capacity, gpu_capacity), 'available_ram_mb': round(ram_mb),
            'free_gpu_mb': gpu_mb, 'estimated_ram_mb_per_stock': round(per_stock_mb, 2),
            'basis': '30% of available RAM; GPU reserves 1 GiB and budgets 20 MiB per stock',
            'measured': False, 'note': 'Memory planning only; runtime, network and allocation peaks remain uncertain.'}


def draw_pool(pool, count, seed=42):
    if not 1 <= count <= len(pool):
        raise ValueError('Sample count must lie within the sampling frame.')
    # Sort first so a row reorder in the source cannot change the seeded sample.
    return pool.sort_values('code').sample(n=count, random_state=seed, replace=False).sort_values('code').reset_index(drop=True)
