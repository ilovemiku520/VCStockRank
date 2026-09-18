"""Explicit runtime setup: importing configuration has no process side effects."""
import os
import random
import numpy as np
import torch

def configure_runtime(config):
    threads = max(1, min(config.CPU_THREADS, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f'Runtime: device={config.DEVICE}, threads={threads}, seed={config.SEED}')
