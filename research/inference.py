"""Shared batched inference with the same history window as the dataset."""
import numpy as np
import pandas as pd
import torch

EXCLUDED = {'close', 'open', 'high', 'low', 'volume', 'amount', 'turnover',
            'stock', 'future_ret_5d', 'future_vol_5d', 'market_ret', 'excess_ret_5d', 'ret_1d', 'target_excess'}

def feature_columns(factors):
    return [c for c in factors.columns if c not in EXCLUDED and not c.startswith('future_')]

def predict_panel(model, config, factors, columns, test_start):
    if test_start is None:
        raise ValueError('缺少测试集起始日期，禁止全样本预测。')
    model.eval()
    records = {}
    windows, keys = [], []

    def flush():
        if not windows:
            return
        x = torch.as_tensor(np.stack(windows), dtype=torch.float32, device=config.DEVICE)
        with torch.inference_mode():
            outputs = model(x, return_attention=False)
        scores = outputs['rank_score'].cpu().numpy().ravel()
        vols = outputs['vol_pred'].cpu().numpy().ravel()
        if not np.isfinite(scores).all() or not np.isfinite(vols).all():
            raise ValueError('模型输出包含非有限值。')
        for (date, stock), score, vol in zip(keys, scores, vols):
            entry = records.setdefault(date, {'stocks': [], 'scores': [], 'vols': []})
            entry['stocks'].append(stock)
            entry['scores'].append(float(score))
            entry['vols'].append(float(vol))
        windows.clear()
        keys.clear()

    for stock, group in factors.groupby(level='stock', sort=True):
        frame = group.droplevel('stock').sort_index()
        values = frame[columns].to_numpy(dtype=np.float32)
        for index in range(config.SEQ_LEN, len(frame)):
            date = pd.Timestamp(frame.index[index])
            if date < pd.Timestamp(test_start):
                continue
            window = values[index - config.SEQ_LEN:index]
            if not np.isfinite(window).all():
                continue
            windows.append(window)
            keys.append((date, stock))
            if len(windows) >= config.BATCH_SIZE:
                flush()
    flush()
    return {date: {key: np.array(value) for key, value in entry.items()}
            for date, entry in sorted(records.items())}
