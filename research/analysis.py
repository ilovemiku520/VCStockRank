"""Descriptive diagnostics from saved artifacts, without training dependencies."""
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard_data import performance, read_returns


def analyze_experiment(directory):
    """Use the full saved interval; never silently fill missing reference dates."""
    directory = Path(directory)
    returns = read_returns(directory / 'backtest_returns.csv', language='en')
    metrics, curves = performance(returns)
    trough = curves['回撤'].idxmin()
    prior = curves.loc[:trough, '净值']
    peak = prior.idxmax() if prior.max() >= 1 else None
    peak_value = max(1.0, float(prior.max()))
    recovered = curves.loc[trough:, '净值']
    recovered = recovered[recovered >= peak_value]
    result = {
        'start': str(returns.index.min().date()), 'end': str(returns.index.max().date()),
        'days': len(returns), 'total_return': float(metrics['累计收益']),
        'max_drawdown': float(metrics['最大回撤']),
        'drawdown_peak': str(peak.date()) if peak is not None else 'initial capital',
        'drawdown_trough': str(trough.date()),
        'drawdown_recovery': str(recovered.index[0].date()) if len(recovered) else None,
        'positive_day_ratio': float((returns > 0).mean()),
        'average_up_day': float(returns[returns > 0].mean()) if (returns > 0).any() else None,
        'average_down_day': float(returns[returns < 0].mean()) if (returns < 0).any() else None,
    }
    monthly = pd.DataFrame({'strategy': (1 + returns).resample('ME').prod() - 1})
    reference_path = directory / 'benchmark_returns.csv'
    if reference_path.exists():
        reference = read_returns(reference_path, language='en').reindex(returns.index)
        if reference.isna().any():
            raise ValueError('Reference is missing dates from the strategy interval.')
        reference_total = float((1 + reference).prod() - 1)
        result['reference_return'] = reference_total
        result['return_difference_pp'] = (result['total_return'] - reference_total) * 100
        result['relative_wealth_return'] = ((1 + result['total_return']) / (1 + reference_total) - 1
                                           if reference_total > -1 else None)
        monthly['reference'] = (1 + reference).resample('ME').prod() - 1
        monthly['difference_pp'] = (monthly['strategy'] - monthly['reference']) * 100
    result['monthly'] = [dict(month=str(date.to_period('M')), **row.to_dict()) for date, row in monthly.iterrows()]
    trade_path = directory / 'trades.csv'
    if trade_path.exists():
        trades = pd.read_csv(trade_path, parse_dates=['date'])
        # v3 debits each close-time rebalance at the next saved return date.
        cost_factors = pd.Series(1.0, index=returns.index)
        for trade in trades.itertuples():
            cost = float(trade.cost)
            if not np.isfinite(cost) or not 0 <= cost < 1:
                raise ValueError('Invalid transaction cost in trade ledger.')
            next_day = returns.index.searchsorted(trade.date, side='right')
            if next_day >= len(returns):
                raise ValueError('Trade has no following return date.')
            cost_factors.iloc[next_day] *= 1 - cost
        gross = (1 + returns) / cost_factors - 1
        result['costs'] = {
            'rebalances': len(trades), 'two_way_turnover': float(trades['turnover'].sum()),
            'mean_two_way_turnover': float(trades['turnover'].mean()) if len(trades) else 0.0,
            'gross_return_same_holdings': float((1 + gross).prod() - 1),
            'net_return': result['total_return'],
            'terminal_drag_pp': float(((1 + gross).prod() - (1 + returns).prod()) * 100),
        }
    history_path = directory / 'training_history.csv'
    if not history_path.exists():
        history_path = directory / 'logs/training_history.csv'
    if history_path.exists():
        history = pd.read_csv(history_path)
        best = int(history['val_loss'].to_numpy().argmin())
        result['training'] = {
            'best_epoch': best + 1, 'epochs': len(history),
            'train_loss_at_best': float(history.iloc[best]['train_loss']),
            'validation_loss_at_best': float(history.iloc[best]['val_loss']),
            'last_validation_loss': float(history.iloc[-1]['val_loss']),
            'train_loss_reduction': float(1 - history.iloc[-1]['train_loss'] / history.iloc[0]['train_loss']),
            'validation_loss_reduction': float(1 - history.iloc[-1]['val_loss'] / history.iloc[0]['val_loss']),
        }
    ic_path = directory / 'daily_ic.csv'
    if ic_path.exists():
        ic = pd.read_csv(ic_path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
        if len(ic):
            result['ic'] = {'mean': float(ic.mean()), 'days': len(ic),
                            'positive_ratio': float((ic > 0).mean()),
                            'monthly': {str(date.to_period('M')): float(value)
                                        for date, value in ic.resample('ME').mean().items() if pd.notna(value)}}
    return result
