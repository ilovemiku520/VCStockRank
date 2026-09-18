"""Dashboard data validation; deliberately independent of the training stack."""
import numpy as np
import pandas as pd

def read_returns(source, language='zh'):
    def fail(zh, en):
        raise ValueError(en if language == 'en' else zh)
    frame = pd.read_csv(source)
    if not {'date', 'return'}.issubset(frame.columns):
        fail('CSV 需要 date 和 return 两列；收益使用小数，例如 0.01 表示 1%。',
             'CSV requires date and return columns; use decimal returns (0.01 = 1%).')
    dates = pd.to_datetime(frame['date'], errors='coerce')
    values = pd.to_numeric(frame['return'], errors='coerce')
    if frame.empty or dates.isna().any() or not np.isfinite(values).all():
        fail('数据不能为空，日期和收益必须完整有效。', 'Data must contain valid dates and finite returns.')
    if dates.duplicated().any():
        fail('每个日期只能有一条收益记录，请先处理重复日期。', 'Duplicate dates found; use one return record per date.')
    if (values < -1).any():
        fail('单日收益不能低于 -100%，请检查收益单位。', 'Daily returns cannot be below -100%; check the units.')
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name='return').sort_index()

def performance(returns):
    equity = (1 + returns).cumprod()
    # Include the initial capital: a loss on the first day is also a drawdown.
    peak = equity.cummax().clip(lower=1.0)
    drawdown = equity / peak - 1
    volatility = returns.std()
    metrics = {
        '累计收益': equity.iloc[-1] - 1,
        '年化收益': equity.iloc[-1] ** (252 / len(returns)) - 1,
        '最大回撤': drawdown.min(),
        '夏普比率': (np.sqrt(252) * (returns.mean() - .02 / 252) / volatility
                     if len(returns) > 1 and volatility > 0 else None),
    }
    return metrics, pd.DataFrame({'净值': equity, '回撤': drawdown})

def demo_returns(horizon=5):
    rng = np.random.default_rng(42 + horizon)
    return pd.Series(rng.normal(.0003, .009, 180),
                     index=pd.bdate_range('2025-01-02', periods=180), name='return')
