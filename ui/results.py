"""Result charts, provenance and artifact downloads."""
import altair as alt
import pandas as pd
import streamlit as st
from dashboard_data import performance, read_returns
from research.experiments import read_json
from ui.i18n import tr

def result_chart(frame, percentage=False):
    data = frame.rename_axis('date').reset_index().melt('date', var_name='series', value_name='value')
    chart = alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X('date:T', title=None, axis=alt.Axis(format='%m/%d')),
        y=alt.Y('value:Q', title=None, scale=alt.Scale(zero=False),
                axis=alt.Axis(format='.0%' if percentage else '.2f')),
        color=alt.Color('series:N', title=None, scale=alt.Scale(range=['#2563eb', '#98a6b8'])),
        tooltip=[alt.Tooltip('date:T', format='%Y-%m-%d'), 'series:N',
                 alt.Tooltip('value:Q', format='.2%' if percentage else '.4f')],
    ).properties(height=330)
    st.altair_chart(chart, width='stretch')

def show_results(returns, source, directory=None):
    st.caption(source)
    summary = read_json(directory / 'summary.json', {}) if directory else {}
    if summary:
        st.success(tr('真实行情 · 样本外测试 · 训练已完成', 'Real market data · Out-of-sample test · Training complete'))
        columns = st.columns(4)
        for col, label, value in zip(columns,
                [tr('股票 / 特征', 'Stocks / features'), tr('完成轮数 / 最优轮', 'Epochs / best epoch'),
                 tr('训练设备', 'Training device'), tr('测试样本数', 'Test samples')],
                [f"{summary['stock_count']} / {summary['feature_count']}",
                 f"{summary['epochs_completed']} / {summary['best_epoch']}",
                 summary.get('gpu') or summary['device'], summary['sample_counts']['test']]):
            col.caption(label)
            col.write(str(value))
    selected = st.date_input(tr('查看日期范围', 'Date range'),
                             (returns.index.min().date(), returns.index.max().date()),
                             min_value=returns.index.min().date(), max_value=returns.index.max().date())
    if len(selected) != 2:
        st.info(tr('请选择完整的起止日期。', 'Select both start and end dates.'))
        return
    returns = returns.loc[str(selected[0]):str(selected[1])]
    if returns.empty:
        st.info(tr('所选日期没有交易记录。', 'No records in the selected date range.'))
        return
    metrics, curves = performance(returns)
    for col, (label, value), english in zip(st.columns(4), metrics.items(),
            ['Total return', 'Annualized return', 'Max drawdown', 'Sharpe ratio']):
        col.metric(tr(label, english), '—' if value is None else (f'{value:.2f}' if label == '夏普比率' else f'{value:.2%}'))
    st.caption(tr('指标按所选区间重新计算；252 个交易日年化，夏普无风险利率 2%，初始本金计入回撤。',
                  'Metrics use the selected interval, 252 trading days/year and a 2% risk-free rate. Drawdown includes initial capital.'))
    equity, risk, details, diagnostics = st.tabs([tr('净值走势', 'Equity'), tr('回撤分析', 'Drawdown'),
                                                 tr('每日明细', 'Daily returns'), tr('训练与评估', 'Training & evaluation')])
    with equity:
        chart = curves[['净值']].rename(columns={'净值': tr('策略', 'Strategy')})
        if directory and (directory / 'benchmark_returns.csv').exists():
            benchmark = read_returns(directory / 'benchmark_returns.csv').reindex(returns.index)
            chart[tr('同股票池等权参考', 'Equal-weight reference')] = (1 + benchmark).cumprod()
            st.caption(tr('参考为同股票池每日等权收益，未扣费，不是沪深 300。',
                          'Reference: daily equal-weight of the same pool, before costs; not CSI 300.'))
        result_chart(chart)
    with risk:
        result_chart(curves[['回撤']].rename(columns={'回撤': tr('回撤', 'Drawdown')}), percentage=True)
    with details:
        st.dataframe(pd.concat([returns.rename(tr('每日收益', 'Daily return')), curves.rename(columns={
            '净值': tr('净值', 'Equity'), '回撤': tr('回撤', 'Drawdown')})], axis=1), width='stretch')
    with diagnostics:
        if not directory:
            st.info(tr('训练记录仅适用于本地实验。', 'Training records are available for local experiments.'))
        else:
            history_path = directory / 'logs/training_history.csv'
            if not history_path.exists():
                history_path = directory / 'training_history.csv'
            if history_path.exists():
                history = pd.read_csv(history_path)
                history.index = range(1, len(history) + 1)
                st.line_chart(history[['train_loss', 'val_loss']].rename(columns={
                    'train_loss': tr('训练损失', 'Train loss'), 'val_loss': tr('验证损失', 'Validation loss')}))
            ic_path = directory / 'daily_ic.csv'
            if ic_path.exists():
                ic = pd.read_csv(ic_path, index_col=0, parse_dates=True)
                st.caption(tr('每日 Spearman IC：预测分数与未来 5 日超额收益的横截面相关性。',
                              'Daily Spearman IC: scores vs. future 5-day excess returns within each trading date.'))
                st.line_chart(ic)
            evaluation = summary.get('evaluation', {})
            if evaluation:
                cols = st.columns(3)
                for col, key in zip(cols, ['ic_mean', 'icir', 'ic_positive_ratio']):
                    value = evaluation.get(key)
                    col.metric(key, '—' if value is None else f'{value:.4f}')
    st.download_button(tr('下载当前区间收益 CSV', 'Download selected returns'),
                       returns.rename_axis('date').to_csv().encode('utf-8-sig'), 'backtest_returns.csv', 'text/csv')
    if directory:
        with st.expander(tr('实验溯源与文件', 'Provenance & artifacts')):
            st.json(summary or read_json(directory / 'settings.json', {}))
            for filename in ['summary.json', 'logs/training_history.csv', 'daily_ic.csv', 'positions.csv', 'trades.csv', 'predictions.csv']:
                path = directory / filename
                if filename == 'logs/training_history.csv' and not path.exists():
                    path = directory / 'training_history.csv'
                if path.exists():
                    st.download_button(filename, path.read_bytes(), path.name, key=f'export-{directory.name}-{filename}')
    st.caption(tr('研究模拟不处理涨跌停、停牌成交约束或历史成分股变更，不代表可交易收益。',
                  'Research simulation omits limit-up/down execution, suspension constraints and historical universe changes. Returns are not live-trading results.'))
