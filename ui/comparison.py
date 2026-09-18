"""Compare explicitly matched experiments, without selecting a winner on test data."""
import pandas as pd
import streamlit as st
from research.experiments import ROOT, read_json
from ui.i18n import tr


def comparison_page():
    st.title(tr('1 / 3 / 5 日真实对照', '1 / 3 / 5-day real comparison'))
    st.write(tr('隔日版对应 1 日，波段版对应 3 日或 5 日。所有版本使用日线收盘到收盘的研究模拟，不是分钟级日内或纯隔夜交易。',
                'Overnight uses 1 day; Swing uses 3 or 5 days. These are daily close-to-close research simulations, not intraday or pure overnight-gap strategies.'))
    paths = {p.name: p for folder in ['reports', 'runs'] for p in (ROOT / folder).glob('shortline-*d')
             if (p / 'summary.json').exists() and (p / 'analysis.json').exists()}
    groups = sorted({name.rsplit('-', 1)[0] for name in paths}, reverse=True)
    if not groups:
        st.info(tr('真实对照尚在运行。可先到研究概览切换两种演示版本。',
                   'The real comparison is not ready yet. Explore both demo versions in Overview.'))
        return
    group = st.selectbox(tr('对照批次', 'Comparison batch'), groups)
    rows = []
    for horizon in (1, 3, 5):
        path = paths.get(f'{group}-{horizon}d')
        if not path:
            continue
        a = read_json(path / 'analysis.json', {})
        summary = read_json(path / 'summary.json', {})
        rows.append({tr('周期（日）', 'Horizon (days)'): horizon,
                     tr('净收益', 'Net return'): a['total_return'],
                     tr('等权参考（未扣费）', 'Reference (gross)'): a.get('reference_return'),
                     tr('最大回撤', 'Max drawdown'): a['max_drawdown'],
                     tr('IC 均值', 'Mean IC'): a.get('ic', {}).get('mean'),
                     tr('费用影响（百分点）', 'Cost drag (pp)'): a.get('costs', {}).get('terminal_drag_pp'),
                     tr('调仓次数', 'Rebalances'): a.get('costs', {}).get('rebalances'),
                     tr('最优轮数', 'Best epoch'): summary.get('best_epoch'),
                     tr('起始日', 'Start'): a['start'], tr('结束日', 'End'): a['end']})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption(tr('收益、回撤和 IC 使用小数；费用影响使用百分点。各周期标签不同，IC 不能简单视作同一个预测任务的排名。',
                  'Returns, drawdown and IC use decimals; cost drag uses percentage points. Different horizons have different labels, so IC values measure different prediction tasks.'))
    st.warning(tr('该对照复用了已查看的测试区间，只能作为探索性结果。不得根据此表调参后再把同一区间称为全新样本外验证。',
                  'This comparison reuses an already inspected test interval and is exploratory. Tuning from this table requires fresh unseen dates for confirmation.'))
    st.markdown(tr('**比较条件**：同一股票池、下载日期、种子 42、51 个特征、30 日历史窗口、70/15/15 时间切分；三个版本统一禁入 5 日。仅预测周期、风险标签窗口和调仓频率不同。协议 v4 风险标签为未来日收益的均方根，旧 v3 实验单独保留。',
                  '**Matched conditions**: same universe, data dates, seed 42, 51 features, 30-day history and 70/15/15 split, with a common 5-day embargo. Only prediction horizon, risk-label window and rebalancing frequency differ. Protocol v4 uses forward RMS daily returns as risk targets; legacy v3 results remain separate.'))
    algorithm_comparison()


def algorithm_comparison():
    paths = {p.parent.name: p.parent for folder in ['reports', 'runs']
             for p in (ROOT / folder).glob('*/summary.json')}
    pairs = {name: read_json(path / 'summary.json', {}).get('source_experiment')
             for name, path in paths.items() if name.startswith('statistical-')}
    pairs = {name: source for name, source in pairs.items() if source in paths}
    if not pairs:
        return
    st.subheader(tr('多元统计模型对照', 'Multivariate model comparison'))
    name = st.selectbox(tr('相同数据与切分的模型对照', 'Models using the same data and splits'), sorted(pairs, reverse=True))
    rows = []
    for experiment in [pairs[name], name]:
        summary = read_json(paths[experiment] / 'summary.json', {})
        a = read_json(paths[experiment] / 'analysis.json', {})
        if not a:
            continue
        rows.append({tr('算法', 'Algorithm'): summary.get('algorithm', 'VCformer-TPA'),
                     tr('股票数', 'Stocks'): summary['stock_count'],
                     tr('周期', 'Horizon'): summary['settings']['LABEL_HORIZON'],
                     tr('净收益', 'Net return'): a['total_return'],
                     tr('最大回撤', 'Max drawdown'): a['max_drawdown'],
                     tr('IC 均值', 'Mean IC'): a.get('ic', {}).get('mean'),
                     tr('费用影响（百分点）', 'Cost drag (pp)'): a.get('costs', {}).get('terminal_drag_pp')})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption(tr('PCA 使用训练集解释 95% 方差的主成分；岭回归按验证 IC 选择惩罚系数；风险使用目标日前的 EWMA。默认前 10 只与单股 10% 上限会使满仓权重接近等权，不能把收益差单独归因于风险估计。', 'PCA retains 95% of training variance; ridge penalties are selected by validation IC; EWMA risk uses only pre-signal history. With top-10 holdings and a 10% cap, fully invested weights are effectively equal; return differences cannot be attributed to the risk estimator alone.'))
