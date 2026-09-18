"""Dashboard pages delegate persistence and charts to separate modules."""
import pandas as pd
import streamlit as st
from dashboard_data import demo_returns, read_returns
from research.experiments import ROOT, list_experiments
from ui.i18n import tr
from ui.results import show_results

def overview():
    st.title(tr('让每一次研究，都看得清楚', 'Every experiment, clearly explained'))
    st.write(tr('从真实行情到样本外结果，查看训练、风险与研究依据。',
                'From real market data to out-of-sample results: inspect training, risk and provenance.'))
    completed = [item for item in list_experiments() if item.get('state') == 'complete'
                 and (item['directory'] / 'backtest_returns.csv').exists()]
    candidates = [item['directory'] / 'backtest_returns.csv' for item in completed]
    candidates += [path.parent / 'backtest_returns.csv' for path in sorted((ROOT / 'reports').glob('*/summary.json'), reverse=True)
                   if (path.parent / 'backtest_returns.csv').exists()
                   and path.parent.name not in {item['id'] for item in completed}]
    modes = {'local': tr('本地实验', 'Local experiments'), 'upload': tr('上传 CSV', 'Upload CSV'),
             'demo': tr('演示数据', 'Demo data')}
    mode = st.radio(tr('结果来源', 'Result source'), list(modes), index=0 if candidates else 2,
                    format_func=modes.__getitem__, horizontal=True, key='source_mode')
    directory = None
    try:
        if mode == 'demo':
            st.warning(tr('演示数据为随机生成，仅用于体验界面，不代表真实模型表现。',
                          'Randomly generated demo data for UI exploration only; not actual model performance.'))
            returns = demo_returns()
            source = tr('演示数据 · 180 个工作日', 'Demo data · 180 business days')
        elif mode == 'upload':
            st.info(tr('CSV 需包含 date、return 两列；收益使用小数，0.01 表示 1%，应已扣费。',
                       'CSV requires date and return columns; use decimal net returns (0.01 = 1%).'))
            upload = st.file_uploader(tr('选择回测收益文件', 'Choose a returns file'), type=['csv'])
            if upload is None:
                return
            returns = read_returns(upload, language=tr('zh', 'en'))
            source = upload.name
        else:
            if (ROOT / 'backtest_returns.csv').exists():
                candidates.append(ROOT / 'backtest_returns.csv')
            if not candidates:
                st.info(tr('还没有完成的实验，请先开始研究或上传 CSV。', 'No completed experiments. Start a study or upload a CSV.'))
                return
            chosen = st.selectbox(tr('选择实验', 'Select experiment'), candidates, format_func=lambda path: path.parent.name)
            directory = chosen.parent
            returns = read_returns(chosen, language=tr('zh', 'en'))
            source = tr('本地实验：', 'Local experiment: ') + directory.name
        show_results(returns, source, directory)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(tr('无法读取结果：', 'Unable to load results: ') + str(error))

def stock_pool():
    st.title(tr('股票池', 'Stock universe'))
    st.write(tr('按代码或名称搜索。训练按文件顺序选取，搜索不影响训练。',
                'Search by ticker or name. Training uses file order and is unaffected by this filter.'))
    try:
        pool = pd.read_csv(ROOT / 'stock_pool.csv', dtype=str)
        if not {'code', 'name'}.issubset(pool.columns):
            raise ValueError('Stock pool requires code and name columns.')
        st.metric(tr('股票池总数', 'Universe size'), len(pool))
        query = st.text_input(tr('搜索代码或名称', 'Search ticker or name'), placeholder='000001 / 平安')
        filtered = pool[pool['code'].str.contains(query, regex=False, na=False) | pool['name'].str.contains(query, regex=False, na=False)]
        st.dataframe(filtered.rename(columns={'code': tr('股票代码', 'Ticker'), 'name': tr('股票名称', 'Company'),
                                              'date': tr('记录日期', 'Record date')}), hide_index=True, width='stretch')
        st.caption(f'{len(filtered)} / {len(pool)}')
        st.download_button(tr('下载股票池 CSV', 'Download stock universe'), pool.to_csv(index=False).encode('utf-8-sig'), 'stock_pool.csv', 'text/csv')
    except (OSError, ValueError) as error:
        st.error(str(error))
