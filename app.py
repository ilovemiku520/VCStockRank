"""Chinese research dashboard. Launch with: streamlit run app.py."""
from datetime import date, timedelta, datetime
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pandas as pd
import altair as alt
import streamlit as st

from dashboard_data import demo_returns, performance, read_returns

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
st.set_page_config(page_title='VCStockRank · 研究工作台', page_icon='📈', layout='wide')
st.markdown('''<style>
.stApp {background: #f7f9fc;}
.block-container {padding-top:3rem;}
[data-testid="stMetric"] {background:white;padding:20px;border:1px solid #e3e8ef;border-radius:12px;}
h1,h2,h3 {letter-spacing:-0.03em;}
</style>''', unsafe_allow_html=True)

with st.sidebar:
    st.title('VCStockRank')
    st.caption('A 股量价研究工作台')
    page = st.radio('导航', ['研究概览', '开始研究', '股票池'], label_visibility='collapsed')
    st.divider()
    st.caption('从数据到结果，每一步都可检查。')
    st.caption('研究用途 · 不构成投资建议')


def show_results(returns, source):
    st.caption(source)
    selected = st.date_input('查看日期范围', (returns.index.min().date(), returns.index.max().date()),
                             min_value=returns.index.min().date(), max_value=returns.index.max().date())
    if len(selected) != 2:
        st.info('请选择完整的起止日期。')
        return
    returns = returns.loc[str(selected[0]):str(selected[1])]
    if returns.empty:
        st.info('该日期范围没有交易记录，请选择其他日期。')
        return
    metrics, curves = performance(returns)
    for column, (label, value) in zip(st.columns(4), metrics.items()):
        column.metric(label, '—' if value is None else (f'{value:.2f}' if label == '夏普比率' else f'{value:.2%}'))
    st.caption(f'{len(returns)} 个收益记录日 · 区间起始净值为 1 · 年化按 252 个交易日计算 · 夏普无风险利率 2%')
    st.caption('导入收益应为逐交易日、已扣除成本的组合收益；页面不再次扣费。短区间年化指标仅供参考。')
    equity_tab, drawdown_tab, table_tab = st.tabs(['净值走势', '回撤分析', '每日明细'])
    with equity_tab:
        chart_data = curves.rename_axis('日期').reset_index()
        chart = alt.Chart(chart_data).mark_line(color='#2563eb', strokeWidth=2).encode(
            x=alt.X('日期:T', title=None),
            y=alt.Y('净值:Q', scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('日期:T', format='%Y-%m-%d'), alt.Tooltip('净值:Q', format='.4f')],
        ).properties(height=330)
        st.altair_chart(chart, use_container_width=True)
    with drawdown_tab:
        chart = alt.Chart(chart_data).mark_area(color='#e07055', opacity=.65).encode(
            x=alt.X('日期:T', title=None),
            y=alt.Y('回撤:Q', axis=alt.Axis(format='.0%')),
            tooltip=[alt.Tooltip('日期:T', format='%Y-%m-%d'), alt.Tooltip('回撤:Q', format='.2%')],
        ).properties(height=330)
        st.altair_chart(chart, use_container_width=True)
    with table_tab:
        st.dataframe(pd.concat([returns.rename('每日收益'), curves], axis=1), use_container_width=True)
    st.download_button('下载当前区间收益 CSV', returns.rename_axis('date').to_csv().encode('utf-8-sig'),
                       file_name='backtest_returns.csv', mime='text/csv')


@st.fragment(run_every='2s')
def job_status():
    job = st.session_state.get('job')
    if not job:
        return
    directory = Path(job['directory'])
    process = job['process']
    code = process.poll()
    if code is None:
        st.info('研究正在运行，日志每 2 秒更新。你可以查看其他页面。')
        if st.button('停止本次研究'):
            process.terminate()
            process.wait(timeout=10)
            (directory / 'status.json').write_text(json.dumps({'state': 'stopped', 'message': '用户已停止任务。'}), encoding='utf-8')
            st.rerun()
    elif (directory / 'status.json').exists():
        status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
        if status['state'] == 'complete':
            st.success('研究完成。在「研究概览 → 本地实验」查看结果。')
        else:
            st.warning(status['message'])
    else:
        st.error(f'任务已退出（代码 {code}），请查看日志中的错误。')
    log = directory / 'run.log'
    if log.exists():
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size - 16000))
            tail = handle.read().decode('utf-8', errors='replace')
        with st.expander('运行日志', expanded=True):
            st.code(tail, language=None)
    st.caption(f'实验编号：{directory.name}')


if page == '研究概览':
    st.title('让每一次研究，都看得清楚')
    st.write('查看策略净值、风险与每日收益，或从一份已有结果开始。')
    mode = st.radio('结果来源', ['演示数据', '上传 CSV', '本地实验'], horizontal=True)
    returns = None
    source = ''
    try:
        if mode == '演示数据':
            st.warning('当前为随机生成的演示数据，仅用于体验界面，不代表模型表现或真实投资收益。')
            returns = demo_returns()
            source = '演示数据 · 固定随机种子 · 180 个工作日'
        elif mode == '上传 CSV':
            st.info('上传包含 date、return 两列的 CSV。return 使用小数，0.01 表示 1%。')
            upload = st.file_uploader('选择回测收益文件', type=['csv'])
            if upload is not None:
                returns = read_returns(upload)
                source = f'上传文件：{upload.name}'
        else:
            candidates = []
            if (ROOT / 'backtest_returns.csv').exists():
                candidates.append(ROOT / 'backtest_returns.csv')
            for status_path in sorted(RUNS.glob('*/status.json'), reverse=True):
                status = json.loads(status_path.read_text(encoding='utf-8'))
                result_path = status_path.parent / 'backtest_returns.csv'
                if status.get('state') == 'complete' and result_path.exists():
                    candidates.append(result_path)
            if not candidates:
                st.info('还没有完成的实验。前往「开始研究」运行任务，或上传已有回测 CSV。')
            else:
                chosen = st.selectbox('选择实验', candidates, format_func=lambda p: '原命令行结果' if p.parent == ROOT else p.parent.name)
                returns = read_returns(chosen)
                source = f'本地实验：{chosen.parent.name}'
                settings = chosen.parent / 'settings.json'
                if settings.exists():
                    with st.expander('本次实验参数'):
                        st.json(json.loads(settings.read_text(encoding='utf-8')))
        if returns is not None:
            show_results(returns, source)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        st.error(f'无法读取结果：{error}')

elif page == '开始研究':
    st.title('开始一次新的研究')
    st.write('设置数据范围与训练规模，自动完成数据准备、模型训练、回测和评估。')
    st.info('每次研究独立保存，不复用旧模型。首次运行需要下载行情；训练可能持续较长时间。')
    needed = ['torch', 'scipy', 'sklearn', 'tqdm', 'baostock', 'akshare', 'matplotlib', 'statsmodels']
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing:
        st.warning('训练依赖尚未安装，请在项目环境执行：pip install -r requirements.txt')
    active = st.session_state.get('job')
    busy = active is not None and active['process'].poll() is None
    with st.form('new_research'):
        left, right = st.columns(2)
        start = left.date_input('数据开始日期', date.today() - timedelta(days=365 * 3))
        end = right.date_input('数据结束日期', date.today(), max_value=date.today())
        count = left.number_input('最多使用股票数', min_value=10, max_value=100, value=25, step=5)
        epochs = right.number_input('训练轮数上限', min_value=1, max_value=200, value=20)
        st.caption('使用股票池前 N 只股票；训练/验证/测试按 70%/15%/15% 切分，标签边界保留 5 个交易日。组合设置沿用 config.py。')
        submitted = st.form_submit_button('开始训练与回测', type='primary', disabled=bool(missing) or busy)
    if submitted:
        if start >= end or (end - start).days < 365:
            st.error('请设置至少一年的日期范围，为训练、验证与测试保留足够历史。')
        elif not (ROOT / 'stock_pool.csv').exists():
            st.error('缺少 stock_pool.csv，请先恢复股票池文件。')
        else:
            directory = RUNS / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
            try:
                (directory / 'data').mkdir(parents=True)
                shutil.copy2(ROOT / 'stock_pool.csv', directory / 'stock_pool.csv')
                settings = {'DATA_START': start.isoformat(), 'DATA_END': end.isoformat(), 'MAX_STOCKS': count, 'EPOCHS': epochs}
                (directory / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
                with (directory / 'run.log').open('wb') as log:
                    process = subprocess.Popen([sys.executable, '-u', '-X', 'utf8', str(ROOT / 'dashboard_worker.py'), str(directory)],
                                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                st.session_state['job'] = {'directory': str(directory), 'process': process}
            except OSError as error:
                st.error(f'无法启动研究：{error}')
    job_status()
    st.caption('停止按钮用于当前浏览器会话。关闭页面不会终止后台任务；完成的结果会保留在本地实验中。')

else:
    st.title('股票池')
    st.write('检查研究覆盖范围，按股票代码或名称快速查找。')
    try:
        pool = pd.read_csv(ROOT / 'stock_pool.csv', dtype=str)
        if not {'code', 'name'}.issubset(pool.columns):
            raise ValueError('股票池需要 code 和 name 两列。')
        st.metric('股票池总数', len(pool))
        query = st.text_input('搜索代码或名称', placeholder='例如：000001 或 平安')
        filtered = pool[pool['code'].str.contains(query, regex=False, na=False) | pool['name'].str.contains(query, regex=False, na=False)]
        st.dataframe(filtered.rename(columns={'code': '股票代码', 'name': '股票名称', 'date': '记录日期'}), hide_index=True, use_container_width=True)
        st.caption(f'显示 {len(filtered)} / {len(pool)} 只。训练按原始文件顺序选取，不受此处搜索影响。')
        st.download_button('下载股票池 CSV', pool.to_csv(index=False).encode('utf-8-sig'), 'stock_pool.csv', 'text/csv')
    except (OSError, ValueError) as error:
        st.error(f'无法读取股票池：{error}')
