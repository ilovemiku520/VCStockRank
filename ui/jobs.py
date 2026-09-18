"""Training form and durable experiment status view."""
from datetime import date, timedelta
import importlib.util
from pathlib import Path
import subprocess
import pandas as pd
import streamlit as st
from research.experiments import create_experiment, launch_experiment, list_experiments, read_json, update_status
from ui.i18n import tr, stage_label

@st.fragment(run_every='3s')
def job_status():
    experiments = list_experiments()
    if not experiments:
        st.info(tr('尚无实验。', 'No experiments yet.'))
        return
    directory = st.selectbox(tr('实验记录', 'Experiment history'), [item['directory'] for item in experiments],
                             format_func=lambda item: item.name, key='job_history')
    status = read_json(directory / 'status.json', {})
    state = status.get('state', 'pending')
    label = stage_label(status.get('stage', 'pending'))
    if state == 'complete':
        st.success(tr('研究完成，可在研究概览查看结果。', 'Complete. Open Overview to view results.'))
    elif state == 'failed':
        st.error(f"{label}: {status.get('message', '')}")
    else:
        st.info(f'{stage_label(state)} · {label}')
    job = st.session_state.get('job')
    if job and job['directory'] == str(directory) and job['process'].poll() is None:
        if st.button(tr('停止本次研究', 'Stop this experiment')):
            job['process'].terminate()
            try:
                job['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                job['process'].kill()
                job['process'].wait(timeout=5)
            update_status(directory, 'stopped', 'stopped')
            st.rerun()
    elif job and job['directory'] == str(directory) and state == 'running':
        update_status(directory, 'failed', status.get('stage', 'pending'), 'Worker exited unexpectedly. See run.log.')
    history_path = directory / 'logs/training_history.csv'
    if history_path.exists():
        history = pd.read_csv(history_path)
        settings = read_json(directory / 'settings.json', {})
        total = settings.get('EPOCHS', 20)
        st.progress(min(len(history) / total, 1.0), text=f"{len(history)} / {total} " + tr('轮', 'epochs'))
        if state == 'complete' and len(history) < total:
            st.caption(tr('验证集早停已触发，完整训练流程已结束。', 'Validation early stopping was reached; the full training pipeline is complete.'))
        st.line_chart(history[['train_loss', 'val_loss']])
    log = directory / 'run.log'
    if log.exists():
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size - 12000))
            tail = handle.read().decode('utf-8', errors='replace')
        with st.expander(tr('原始运行日志（保留原文）', 'Raw execution log (original language)')):
            st.code(tail, language=None)

def research_page():
    st.title(tr('开始一次新的研究', 'Start a new experiment'))
    st.write(tr('独立保存行情、参数、模型与报告，自动选择可用 CUDA GPU。',
                'Save data, settings, models and reports separately. CUDA is selected automatically when available.'))
    needed = ['torch', 'scipy', 'sklearn', 'tqdm', 'baostock', 'pyarrow']
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing:
        st.warning(tr('训练依赖缺失，请执行：', 'Missing training dependencies. Run: ') + 'pip install -r requirements.txt')
    active = st.session_state.get('job')
    busy = active is not None and active['process'].poll() is None
    with st.form('new_research'):
        left, right = st.columns(2)
        start = left.date_input(tr('数据开始日期', 'Start date'), date.today() - timedelta(days=365 * 3))
        end = right.date_input(tr('数据结束日期', 'End date'), date.today() - timedelta(days=1), max_value=date.today())
        count = left.number_input(tr('最多使用股票数', 'Maximum stocks'), min_value=10, max_value=100, value=25, step=5)
        epochs = right.number_input(tr('训练轮数上限', 'Maximum epochs'), min_value=1, max_value=200, value=20)
        seed = left.number_input(tr('随机种子', 'Random seed'), min_value=0, max_value=2147483647, value=42)
        st.caption(tr('70% 训练 / 15% 验证 / 15% 测试；边界禁入 5 日；验证集早停；按日期分组排序。',
                      '70% train / 15% validation / 15% test; 5-day boundary embargo; validation early stopping; within-date ranking.'))
        submitted = st.form_submit_button(tr('开始训练与回测', 'Start training & backtest'), type='primary', disabled=bool(missing) or busy)
    if submitted:
        if start >= end or (end - start).days < 365:
            st.error(tr('请设置至少一年的日期范围。', 'Select a date range of at least one year.'))
        else:
            try:
                directory = create_experiment({'DATA_START': start.isoformat(), 'DATA_END': end.isoformat(),
                                                'MAX_STOCKS': count, 'EPOCHS': epochs, 'SEED': seed})
                st.session_state['job'] = launch_experiment(directory)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(str(error))
    job_status()
    st.caption(tr('关闭页面不会停止任务；停止按钮仅控制当前会话启动的任务。本机单用户使用。',
                  'Closing this page does not stop a job. The stop button controls jobs launched in this session. For local single-user use.'))
