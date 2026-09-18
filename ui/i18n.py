"""Language selection is presentation-only and never changes experiment data."""
import streamlit as st

def tr(zh, en):
    return en if st.session_state.get('language') == 'English' else zh

def stage_label(stage):
    labels = {'pending': ('等待启动', 'Pending'), 'initializing': ('初始化', 'Initializing'),
              'download': ('行情与特征', 'Market data & features'), 'training': ('模型训练', 'Training'),
              'prediction': ('测试集预测', 'Test inference'), 'backtest': ('组合回测', 'Backtest'),
              'evaluation': ('模型评估', 'Evaluation'), 'reporting': ('保存报告', 'Exporting report'),
              'complete': ('已完成', 'Complete'), 'failed': ('失败', 'Failed'),
              'running': ('运行中', 'Running'), 'stopped': ('已停止', 'Stopped')}
    return tr(*labels.get(stage, (stage, stage)))
