"""Bilingual dashboard entry point: streamlit run app.py."""
import streamlit as st
from ui.i18n import tr
from ui.pages import overview, stock_pool
from ui.jobs import research_page
from ui.comparison import comparison_page

st.set_page_config(page_title='VCStockRank', page_icon='📈', layout='wide')
st.markdown('''<style>
.stApp {background: #f7f9fc;}
.block-container {padding-top:3rem;}
[data-testid="stMetric"] {background:white;padding:18px;border:1px solid #e3e8ef;border-radius:12px;}
[data-testid="stMetricValue"] {font-size:clamp(1.15rem,2.1vw,1.75rem);}
h1,h2,h3 {letter-spacing:-0.03em;}
</style>''', unsafe_allow_html=True)
with st.sidebar:
    st.title('VCStockRank')
    st.selectbox('语言 / Language', ['简体中文', 'English'], key='language')
    st.caption(tr('A 股量价研究工作台', 'A-share quantitative research'))
    labels = {'overview': tr('研究概览', 'Overview'), 'research': tr('开始研究', 'New experiment'),
              'stocks': tr('股票池', 'Stock universe'), 'comparison': tr('版本对照', 'Compare versions')}
    page = st.radio(tr('导航', 'Navigation'), list(labels), format_func=labels.__getitem__,
                    label_visibility='collapsed', key='navigation')
    st.divider()
    st.caption(tr('研究用途 · 不构成投资建议', 'Research only · Not investment advice'))
{'overview': overview, 'research': research_page, 'stocks': stock_pool, 'comparison': comparison_page}[page]()
