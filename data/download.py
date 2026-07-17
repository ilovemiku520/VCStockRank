# data/download.py
import baostock as bs
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import time
import warnings

warnings.filterwarnings('ignore')

# ================= 全局配置 =================
DATA_START = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
DATA_END = datetime.now().strftime('%Y-%m-%d')
MIN_TRADING_DAYS = 180          # 剔除不足180个交易日的次新股
REQUEST_DELAY = 0.1             # 单股请求间隔（秒）
RETRY_TIMES = 3                 # 失败重试次数
MAX_STOCKS = 500                # 最多获取的股票数量
RETRY_SLEEP = 2                 # 重试等待时间（秒）

# ================= 1. 股票池加载 =================
def load_stock_pool(csv_path="stock_pool.csv", max_stocks=MAX_STOCKS):
    """从CSV加载股票池，返回带市场前缀的代码列表（仅前max_stocks只）"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"请准备 {csv_path}，包含 'code' 列")
    df = pd.read_csv(csv_path, dtype={'code': str})
    if 'code' not in df.columns:
        raise ValueError("CSV必须包含 'code' 列")

    # 剔除ST
    if 'name' in df.columns:
        df = df[~df['name'].str.contains('ST|\\*ST', case=False, na=False)]
    # 剔除北交所（8开头）
    df = df[~df['code'].str.startswith('8')]

    def add_prefix(code):
        code = code.zfill(6)
        if code.startswith(('6', '5', '7', '9')):
            return f"sh.{code}"
        else:
            return f"sz.{code}"

    stock_list = df['code'].apply(add_prefix).tolist()
    stock_list = stock_list[:max_stocks]
    print(f"有效股票数（截取前{max_stocks}只）: {len(stock_list)}")
    return stock_list

# ================= 2. 单股数据获取 =================
def fetch_daily_with_retry(code, start, end, retry=RETRY_TIMES):
    """带重试的查询，返回原始数据列表"""
    for attempt in range(retry):
        try:
            lg = bs.login()
            if lg.error_code != '0':
                bs.logout()
                time.sleep(RETRY_SLEEP)
                continue
            rs = bs.query_history_k_data_plus(
                code=code,
                fields="date,open,high,low,close,volume,amount,pctChg,turn",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="2"
            )
            if rs is None or rs.error_code != '0':
                bs.logout()
                time.sleep(RETRY_SLEEP)
                continue
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            bs.logout()
            if data:
                return data
            else:
                time.sleep(RETRY_SLEEP)
        except Exception as e:
            print(f"  第{attempt + 1}次尝试失败: {e}")
            time.sleep(RETRY_SLEEP)
    return None

def compute_rsi(series, period=14):
    """计算RSI指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# data/download.py 中的 process_daily_data 函数

def process_daily_data(data, _code=None):
    """
    将 baostock 原始数据转换为 DataFrame，计算基础技术因子。

    Parameters:
    -----------
    data : list of list
        从 baostock 获取的原始数据行
    _code : str, optional
        股票代码

    Returns:
    --------
    pd.DataFrame or None
        包含日期索引和 OCHLV、技术指标的 DataFrame；
        如果输入数据为空，返回 None。
    """
    if not data:
        return None

    # 列名与 baostock 查询字段对应
    df = pd.DataFrame(
        data,
        columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'turn']
    )

    # 日期列转为 datetime，并设为索引
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # ---------- 安全数值转换 ----------
    # 将各列转为数值类型，无法转换的变为 NaN，避免空字符串报错
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.sort_index(inplace=True)

    # 重命名换手率列
    df.rename(columns={'turn': 'turnover'}, inplace=True)

    # ---------- 基础技术因子 ----------
    # 20日已实现波动率（年化）
    df['rv_20d'] = df['pct_chg'].rolling(20).std() * np.sqrt(252)

    # 14日 RSI
    df['rsi_14'] = compute_rsi(df['close'], 14)

    # 简易收益率（用于后续标签构建）
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)

    # 返回需要的列（顺序固定，便于后续拼接）
    return df[[
        'open', 'high', 'low', 'close',
        'volume', 'amount', 'pct_chg', 'turnover',
        'rv_20d', 'rsi_14',
        'ret_1d', 'ret_5d', 'ret_20d'
    ]]

# ================= 3. 批量下载 =================
def download_all(stock_list, start, end):
    """批量下载所有股票日频数据，返回MultiIndex DataFrame"""
    all_dfs = []
    for code in tqdm(stock_list, desc="下载日频数据"):
        raw = fetch_daily_with_retry(code, start, end)
        if raw is None:
            continue
        df = process_daily_data(raw, code)
        if df is None:
            continue
        if len(df) >= MIN_TRADING_DAYS:
            df['stock'] = code
            df.set_index('stock', append=True, inplace=True)
            df.index.names = ['date', 'stock']
            all_dfs.append(df)
        time.sleep(REQUEST_DELAY)

    if not all_dfs:
        return pd.DataFrame()

    full = pd.concat(all_dfs)
    full.sort_index(level=['date', 'stock'], inplace=True)

    # ---------- 关键修复：提取唯一日期并格式化 ----------
    # 1. 获取日期级别（已是 DatetimeIndex）
    date_level = full.index.get_level_values('date')
    # 2. 提取唯一日期，并转为字符串格式
    unique_dates = date_level.unique()
    date_str_unique = unique_dates.strftime('%Y-%m-%d')
    # 3. 用唯一值设置第一级索引（level=0）
    full.index = full.index.set_levels(date_str_unique, level=0)
    # ----------------------------------------------------

    return full
# ================= 4. 宏观数据（已移至 macro.py 模块） =================
# 宏观数据获取请使用: from data.macro import fetch_macro_data
# 原因：宏观数据源(akshare)与股票数据源(baostock)不同，分开管理更清晰