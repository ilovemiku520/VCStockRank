# data/download.py
import baostock as bs
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import time
import warnings
import json
import socket
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_START = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
DATA_END = datetime.now().strftime('%Y-%m-%d')
MIN_TRADING_DAYS = 180
REQUEST_DELAY = 0.1
RETRY_TIMES = 3
MAX_STOCKS = 500
RETRY_SLEEP = 2

def load_stock_pool(csv_path="stock_pool.csv", max_stocks=MAX_STOCKS):
    """Read zero-padded tickers, filter excluded names and add exchange prefixes."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"请准备 {csv_path}，包含 'code' 列")
    df = pd.read_csv(csv_path, dtype={'code': str})
    if 'code' not in df.columns:
        raise ValueError("CSV必须包含 'code' 列")

    if 'name' in df.columns:
        df = df[~df['name'].str.contains('ST|\\*ST', case=False, na=False)]

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

def fetch_daily_with_retry(code, start, end, retry=RETRY_TIMES):
    """Query BaoStock with bounded socket timeouts and retry transient failures."""
    socket.setdefaulttimeout(20)
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
    """Calculate RSI from rolling positive and negative price changes."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_daily_data(data, _code=None):
    """Parse source rows and calculate basic OHLCV indicators."""
    if not data:
        return None

    df = pd.DataFrame(
        data,
        columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'turn']
    )

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.sort_index(inplace=True)

    df.rename(columns={'turn': 'turnover'}, inplace=True)

    df['rv_20d'] = df['pct_chg'].rolling(20).std() * np.sqrt(252)

    df['rsi_14'] = compute_rsi(df['close'], 14)

    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)

    return df[[
        'open', 'high', 'low', 'close',
        'volume', 'amount', 'pct_chg', 'turnover',
        'rv_20d', 'rsi_14',
        'ret_1d', 'ret_5d', 'ret_20d'
    ]]

def download_all(stock_list, start, end, cache_dir=None):
    """Download adjusted daily bars, cache source rows and persist data provenance."""
    all_dfs = []
    metadata = {'source': 'BaoStock', 'adjustflag': '2', 'adjustment': 'forward-adjusted',
                'requested_start': start, 'requested_end': end, 'stocks': [], 'failed_stocks': [],
                'retrieved_at': datetime.now().isoformat()}
    for code in tqdm(stock_list, desc="下载日频数据"):
        cache_path = Path(cache_dir) / f'{code}_{start}_{end}_qfq.json' if cache_dir else None
        cached = False
        raw = None
        if cache_path and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding='utf-8'))
                raw = payload['rows']
                cached = True
            except (ValueError, KeyError):
                pass
        if raw is None:
            raw = fetch_daily_with_retry(code, start, end)
            if raw and cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {'retrieved_at': datetime.now().isoformat(), 'rows': raw}
                cache_path.write_text(json.dumps(payload), encoding='utf-8')
        if raw is None:
            metadata['failed_stocks'].append(code)
            continue
        df = process_daily_data(raw, code)
        if df is None:
            continue
        if len(df) >= MIN_TRADING_DAYS:
            df['stock'] = code
            df.set_index('stock', append=True, inplace=True)
            df.index.names = ['date', 'stock']
            all_dfs.append(df)
            metadata['stocks'].append({'code': code, 'rows': len(df), 'cached': cached,
                                       'first_date': str(df.index.get_level_values('date').min().date()),
                                       'last_date': str(df.index.get_level_values('date').max().date()),
                                       'retrieved_at': payload['retrieved_at'] if cache_path else metadata['retrieved_at']})
        else:
            metadata['failed_stocks'].append(code)
        time.sleep(REQUEST_DELAY)

    Path('data').mkdir(exist_ok=True)
    Path('data/source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    if not all_dfs:
        return pd.DataFrame()

    full = pd.concat(all_dfs)
    full.sort_index(level=['date', 'stock'], inplace=True)

    # Preserve datetime levels throughout features, splits and inference.
    full.to_parquet('data/market.parquet')
    return full
