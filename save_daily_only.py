# save_daily_only.py
"""Standalone research entry point."""

from data.download import load_stock_pool, download_all
from config import ModelConfig

print("=" * 60)
print("生成最原始的日频价格数据 (不经过任何处理)")
print("=" * 60)

config = ModelConfig()
stocks = load_stock_pool("stock_pool.csv", max_stocks=config.MAX_STOCKS)
print(f"加载股票池: {len(stocks)} 只")

daily = download_all(stocks, config.DATA_START, config.DATA_END)
print(f"原始数据形状: {daily.shape}")

price_raw = daily[['close']].copy()
price_raw.to_parquet("data/daily_raw.parquet")
print(f"✓ 原始价格数据已保存至 data/daily_raw.parquet")
print(f"  close 列统计: 均值={price_raw['close'].mean():.4f}, 标准差={price_raw['close'].std():.4f}")
