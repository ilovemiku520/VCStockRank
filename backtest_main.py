# backtest_main.py
"""Standalone research entry point."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import ModelConfig, PortfolioConfig
from research.horizons import portfolio_for
from research.inference import predict_panel
from research.runtime import configure_runtime
from research.checkpoints import restore_config
from research.experiments import read_json
from model.multitask import MultiTaskVCformerTPA
from portfolio.backtest import Backtester
from evaluation.factor_analysis import FactorAnalyzer

def load_model_and_data(model_path='logs/best_model.pt', factor_path='data/factors_filled.csv'):
    print("\n[1/4] 加载模型与因子数据...")
    if not os.path.exists(factor_path):
        raise FileNotFoundError(f"因子数据文件不存在: {factor_path}")
    factors = pd.read_csv(factor_path, index_col=[0, 1], parse_dates=[0])
    print(f"  ✓ 因子数据加载成功，形状: {factors.shape}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    config = ModelConfig()
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    if checkpoint.get('protocol_version') not in (3, 4):
        raise ValueError("checkpoint 来自旧评估协议，请先运行 main.py 重新训练")
    summary_path = Path(factor_path).resolve().parent.parent / 'summary.json'
    restore_config(config, checkpoint, read_json(summary_path, {}).get('settings'))
    configure_runtime(config)

    model = MultiTaskVCformerTPA(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    print("  ✓ 模型加载成功")
    return model, config, factors

def generate_predictions(model, config, factors):
    return predict_panel(model, config, factors, config.FEATURE_COLS, config.TEST_START_DATE)

def get_price_data(factors):
    raw_path = "data/daily_raw.parquet"
    if os.path.exists(raw_path):
        price_data = pd.read_parquet(raw_path)[['close']].copy()
        print("  ✓ 使用原始价格数据 'data/daily_raw.parquet' (未经任何处理)")
        print(f"    close 均值={price_data['close'].mean():.4f}, 标准差={price_data['close'].std():.4f}")
        return price_data

    daily_path = "data/daily.parquet"
    if os.path.exists(daily_path):
        price_data = pd.read_parquet(daily_path)[['close']].copy()
        print("  ✓ 使用日频数据 'data/daily.parquet'")
        return price_data

    if 'close' in factors.columns:
        print("  ⚠ 使用因子数据中的 close 列（可能被标准化）")
        return factors[['close']].copy()
    else:
        raise FileNotFoundError("未找到价格数据，请运行 save_daily_only.py 生成。")

def run_backtest_and_evaluate(predictions, price_data, factors, portfolio_config, horizon=5):
    print("\n[3/4] 执行回测...")

    price_stocks = set(price_data.index.get_level_values('stock').unique())
    filtered_predictions = {}
    for date, preds in predictions.items():
        filtered = {}
        for stock, score, vol in zip(preds['stocks'], preds['scores'], preds['vols']):
            if stock in price_stocks:
                filtered[stock] = {'score': float(score), 'vol': float(vol)}
        if len(filtered) >= 5:
            filtered_predictions[date] = filtered
        else:
            print(f"  日期 {date} 有效股票数 {len(filtered)} < 5，跳过")

    print(f"  过滤后交易日数: {len(filtered_predictions)}")
    if len(filtered_predictions) == 0:
        raise ValueError("过滤后无有效交易日，请检查股票代码是否与价格数据匹配。")

    backtester = Backtester(portfolio_config, verbose=True)
    result = backtester.run(filtered_predictions, price_data)
    metrics = result.metrics

    print("\n[4/4] 计算评估指标...")

    target = 'target_excess' if 'target_excess' in factors else 'excess_ret_5d'
    temp_factors = factors[[target]].rename(columns={target: 'excess_ret_5d'}).copy()

    for date, preds in filtered_predictions.items():
        if date not in temp_factors.index.get_level_values('date'):
            continue
        for stock, info in preds.items():
            try:
                temp_factors.loc[(date, stock), 'pred_score'] = info['score']
            except:
                pass

    temp_factors = temp_factors.dropna(subset=['pred_score', 'excess_ret_5d'])

    print("  计算信息系数 (IC)...")
    if not temp_factors.empty:
        fa = FactorAnalyzer(temp_factors)
        ic_series = fa.compute_ic(factor_col='pred_score', ret_col='excess_ret_5d', method='spearman')
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else np.nan
        ic_positive_ratio = (ic_series > 0).mean()
        print(f"    ✓ IC 均值: {ic_mean:.4f}, ICIR: {icir:.4f}")
    else:
        ic_series = pd.Series(dtype=float)
        ic_mean = ic_std = icir = ic_positive_ratio = np.nan
        print("    ⚠ IC 分析失败：无有效数据")

    print(f"  计算分层收益（基于未来 {horizon} 日超额收益）...")
    if not temp_factors.empty:
        try:
            decile_returns, long_short = fa.compute_decile_returns(
                factor_col='pred_score',
                ret_col='excess_ret_5d',
                n_groups=10,
                period=horizon
            )
            decile_metrics = {}
            for col in decile_returns.columns:
                rets = decile_returns[col].dropna()
                if len(rets) > 0:
                    annual_ret = (1 + rets).prod() ** ((252 / horizon) / len(rets)) - 1
                    sharpe = rets.mean() / rets.std() * np.sqrt(252 / horizon) if rets.std() > 0 else np.nan
                    decile_metrics[col] = {'annual_return': annual_ret, 'sharpe': sharpe}
            long_short_rets = long_short.dropna()
            if len(long_short_rets) > 0:
                ls_annual = (1 + long_short_rets).prod() ** ((252 / horizon) / len(long_short_rets)) - 1
                ls_sharpe = long_short_rets.mean() / long_short_rets.std() * np.sqrt(252 / horizon) if long_short_rets.std() > 0 else np.nan
                decile_metrics['long_short'] = {'annual_return': ls_annual, 'sharpe': ls_sharpe}
            print(f"    ✓ 分层收益计算完成")
        except Exception as e:
            print(f"    ⚠ 分层收益计算失败: {e}")
            decile_metrics = {}
            long_short = pd.Series(dtype=float)
    else:
        decile_metrics = {}
        long_short = pd.Series(dtype=float)

    results = {
        'backtest_metrics': metrics,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': ic_positive_ratio,
        'daily_ic': ic_series,
        'long_short_returns': long_short,
        'decile_metrics': decile_metrics,
        'returns': result.returns,
    }
    return results

def main():
    print("=" * 60)
    print("回测与评估（独立脚本）")
    print("=" * 60)

    try:
        model, config, factors = load_model_and_data()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    predictions = generate_predictions(model, config, factors)
    if not predictions:
        print("无有效预测结果，终止。")
        return

    print("\n准备价格数据...")
    price_data = get_price_data(factors)
    print(f"  ✓ 价格数据形状: {price_data.shape}")

    portfolio_config = portfolio_for(config)
    results = run_backtest_and_evaluate(predictions, price_data, factors, portfolio_config, config.LABEL_HORIZON)

    metrics = results['backtest_metrics']
    print("\n" + "=" * 60)
    print("回测绩效指标")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    print("\n" + "=" * 60)
    print("信息系数 (IC) 分析")
    print("=" * 60)
    print(f"IC 均值     : {results['ic_mean']:.4f}")
    print(f"IC 标准差   : {results['ic_std']:.4f}")
    print(f"ICIR        : {results['icir']:.4f}")
    print(f"IC 正收益比率: {results['ic_positive_ratio']:.2%}")

    if results['decile_metrics']:
        print("\n" + "=" * 60)
        print("分层组合（十分位）年化收益与夏普（基于日收益）")
        print("=" * 60)
        for group, m in results['decile_metrics'].items():
            print(f"{group:12s}: 年化收益 {m.get('annual_return', np.nan):.2%}, 夏普 {m.get('sharpe', np.nan):.3f}")

    returns = results['returns']
    if not returns.empty:
        returns_df = pd.DataFrame({'date': returns.index, 'return': returns.values})
        returns_df.to_csv('backtest_returns.csv', index=False)
        print("\n回测收益序列已保存至 backtest_returns.csv")

    daily_ic = results['daily_ic']
    if not daily_ic.empty:
        daily_ic.to_csv('daily_ic.csv', header=True)
        print("每日 IC 已保存至 daily_ic.csv")

    if 'long_short_returns' in results and not results['long_short_returns'].empty:
        ls_df = pd.DataFrame({'date': results['long_short_returns'].index,
                              'long_short_return': results['long_short_returns'].values})
        ls_df.to_csv('long_short_returns.csv', index=False)
        print("多空组合收益已保存至 long_short_returns.csv")

    print("\n✅ 评估完成。")

if __name__ == "__main__":
    from research.console import configure_console
    configure_console()
    main()
