import numpy as np
import pandas as pd
import pytest

from data.clean import DataCleaner
from data.features import FactorBuilder
from training.dataset import create_train_val_test_datasets
from portfolio.backtest import Backtester


def test_forward_fill_never_backfills_from_future():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["000001"]],
        names=["date", "stock"],
    )
    frame = pd.DataFrame({"close": [np.nan, 10.0, 11.0]}, index=index)

    cleaned = DataCleaner(winsorize_limits=None).clean(frame)

    assert np.isnan(cleaned.iloc[0]["close"])
    assert cleaned.iloc[1]["close"] == 10.0


def test_factor_at_cutoff_is_invariant_to_future_prices():
    dates = pd.date_range("2024-01-01", periods=130)
    base = np.linspace(10.0, 20.0, len(dates))
    frame = pd.DataFrame(
        {
            "open": base - 0.1,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base,
            "volume": np.linspace(1_000, 2_000, len(dates)),
            "turnover": np.linspace(0.01, 0.03, len(dates)),
        },
        index=dates,
    )
    changed = frame.copy()
    changed.loc[dates[111]:, "close"] *= 10
    builder = FactorBuilder(winsorize=False, standardize=False)

    original = builder.build_all_factors(frame)
    with_future_changed = builder.build_all_factors(changed)

    pd.testing.assert_series_equal(
        original.loc[dates[110]],
        with_future_changed.loc[dates[110]],
        check_names=False,
    )


def test_chronological_split_has_label_embargo():
    dates = pd.date_range("2024-01-01", periods=100)
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "stock"])
    frame = pd.DataFrame(
        {
            "feature": np.arange(len(index), dtype=float),
            "excess_ret_5d": 0.01,
            "future_vol_5d": 0.02,
        },
        index=index,
    )

    train, validation, test = create_train_val_test_datasets(
        frame,
        ["feature"],
        seq_len=10,
        train_ratio=0.7,
        val_ratio=0.15,
        embargo_days=5,
        normalize=False,
    )
    metadata = train.split_metadata

    assert metadata["train_end"] < metadata["validation_start"]
    assert metadata["validation_end"] < metadata["test_start"]
    assert (metadata["validation_start"] - metadata["train_end"]).days >= 5
    assert max(sample["date"] for sample in train.samples) == metadata["train_end"]
    assert min(sample["date"] for sample in test.samples) == metadata["test_start"]


def test_backtest_applies_signal_to_next_period_return():
    class Config:
        TOP_K = 1
        REBALANCE_FREQ = 1
        MAX_WEIGHT = 1.0
        TRANSACTION_COST = 0.0
        SLIPPAGE = 0.0

    dates = pd.date_range("2024-01-01", periods=3)
    stocks = ["A", "B", "C", "D", "E"]
    index = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock"])
    prices = pd.DataFrame({"close": 10.0}, index=index)
    prices.loc[(dates[1], "A"), "close"] = 20.0
    prices.loc[(dates[2], "A"), "close"] = 20.0
    predictions = {
        date: {
            stock: {"score": 10.0 if stock == "A" else 0.0, "vol": 1.0}
            for stock in stocks
        }
        for date in dates
    }

    result = Backtester(Config(), verbose=False).run(predictions, prices)

    assert result.returns.index[0] == dates[1]
    assert result.returns.iloc[0] == pytest.approx(1.0)
