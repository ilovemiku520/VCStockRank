# VCStockRank

[简体中文](README.md) | **English**

A Python prototype for A-share cross-sectional ranking research: real market-data downloads, causal features, GPU multitask training, out-of-sample backtests and a bilingual research dashboard.

> This is a research prototype, not a live-trading system. Historical simulations are not executable investment returns or investment advice.

## Completed real-data experiment

On 2026-09-18, an **NVIDIA GeForce RTX 3060 12GB** completed data preparation, training, best-checkpoint restoration, held-out inference, backtesting and evaluation. A separate checkpoint replay verified the saved results.

| Item | Observed result |
| --- | --- |
| Source / adjustment | BaoStock daily bars / forward-adjusted (adjustflag=2) |
| Requested and observed dates | 2023-09-18 to 2026-09-17 |
| Stocks / trading dates / daily records | 25 / 727 / 18,175 |
| Features / history window | 51 / 30 trading days, ending before the target date |
| Random seed | 42 |
| Train / validation / test samples | 11,825 / 2,600 / 2,625 |
| Training | Maximum 20 epochs; validation early stop at 19; best epoch 14 |
| Framework | PyTorch 2.8.0+cu126 |
| Return interval / observations | 2026-04-14 to 2026-09-17 / 109 days |
| Total return after costs and slippage | **-1.88%** |
| Annualized return / Sharpe | -4.29% / -0.161 |
| Maximum drawdown | **18.19%** |
| Mean Spearman IC / ICIR | 0.0477 / 0.1727 |

The initial market download took about 110 seconds. Feature preparation, training, backtesting and reporting using those cached rows took about 227 seconds. Runtime depends on local load. There are 2,750 prediction rows; the final five dates lack complete forward labels, so IC and labeled test samples cover fewer dates than trading signals.

[Summary and hashes](reports/real-20260918/summary.json) · [Daily returns](reports/real-20260918/backtest_returns.csv) · [Training history](reports/real-20260918/training_history.csv) · [Replay validation](reports/real-20260918/replay_validation.json)

The repository contains compact result artifacts, a universe snapshot and source metadata. Full market bars, features and weights remain in local `runs/` directories. Checkpoint replay reproduced 2,750 predictions and 109 daily returns with maximum absolute differences below 1e-12. This does not guarantee bitwise-identical retraining on different hardware.

## Open the dashboard

Python 3.11 or 3.12 is recommended. From the project directory:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-ui.txt
python -m streamlit run app.py
```

Open http://127.0.0.1:8501 . Use **语言 / Language** in the sidebar to switch between Simplified Chinese and English. README language links are at the top. Interface controls and explanations switch languages; company names, raw logs and experiment field names remain in their original form.

- **Overview:** real local or archived results by default, CSV imports, date filters, equity and equal-weight reference curves, drawdowns, training curves, daily IC and downloads.
- **New experiment:** configure dates, universe size, epoch limit and seed; inspect pipeline stages, training progress and experiment history.
- **Stock universe:** search and export tickers without losing leading zeros. Training filters excluded names such as ST stocks and uses the first N eligible rows; UI searches do not change the universe.

Viewing the interface and saved reports does not require PyTorch. Uploaded CSVs need `date,return` columns, one observation per day and decimal net returns (0.01 = 1%). Invalid dates, duplicates, empty data and non-finite returns are rejected. Random demo data is explicitly labeled.

## Real training and GPU setup

```bash
python -m pip install -r requirements.txt -r requirements-ui.txt
python -m streamlit run app.py
```

The pipeline selects CUDA when PyTorch can access it, otherwise CPU. Install a CUDA-enabled PyTorch build compatible with the driver first. This run used cu126, for example:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for other hardware. The core workflow does not need torchvision, AKShare or XGBoost. Optional explanation and regression plotting dependencies are in `requirements-extras.txt`.

The same isolated pipeline is available from the command line:

```bash
python run_experiment.py --start 2023-09-18 --end 2026-09-17 --stocks 25 --epochs 20 --seed 42
python verify_experiment.py runs/<experiment-id>
```

Each experiment trains from scratch in its own directory. Source rows are cached by ticker, date range and adjustment under `cache/market/`. Compatibility entry points `main.py` and `backtest_main.py` remain available. To backtest independently, run `python ../../backtest_main.py` from the experiment directory.

Closing the browser does not stop background training. The stop button controls jobs launched in the current session; completed artifacts remain available after reopening. This is a local, single-user dashboard bound to 127.0.0.1 by default.

## Evaluation protocol v3 and fixes

1. Split trading dates into 70% training, 15% validation and 15% test, purging five forward-label dates at both boundaries.
2. Historical windows end before the target date. No future backfilling is used. Panel feature clipping and normalization are cross-sectional by date; single-series processing uses expanding historical statistics.
3. Batches retain daily cross-sections. Ranking compares stocks only within the same date and excludes tied labels. Date metadata uses collatable strings.
4. Early stopping and checkpoint selection use validation loss only. Test dates are excluded from model selection. Checkpoints store feature order and model settings; both entry points share batched inference.
5. Defaults select the top 10 stocks, rebalance every five trading dates and cap a position at 10%. Weights drift with prices; infeasible allocation caps leave cash; dates without new signals still accrue returns.
6. Both buys and sells incur 0.1% costs plus 0.05% slippage on traded notional, including initial entry. Signals use a research close-to-next-close approximation. No pointless trade is placed on the last date.
7. Drawdown includes initial capital. Forward five-day group returns use non-overlapping windows. The dashboard reference is the same pool's daily equal-weight return before costs, **not CSI 300**.

Protocol v3 changes ranking and portfolio accounting; v1/v2 checkpoints must be retrained. Historical full-sample performance claims were withdrawn and are not comparable with this run.

## Code layout

```text
app.py                     Thin dashboard entry point
ui/                        Language helpers, pages, charts and job views
research/experiments.py     Experiment directories, processes and durable status
research/pipeline.py        Data, training, inference and evaluation orchestration
research/inference.py       Shared feature selection and batched predictions
research/runner.py          Staged execution and failure reporting
research/reporting.py       Results, provenance, environment and file hashes
data/                      BaoStock downloads, caching, cleaning and features
model/                     Decomposition, attention, temporal CNNs and task heads
training/                  Datasets, date-aware sampling, losses and trainer
portfolio/                 Allocation, holdings simulation and metrics
evaluation/                IC, grouped returns and optional explanations
reports/                   Compact, versioned evidence from real runs
runs/                      Local market data, weights, logs and results
tests/                     Temporal, ranking, portfolio, UI and GPU checks
```

Python comments and docstrings are concise English. Core, UI and optional-analysis dependencies are separate. Importing configuration no longer prints settings or globally changes CPU thread counts; runtime initialization explicitly sets threads and random seeds.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover date collation, within-date ranking, matching inference windows, weight drift, cash constraints, first-day costs, CSV validation, language switching and failed jobs. GPU forward/backward checks skip automatically on CPU-only hosts. CI runs on Windows and Linux.

## Known limitations

- A fixed present-day universe can introduce survivorship bias. These 25 stocks do not represent the entire market.
- Forward adjustments are as of download time; point-in-time constituents and corporate-action snapshots are unavailable.
- Close-price simulations omit limit-up/down execution, suspension constraints, tax differences and actual execution delays.
- Only one chronological split and one seed have been evaluated; full walk-forward robustness remains future work.
- VCFormer is a historical module name. Its current attention implementation is standard sequence-axis multihead attention, not evidence for a validated variable-axis-specific architecture.
- This experiment lost money. Treat it as an inspectable research baseline, not a return promise.

## License

[MIT License](LICENSE) · [@ilovemiku520](https://github.com/ilovemiku520)
