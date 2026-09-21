<p align="center"><img src="assets/readme-hero.svg" alt="VCStockRank — A-share ranking research workbench" width="100%"></p>

<p align="center"><strong>Real market data · Deep and statistical models · Replayable backtests</strong></p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11 or 3.12"></a>
  <a href="#methods"><img src="https://img.shields.io/badge/PyTorch-GPU%20ready-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch with optional GPU acceleration"></a>
  <a href="#overview"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&amp;logo=streamlit&amp;logoColor=white" alt="Streamlit interface"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-All%20Rights%20Reserved-4c9579?style=flat-square" alt="All rights reserved"></a>
  <a href="https://github.com/ilovemiku520/VCStockRank/stargazers"><img src="https://img.shields.io/github/stars/ilovemiku520/VCStockRank?style=flat-square&amp;color=d0ac60" alt="GitHub stars"></a>
</p>

<p align="center"><a href="README.md">简体中文</a> | <strong>English</strong></p>
<p align="center"><a href="#overview">Overview</a> · <a href="#quick-start">Quick start</a> · <a href="#results">Results</a> · <a href="#methods">Methods</a> · <a href="#documentation">Documentation</a></p>

> **Star first:** before using, referencing, or learning from this project—including AI use and learning—star this repository. [Full notice](#usage-notice).

<a id="overview"></a>
## A workbench for inspectable research

VCStockRank is an A-share cross-sectional ranking research prototype. Compare GPU deep models with PCA-ridge baselines, inspect risk and trading costs, and replay saved experiments in a bilingual workbench.

| Model comparison | Inspectable evidence | Interactive analysis |
|---|---|---|
| Deep ranking + PCA-ridge / EWMA | Saved settings, reports and replay checks | Equity, drawdown, IC and cost analysis |
| Matched 1 / 3 / 5-day horizons | Seeded sampling and time-based splits | Real reports, CSV import and labeled demos |

![Actual English workbench: saved 25-stock PCA-ridge pilot report](assets/workbench-en.jpg)

<sub>Actual interface showing the saved 25-stock pilot. The expanded 225-stock study is listed separately below. Research simulation only; no live orders or established persistent return advantage.</sub>

**Research horizons:** 1 trading day (Overnight) and 3 / 5 trading days (Swing). “Overnight” means daily close-to-close simulation, not intraday trading or overnight-gap-only returns. The 5-day model is not a long-term investment model.

<a id="quick-start"></a>
## Quick start

Python 3.11 / 3.12. Saved reports and synthetic demos work without PyTorch, a GPU, a market account, or downloading fresh market data.

```bash
git clone https://github.com/ilovemiku520/VCStockRank.git
cd VCStockRank
python -m venv .venv
```

| Activate the environment | Command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

```bash
python -m pip install -r requirements-ui.txt
python -m streamlit run app.py
```

Open **http://127.0.0.1:8501** and select a saved experiment in Overview. [Guided tour and troubleshooting](docs/DEMO.md).

<details>
<summary><strong>Workbench controls and CSV format</strong></summary>

1. Switch **语言 / Language** in the sidebar; the README links switch documentation language.
2. In **Overview**, choose Overnight or Swing, then real experiments, uploaded CSV or synthetic demo.
3. Inspect equity, drawdown, monthly returns, cost drag, training, IC and research recommendations; download evidence.
4. **Version comparison** shows real 1 / 3 / 5-day runs and matched deep/statistical model comparisons.
5. **Start research** shows the statistical requirement, adjustable capacity budget, planned sample and precision shortfall. Completed deep runs offer a one-click PCA-ridge comparison.

Demos have fixed seeds and are explicitly synthetic. Uploaded CSVs require `date,return`, one row per day, with decimal net returns (`0.01` = 1%). Duplicate dates, empty data and nonfinite values are rejected. Stock names, raw logs and some artifact fields retain their original language.

</details>

<a id="results"></a>
## Real results and their limits

### Expanded study · 225 stocks · 1-day horizon

A seeded random draw selected **227 stocks** from the 4,992-stock eligible frame. **225 stocks** supplied sufficient histories, totaling **160,526 rows**; `sz.001369` and `sh.603407` remained recorded as missing/insufficient rather than being replaced. GPU training completed 20 epochs, selecting epoch 16. Replay checked 24,750 predictions and 109 daily returns.

| Same 225 stocks / 1-day horizon | Net return | Max drawdown magnitude | Cost drag (pp) |
| --- | ---: | ---: | ---: |
| [Deep model](reports/20260919-002021-54513a/summary.json) | −14.57% | 29.33% | 11.32 |
| [PCA-ridge](reports/statistical-scientific-20260919-1d/summary.json) | −0.27% | 34.13% | 12.60 |
| [PCA-ridge + validation-selected holding buffer](reports/statistical-buffered-scientific-20260919-1d/summary.json) | **+6.53%** | **29.80%** | 5.51 |

Validation net return selected buffer **20** from 0 / 5 / 10 / 20. Mean two-way turnover fell from 72.73% to 30.85%. This is exploratory improvement: **drawdown remains nearly 30%, with no established persistent edge**. The block interval for mean daily return differences still includes zero. The buffer hypothesis followed inspection of earlier test costs, so these dates are not fresh independent confirmation. See the [full analysis](docs/EXPERIMENT_ANALYSIS.md).

<details>
<summary><strong>25-stock pilot · matched 1 / 3 / 5-day experiments and return curves</strong></summary>

![Real-data comparison of pilot models and horizons](assets/real-comparison.svg)

Generated from committed return files. Blue is the deep model, green is the statistical model; the dashed reference is before costs.

Three protocol-v4 GPU experiments and three statistical baselines completed on September 19, 2026 (Beijing time), with independent checkpoint replay. All used the same 25-stock pilot pool, 727 dates and 18,175 forward-adjusted BaoStock daily bars from 2023-09-18 through 2026-09-17. The return interval contains 109 days, 2026-04-14 through 2026-09-17.

| Method | 1-day net return | 3-day net return | 5-day net return |
| --- | ---: | ---: | ---: |
| VCformer-TPA deep model | −5.38% | −1.88% | −13.31% |
| PCA-ridge + historical EWMA risk | +0.15% | +4.01% | −0.52% |
| Same-pool daily equal-weight reference, before costs | −7.30% | −7.30% | −7.30% |

These are **exploratory results**: a legacy first-25 pool, one seed and an already-inspected test period. Do not choose a model from this table and call the same dates independent final validation. Block intervals for the statistical models’ mean daily differences include zero; a stable advantage has not been established. See [costs, drawdowns and monthly analysis](docs/EXPERIMENT_ANALYSIS.md).

[1-day deep](reports/shortline-20260919-1d/summary.json) · [3-day deep](reports/shortline-20260919-3d/summary.json) · [5-day deep](reports/shortline-20260919-5d/summary.json) · [1-day statistical](reports/statistical-shortline-20260919-1d/summary.json) · [3-day statistical](reports/statistical-shortline-20260919-3d/summary.json) · [5-day statistical](reports/statistical-shortline-20260919-5d/summary.json)

The legacy protocol-v3 run returning −1.88% remains in [real-20260918](reports/real-20260918/summary.json). Protocol v4 changes risk labels to horizon-specific forward RMS daily returns, allowing a valid 1-day target. Comparing v3 with v4 does not isolate holding period alone.

</details>

<a id="methods"></a>
## Models and reproducibility

**Historical data → causal features → time splits → model comparison → portfolio simulation → saved evidence.**

<details>
<summary><strong>Algorithms and scientific sampling</strong></summary>

- **Deep model:** 51 causal features, a 30-day historical window, within-date ranking and an auxiliary future-risk task. CUDA is used automatically when available.
- **Statistical model:** last value, mean and standard deviation form 153 historical inputs. Train-only standardization and full-SVD PCA retain 95% of feature variance; SVD ridge handles collinearity with validation-selected penalties. Historical EWMA estimates risk; moving blocks describe time-dependent uncertainty. See [formulas and limitations](docs/ALGORITHMS.md).
- **New sampling:** seeded simple random sampling without replacement; count is the smaller of statistical requirement and adjustable capacity budget. Defaults: 95% confidence, ±5 percentage-point proportion margin, worst-case proportion 50%. The 4,992-stock eligible frame requires 357 stocks. The current 227-stock budget plans about ±6.36 pp, missing the target. This is stock-pool proportion planning, **not return confidence**.

</details>

<details>
<summary><strong>Run training and replay an experiment</strong></summary>

```bash
python -m pip install -r requirements.txt -r requirements-ui.txt
python run_experiment.py --start 2023-09-18 --end 2026-09-17 --horizon 1 --stocks 227 --epochs 20 --seed 42
python verify_experiment.py runs/<experiment-id>
# Exact source data and splits, with validation-only parameter selection
python compare_models.py runs/<experiment-id>
python compare_models.py runs/statistical-<experiment-id> --verify
# Matched pilot comparison, retaining legacy first-25 selection
python compare_horizons.py --start 2023-09-18 --end 2026-09-17 --stocks 25 --epochs 20 --name shortline-my-run
```

Omit `--stocks` for an available-memory estimate. An explicit stock budget is still capped at the statistical requirement; use `--sampling legacy` only for first-N reproduction. Each run saves independent settings, pool snapshot, prices, model, logs and evidence under `runs/`. Exact replay requires the original local snapshot; source revisions and adjusted-price updates can change fresh downloads.

New checkpoints freeze all model defaults and portfolio settings; source hashes are captured at experiment start. Legacy object-based checkpoints require the original `summary.json` to avoid inheriting changed code defaults during replay.

Real deep runs used an RTX 3060 12GB and PyTorch 2.8.0+cu126. Deep training falls back to CPU; statistical fitting uses CPU. Install a driver-compatible build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Closing the page does not stop a worker; the stop button controls only jobs launched by the current session. The dashboard defaults to localhost and is intended for local single-user use.

</details>

<details>
<summary><strong>Evaluation protocol and cost assumptions</strong></summary>

Chronological 70% / 15% / 15% splits use a common 5-day embargo at both boundaries. Features end before the signal date and never backfill from the future. Deep checkpoints are selected by validation loss; ridge penalties by validation IC. Test metrics do not enter these automated choices, but inspected dates are no longer independent confirmation of new hypotheses.

Prediction and rebalancing horizons are 1 / 3 / 5 days. Default top-10 holdings and a 10% cap effectively impose equal weights when fully invested; do not attribute algorithm differences to the risk estimator alone. Weights drift with prices; buys and sells incur 0.1% costs plus 0.05% slippage, including entry. No final-day rebalance is created. Drawdown includes starting capital. IC uses horizon-specific labels; quantile returns use nonoverlapping periods. The reference is same-pool daily equal-weight gross return, **not CSI 300**.

</details>

<details>
<summary><strong>Source map and validation commands</strong></summary>

```text
app.py / ui/                Bilingual dashboard, versions, comparisons and analysis
research/pipeline.py        Deep-model experiment workflow
research/statistical.py     PCA, SVD ridge, EWMA and block intervals
research/sampling.py        Sampling frame, statistical count and capacity budget
research/analysis.py        Monthly returns, drawdown, costs and diagnostics
research/experiments.py     Isolated workers and persistent status
research/reporting.py       Evidence, source metadata, environment and hashes
data/ / training/ / model/  Downloads, causal labels, features and deep model
portfolio/ / evaluation/    Shared simulation, costs and evaluation
reports/ / docs/            Compact real evidence and guides
runs/                      Local prices, factors and checkpoints, not committed
tests/                     Time integrity, matrices, sampling, backtests and UI
```

Functional comments and docstrings are primarily English; file headers retain the bilingual author notice. Core, UI and optional analysis dependencies are separate. Windows and Linux CI use the same suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

</details>

<a id="documentation"></a>
## Documentation

| Start here | What you will find |
|---|---|
| [Demo guide](docs/DEMO.md) | Interface tour, CSV import and troubleshooting |
| [Algorithms](docs/ALGORITHMS.md) | Model definitions, formulas and mathematical assumptions |
| [Scientific sampling](docs/SAMPLING.md) | Sampling frame, capacity budget and precision gap |
| [Experiment analysis](docs/EXPERIMENT_ANALYSIS.md) | Real results, cost diagnostics and research limitations |

## Limitations

Current-universe and as-of-download adjustment biases remain; random sampling does not remove them. Simulations omit limit-up/down execution, suspension constraints, tax differences and real execution latency. Independent multi-window, multi-seed confirmation is not complete. This is an auditable research prototype, not a live-trading system or a return promise.

[All Rights Reserved](LICENSE) · [@ilovemiku520](https://github.com/ilovemiku520)

<a id="usage-notice"></a>
## Usage and author

<details>
<summary><strong>Complete usage and AI-use notice / 完整中英文声明</strong></summary>

<!-- BEGIN RIGHTS NOTICE -->
## 版权与使用限制 / Copyright and use restrictions

**保留所有权利。未经著作权人事先书面许可，不得使用、运行、复制、修改或分发本项目受保护的原创内容，包括个人、学习、研究、非商业和商业用途，以及依法需要许可的 AI 使用。Star 不构成授权。**

**All rights reserved. Prior written permission is required to use, run, copy, modify or distribute the project's protected original material, including personal, educational, research, non-commercial and commercial use, and AI use where permission is required by law. A GitHub Star does not grant permission.**

完整条款见 [LICENSE](LICENSE)。第三方内容仍适用其各自许可；此前已授予的许可、法定权利及 GitHub 平台条款项下权利不受影响。本文中的安装、运行及开发说明仅为技术说明，不构成使用授权。

See [LICENSE](LICENSE) for the full terms. Third-party licenses, previously granted permissions, statutory rights and rights under GitHub's Terms of Service remain unaffected. Setup, usage and development instructions are technical documentation, not permission to use the material.

书面授权 / Permission requests: [ilovemiku520@outlook.com](mailto:ilovemiku520@outlook.com)

关注初音未来谢谢喵，ilovemiku520  
Please follow Hatsune Miku, thank you, meow. ilovemiku520
<!-- END RIGHTS NOTICE -->

</details>

<p align="center">关注初音未来谢谢喵，ilovemiku520<br><sub>Please follow Hatsune Miku, thank you, meow. ilovemiku520</sub></p>
<p align="center"><a href="https://github.com/ilovemiku520/VCStockRank/issues">Report an issue</a> · <a href="https://github.com/ilovemiku520">@ilovemiku520</a></p>
