# 演示指南 / Demo guide

[中文 README](../README.md) · [English README](../README.en.md)

![Real-data pilot comparison](../assets/real-comparison.svg)

真实收益图，与工作台的合成演示分开。Real return curves, separate from synthetic dashboard demos.

## 无需训练，先体验

```bash
python -m pip install -r requirements-ui.txt
python -m streamlit run app.py
```

打开 http://127.0.0.1:8501 。无需 PyTorch、GPU、行情账户或联网下载行情即可查看已发布报告与合成演示。

1. 在侧栏选择语言。
2. 在“研究概览”选择 **隔日版 · 1 日** 或 **波段版 · 3 / 5 日**。波段版可再选 3 日或 5 日。
3. 选择“真实实验”查看仓库附带的训练报告；选择“演示数据”体验固定种子的合成曲线。两者有不同提示，不混用。
4. 调整日期范围，切换净值、回撤、每日明细。下载 CSV 后，再切换“上传 CSV”验证导入。
5. 对真实实验打开“结果分析与建议”：查看完整区间的月度表现、等权参考、费用影响、IC 和改进优先级。
6. 进入“版本对照”，比较同一批次 1/3/5 日结果。尚未完成的版本不会伪造结果。
7. 安装完整依赖后，可在“开始研究”选择周期并启动自己的实验。完成后点击“用本次数据拟合 PCA 岭回归对照”，自动拟合并重放验证。

**分析范围**：顶部日期筛选影响收益指标、净值、回撤和每日明细。训练曲线、IC、分析建议使用完整实验区间，并在页面中明确标注。

**隔日版的含义**：输入为目标日前的历史日线，按目标日收盘成交、下一交易日收盘计收益的近似模拟；不是只赚隔夜跳空，也不是日内高频交易。两个版本没有实盘下单功能。

## Explore without training

Run the commands above, then open http://127.0.0.1:8501 . Saved reports and synthetic demos require no PyTorch, GPU, market account or fresh market download.

1. Switch the sidebar language to English.
2. Choose **Overnight · 1 day** or **Swing · 3 / 5 days** in Overview. Swing provides a second horizon selector.
3. Select **Real experiments** for published training evidence, or **Demo data** for repeatable synthetic curves. The interface labels them separately.
4. Filter dates, explore Equity/Drawdown/Daily returns, download a CSV and reimport it with Upload CSV.
5. For real experiments, open **Analysis & next steps** for monthly comparisons, cost drag, drawdown, IC and research priorities.
6. Open **Compare versions** for the matched 1/3/5-day batch. Incomplete runs never receive fabricated results.
7. Install the full requirements to launch your own experiment. Once complete, choose **Fit a PCA-ridge comparison on these data** for automatic fitting and replay.

**Scope**: the date filter affects headline performance, equity, drawdown and daily details. Training, IC and diagnostics use the full saved experiment and are labeled accordingly.

**Overnight is a product label**: the simulator uses daily close-to-close holding periods with historical inputs ending before the signal date. It is neither a pure overnight-gap strategy nor an intraday system. Neither version places live orders.

## 排错 / Troubleshooting

| 现象 / Symptom | 处理 / Action |
| --- | --- |
| 没有真实实验 / No real experiments | 核对选择的周期；仓库 reports/ 需完整。Check the selected horizon and retain the reports/ directory. |
| 没有 CUDA / CUDA unavailable | 演示不需要 GPU；训练可用 CPU，或安装驱动兼容的 CUDA PyTorch。The demo needs no GPU; train on CPU or install a compatible CUDA PyTorch build. |
| 上传 CSV 失败 / CSV rejected | 必须有 date,return；收益为小数，日期唯一。Require date,return, decimal returns and unique dates. |
| 端口占用 / Port in use | 使用 --server.port 8502 并打开对应地址。Use --server.port 8502 and open that port. |
| 页面显示旧内容 / Stale page | 保存文件后刷新；更新依赖或模块后重启 Streamlit。Refresh after edits; restart Streamlit after dependency/module changes. |

原始日志和股票名称保留原文，界面与说明可中英文切换。Raw logs and stock names retain their original language.

图形可由 `python plot_results.py` 从提交的收益文件重新生成，需要 Matplotlib。The chart can be regenerated from committed returns with `python plot_results.py` and Matplotlib.
