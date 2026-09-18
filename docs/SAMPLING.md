# 科学抽样 / Scientific sampling

从去重后的沪深 A 股股票池中排除 ST 和退市名称，以固定种子无放回简单随机抽样。旧实验的“前 25 只”原样保留，不能事后称为随机样本。The frame is the supported Shanghai/Shenzhen A-share pool; legacy first-25 experiments remain nonrandom pilot studies.

默认置信水平 95%、比例估计误差 $e=0.05$、未知比例按 $p=0.5$ 保守规划，$z=1.95996$。有限总体大小为 $N$ 时：

$$n_{required}=\left\lceil\frac{Nz^2p(1-p)}{e^2(N-1)+z^2p(1-p)}\right\rceil.$$

$$n_{selected}=\min(n_{required},n_{capacity},N).$$

容量不足时报告近似规划误差：

$$e_{planned}=z\sqrt{\frac{p(1-p)}{n_{selected}}\frac{N-n_{selected}}{N-1}}.$$

当前合格池 **4,992** 只，目标需要 **357** 只；本次按容量预算 **227** 只抽样，规划误差约 **±6.36 个百分点**，**不满足 ±5 个百分点目标**。This normal-approximation calculation concerns a binary proportion of a fixed stock pool. It does not establish model-return confidence, forecast accuracy, or statistical power for a trading strategy.

容量预算按可用 RAM 的 30%、历史窗口物化开销，以及 GPU 保留 1 GiB 后每股票 20 MiB 的保守规则估计，可手动调整。这是内存规划启发式，未经极限压力测试，不是硬件支持股票数量的精确最大值。Capacity is an adjustable heuristic, not a measured maximum, runtime promise or reason to claim the statistical target was met. The value can change with machine load.

每个实验保存 `stock_pool.csv` 和 `sampling_plan.json`：总体数、目标数、预算、抽样数、规划误差、是否达标与随机种子。`data/source.json` 另外记录下载失败及成功行情。行情不足、缺失或训练窗口要求可能进一步缩小有效样本；不得把抽取数量直接当成完整有效训练样本，更不能在失败后随意补入方便取得的股票。Missing histories create nonresponse and possible selection bias; the planned precision is not a guarantee for the final usable sample.

```bash
# Auto-estimated capacity with a saved random sample
python run_experiment.py --start 2023-09-18 --end 2026-09-17 --horizon 1
# Explicit capacity budget; still capped at the statistical requirement
python run_experiment.py --start 2023-09-18 --end 2026-09-17 --horizon 1 --stocks 227 --sampling random
# Legacy first-N selection is available only for reproducing pilot comparisons
python run_experiment.py --start 2023-09-18 --end 2026-09-17 --stocks 25 --sampling legacy
```

Sources: [Penn State — sample-size planning](https://online.stat.psu.edu/stat506/Lesson02), [finite-population sampling](https://online.stat.psu.edu/stat506/Lesson01), [NIST — proportion intervals and approximation limitations](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

本次扩展实跑已完成：抽取 227 只，225 只有足够历史，160,526 条行情。两只不足样本未被替换，完整记录见 [扩展实验](../reports/20260919-002021-54513a/summary.json)。The expanded run is complete; missing histories remain disclosed rather than silently replaced.
