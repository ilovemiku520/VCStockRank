# 真实实验分析 / Real experiment analysis

[中文 README](../README.md) · [English README](../README.en.md) · [Algorithms](ALGORITHMS.md)

## 扩大随机样本：225 只的隔日对照 / Expanded random-sample comparison

实际抽取 227 只、取得足够历史 225 只、160,526 行行情。总体 4,992，统计目标 357，预算 227，规划误差约 ±6.36 个百分点，未达到 ±5 个百分点。两只缺失/不足股票 `sz.001369`、`sh.603407` 没有被替换；有效训练样本不能继承完整简单随机样本的精度保证。

The larger run uses the same data dates, split proportions and 1-day protocol, but a different random universe. Comparing it with the first-25 pilot does **not** isolate stock-count effects. The deep run completed all 20 epochs on RTX 3060, choosing epoch 16, with 103,558 training, 23,233 validation and 24,525 labeled test samples. All three larger-run models passed saved-parameter replay.

| 方法 / Method | Net return | Drawdown magnitude | Gross, same holdings | Cost drag (pp) | Mean two-way turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Deep](../reports/20260919-002021-54513a/summary.json) | −14.57% | 29.33% | −3.25% | 11.32 | 76.05% |
| [PCA-ridge](../reports/statistical-scientific-20260919-1d/summary.json) | −0.27% | 34.13% | +12.33% | 12.60 | 72.73% |
| [PCA-ridge + buffer](../reports/statistical-buffered-scientific-20260919-1d/summary.json) | +6.53% | 29.80% | +12.05% | 5.51 | 30.85% |

同池等权毛收益为 −4.67%。缓冲保留仍排在前 30 名以内的已有持仓，再填补到前 10 个持仓；排名缓冲 20 由验证净收益从 0/5/10/20 选出。验证候选净收益分别为 −3.70%、−1.56%、+1.61%、+4.93%，全部保存在 `model_diagnostics.json`。没有用测试净收益选缓冲参数。

预测分数与 IC 不变，变化来自持仓规则；每日仍检查并调整权重，因此调仓记录数仍为 109。费用拖累从 12.60 降为 5.51 个百分点，同时毛收益也略有变化，不能把全部净收益差理解成固定持仓下的单一费用效应。

The buffered model's test net return is +6.53%, but drawdown is 29.80%. Its exploratory block interval for mean daily net strategy minus gross reference is **−0.1447% to +0.4635%**, including zero. The hypothesis was developed after inspecting earlier test costs; validation-only parameter selection does not turn this reused test interval into independent confirmation. New dates and predeclared robustness checks remain necessary.

## 相同试点股票池的 1 / 3 / 5 日对照 / Matched pilot comparison

数据：BaoStock 前复权日线，2023-09-18—2026-09-17；25 只旧流程前排股票，727 个交易日、18,175 条记录，种子 42。统一 70/15/15 时间切分与 5 日边界禁入；109 日收益区间为 2026-04-14—2026-09-17。下载快照、成本和组合约束相同。

The pilot retains the legacy first-25 universe, not a scientific random sample. Deep inputs have 51 factors over 30 days; the statistical model uses 153 historical summaries. Both exclude the signal date from features. Changing horizon also changes forward labels, so cross-horizon IC values measure different tasks.

| 算法 / Algorithm | 周期 / Days | 净收益 / Net return | 回撤幅度 / Drawdown | Mean IC | 成本拖累 / Cost drag (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [VCformer-TPA](../reports/shortline-20260919-1d/summary.json) | 1 | -5.38% | 16.48% | 0.0243 | 1.23 |
| [VCformer-TPA](../reports/shortline-20260919-3d/summary.json) | 3 | -1.88% | 21.44% | 0.0620 | 2.23 |
| [VCformer-TPA](../reports/shortline-20260919-5d/summary.json) | 5 | -13.31% | 25.15% | 0.0402 | 1.94 |
| [PCA-ridge + EWMA](../reports/statistical-shortline-20260919-1d/summary.json) | 1 | 0.15% | 17.23% | 0.0256 | 4.42 |
| [PCA-ridge + EWMA](../reports/statistical-shortline-20260919-3d/summary.json) | 3 | 4.01% | 15.84% | 0.0454 | 2.66 |
| [PCA-ridge + EWMA](../reports/statistical-shortline-20260919-5d/summary.json) | 5 | -0.52% | 14.61% | 0.0487 | 1.67 |

同股票池每日等权参考毛收益为 **−7.30%**；未扣费且不是沪深 300。费用拖累通过交易账本还原相同持仓的毛收益，再计算期末净值差，不是简单累计费率。The reference is a gross daily-rebalanced same-pool portfolio. Return differences are not risk-adjusted alpha.

## 从结果能看出什么 / What the results support

1. **缩短周期不等于提高收益。** 原模型并非长线：5 日版已经是短周期研究。此次 3 日深度模型比 1 / 5 日少亏，但不能依此宣布 3 日是最优持有期。Shortening the horizon did not monotonically improve performance.
2. **统计基线值得保留，尚不能宣布胜出。** 这次 PCA 岭回归三个周期的净收益均高于对应深度模型，但仍有较大回撤，且统计模型 1 日费用拖累约 4.42 个百分点。Lower complexity improved this particular comparison, but turnover and instability remain.
3. **IC 为正不保证赚钱。** 深度 5 日模型 IC 均值约 0.0402，却净亏 13.31%；5 月策略亏约 12.10%，参考亏约 4.28%。排序相关性、收益幅度、选股截断、持仓路径与成本是不同环节。A positive mean ranking correlation does not ensure a profitable long-only portfolio.
4. **风险输出未被单独验证。** 默认前 10 只、单股 10% 上限几乎限定为等权；新模型的变化包括特征汇总、PCA、岭惩罚和风险估计，不能把收益差归因于其中某一个模块。Separate ablations are required.

## 统计不确定性 / Statistical uncertainty

统计模型采用 10 日循环区块、2,000 次重抽样、固定种子，估计“策略日净收益减参考日毛收益”的日均值区间：

| 周期 / Horizon | 日均差的 95% 探索性区间 / Exploratory interval for mean daily difference |
| --- | --- |
| 1 日 / day | −0.0448% 至 / to +0.1880% |
| 3 日 / days | −0.0436% 至 / to +0.2579% |
| 5 日 / days | −0.0664% 至 / to +0.2028% |

全部包含零。区间依赖近似平稳假设，未校正多次比较，区块长度敏感性尚未验证；不能解释为累计收益区间或未来盈利概率。All intervals include zero. They are descriptive uncertainty estimates, not proof of a persistent edge. Details and machine-readable values are saved in each statistical run’s `analysis.json`.

## 训练与验证 / Fitting and verification

深度模型均使用 RTX 3060 CUDA：1 / 3 日训练达到 20 轮上限；5 日在第 12 轮早停，最佳第 7 轮。各实验按自己的验证损失选模型；没有按本表收益选检查点。统计模型的标准化、PCA 和回归系数只在训练段拟合，岭惩罚仅按验证 IC 选择。

每个模型均重新加载保存参数并重放预测与每日收益；公开 `replay_validation.json` 记录实际误差。重放验证保证本地保存实验的可复核性，不代表不同 GPU 重新训练会逐位一致。Fresh market downloads may also change because of corporate-action adjustments and data revisions.

旧 v3 的 5 日净收益 −1.88% 来自不同风险标签口径；它与 v4 的 −13.31% 不构成只有算法一项改变的对照，也不能删去较差的新结果。Legacy evidence remains available for audit.

## 改进建议 / Research priorities

- **优先做新时间窗口验证**：多市场阶段、多个种子、walk-forward；新的最终区间只使用一次。Do not tune again on the already-inspected 109-day period.
- **控制成本**：在验证段比较持仓缓冲、换手惩罚与成本敏感目标，同时报告回撤。The statistical 1-day strategy’s larger cost drag makes this a concrete priority.
- **做分项消融**：PCA vs. no PCA、不同固定正则强度、历史汇总 vs. 序列网络、风险任务开关；预先固定规则。Separate component effects rather than adding complexity indiscriminately.
- **改进抽样与数据**：新实验改为无放回随机抽样，保留失败下载记录，逐步接入历史成分股。A larger sample does not itself remove survivorship or execution bias.
- **补齐可成交性**：停牌、涨跌停、费用差异与实际成交时点；当前结果属于研究模拟。Do not treat close-to-close research results as executable returns.
