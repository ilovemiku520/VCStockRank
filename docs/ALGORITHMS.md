# 模型算法 / Model algorithms

[简体中文 README](../README.md) · [English README](../README.en.md)

## 实现与适用范围 / Implementation and scope

**深度模型 / Deep model**：VCformer-TPA 多任务横截面排序，保留原有时序卷积、注意力、排序及未来风险任务。输入为目标日之前 30 日的 51 维特征。VCformer 是项目模块名，当前注意力沿时间维运行，不声称复现其他论文的全部结构。

**统计模型 / Statistical model**：`research/statistical.py` 把同一历史窗口的末值、均值、标准差组合为 153 维输入，再做训练集标准化、PCA 和岭回归。它是另一条可审计的建模路线，不能仅凭名称或复杂程度宣称优于深度模型。Both methods use features strictly before the signal date. The statistical model summarizes the same history, so this is a full algorithm comparison rather than an isolated architecture ablation.

### 多元统计与矩阵计算 / Multivariate statistics and matrix computation

1. 标准化的均值与尺度只用训练集估计。Fit standardization on training rows only.
2. 对训练矩阵做完整 SVD，保留累计解释方差达到 95% 的主成分。该比例是特征方差覆盖率，**不是预测准确率**。Fit full-SVD PCA on training data; retain 95% of feature variance, not 95% prediction accuracy.
3. 对主成分矩阵 $Z=USV^T$，使用

   $$\hat\beta_\alpha=V\operatorname{diag}\left(\frac{s_j}{s_j^2+n\alpha}\right)U^T(y-\bar y).$$

   对应目标是平均平方误差加 $\alpha\|\beta\|_2^2$。直接使用 SVD，避免显式求逆及形成正规方程带来的条件数平方。The objective is mean squared error plus an L2 penalty. SVD avoids explicit inversion and handles collinearity.
4. 预先固定候选 $\alpha\in\{10^{-4},10^{-3},10^{-2},10^{-1},1\}$，只按验证集每日 Spearman IC 均值选择，平局用验证 MSE；不扩大网格追逐测试结果，也不把验证集重新并入训练。The candidate grid is fixed; test outcomes never enter parameter selection.

保存主成分数、解释方差比例、保留子空间条件数、有效自由度 $\sum_j s_j^2/(s_j^2+n\alpha)$ 以及所有验证候选结果。模型保存为不需要 pickle 的 NumPy 数组，支持独立重放。

Reference: [PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

### 随机过程与条件风险 / Stochastic risk estimates

令 $r_t$ 为原始复权收盘日收益，$\lambda=2^{-1/10}$。EWMA 递推为 $v_t=\lambda v_{t-1}+(1-\lambda)r_t^2$，信号日 $t$ 只使用 $\sqrt{v_{t-1}}$，且至少有 30 个历史收益观测。This estimates a time-varying second moment; interpreting it as variance assumes a negligible conditional mean. It is not a fitted Gaussian price process or a profit probability.

默认组合为前 10 只、单股上限 10%；满仓时权重实际上受到等权约束。因此风险输出虽进入配置相同的优化器，**不能把本次收益改善单独归因于 EWMA**。研究风险配置的价值，需要另设并预先固定持仓数量、权重约束的消融实验。

### 时间依赖与区间 / Dependence and uncertainty

统计模型报告用 10 日循环移动区块、2,000 次重抽样、种子 42，计算平均每日 IC 和“策略日净收益减参考日毛收益”的 95% 百分位区间。区块保留局部时间依赖，仍依赖近似平稳假设；109 个收益日很短，区块长度敏感性尚未验证。区间不是累计收益区间、未来盈利概率，也未校正多模型比较。The intervals are exploratory; overlapping labels and repeated test inspection prevent an independent final-validation claim.

Reference: [Block bootstrap for time series — Forecasting: Principles and Practice](https://otexts.com/fpp3/bootstrap.html).

## 运行 / Run

先完成真实实验，然后使用其确切数据和切分：

```bash
python compare_models.py runs/<completed-v4-experiment>
python compare_models.py runs/statistical-<completed-v4-experiment> --verify
```

行情、特征和检查点必须留在本地；GitHub 的轻量报告不能单独重放模型。The baseline reuses downloaded data, preserves split dates and transaction assumptions, and selects hyperparameters using validation only. Its input manifest hashes the source factors and prices. Statistical fitting runs on CPU; deep training automatically uses CUDA when available.

## 已实现的换手控制 / Implemented turnover control

`--turnover-control` 在验证区间分别回测排名缓冲 0 / 5 / 10 / 20，按验证净收益选优，平局依次选择换手较低和缓冲较小者。持有股票仍位于前 `TOP_K + buffer` 名时优先保留，再按排名填补空位；只使用当日可用信号及此前持仓，不查看未来收益。预测分数和 IC 不变，改变的是持仓选择。

The buffered variant selects its holding rule on validation net return only. It checks holdings daily but reduces constituent turnover; it does not eliminate daily weight rebalancing, transaction costs, or drawdown risk. Its development followed inspection of earlier test costs, so the reported test comparison remains exploratory.

```bash
python compare_models.py runs/<completed-v4-experiment> --turnover-control
```

工作台“开始研究”的已完成实验也提供此选项。The completed-experiment panel offers the same option.

## 下一步优先级 / Next priorities

- 在新的时间区间做 walk-forward 与多种子重复，检验方向是否稳定。Use fresh-date walk-forward evaluation and multiple seeds before selecting a production method.
- 在验证集比较成本敏感目标和持仓缓冲。本次线性模型的换手更高，毛收益改善可能被费用抵消。Evaluate turnover buffers and cost-aware selection on validation dates.
- 分别检验 PCA、正则化、历史聚合、风险输出的贡献，避免同时更改多个模块后错误归因。Ablate PCA, shrinkage, summaries and risk separately.
- 接入历史成分股、停牌和涨跌停可成交条件，再评估可执行性。Add point-in-time universes and execution constraints.
