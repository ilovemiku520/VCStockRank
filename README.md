# VCStockRank

**简体中文** | [English](README.en.md)

面向 A 股横截面排序研究的 Python 项目：真实行情下载、因果特征处理、GPU 多任务训练、样本外回测，以及可切换中英文的研究工作台。

> 这是研究原型，不是实盘交易系统。以下结果来自真实历史行情，但不代表可实现的投资收益，也不构成投资建议。

## 已完成的真实实验

2026-09-18 使用 **NVIDIA GeForce RTX 3060 12GB** 完成行情下载、训练、最佳模型恢复、测试集预测、回测和评估，并通过保存模型的独立重放检查。

| 项目 | 本次结果 |
| --- | --- |
| 数据源 / 复权 | BaoStock 日线 / 前复权（adjustflag=2） |
| 请求并取得的数据区间 | 2023-09-18 至 2026-09-17 |
| 股票 / 交易日 / 日线记录 | 25 / 727 / 18,175 |
| 模型特征 / 历史窗口 | 51 / 30 个交易日，截止目标日前一日 |
| 随机种子 | 42 |
| 训练 / 验证 / 测试样本 | 11,825 / 2,600 / 2,625 |
| 训练 | 上限 20 轮；第 19 轮验证早停；最佳模型第 14 轮 |
| 训练框架 | PyTorch 2.8.0+cu126 |
| 测试收益区间 / 记录数 | 2026-04-14 至 2026-09-17 / 109 日 |
| 累计收益（计入成本和滑点） | **-1.88%** |
| 年化收益 / 夏普 | -4.29% / -0.161 |
| 最大回撤 | **18.19%** |
| 平均 Spearman IC / ICIR | 0.0477 / 0.1727 |

首次行情下载约 110 秒；使用该缓存完成特征构建、训练、回测与报告约 227 秒。性能取决于本机负载，不能作为其他设备的耗时保证。预测共 2,750 行；最后 5 日没有完整未来标签，因此 IC 与测试标签样本数少于回测信号覆盖量。

[实验摘要与哈希](reports/real-20260918/summary.json) · [逐日收益](reports/real-20260918/backtest_returns.csv) · [训练历史](reports/real-20260918/training_history.csv) · [模型重放检查](reports/real-20260918/replay_validation.json)

仓库保存轻量结果、股票池快照及来源说明；完整行情、特征和模型权重留在本地 `runs/`。重放检查验证了本次保存模型对 2,750 行预测和 109 日收益的复现，最大绝对差小于 1e-12；这不等于在其他硬件上重新训练会逐位一致。

## 启动工作台

建议 Python 3.11 或 3.12。进入项目目录后创建独立环境：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-ui.txt
python -m streamlit run app.py
```

打开 http://127.0.0.1:8501 。左侧 **语言 / Language** 可切换简体中文与 English；README 顶部链接切换文档语言。界面控件和说明随语言切换，股票名称、原始运行日志及实验字段保留原文。

- **研究概览**：默认展示已有真实实验，支持归档报告、CSV 导入、日期筛选、策略与等权参考净值、回撤、训练曲线、每日 IC 和文件下载。
- **开始研究**：设置日期、股票数量、训练轮数与种子，查看当前阶段、训练进度和历史任务。
- **股票池**：按代码或名称搜索和导出。训练在排除 ST 等不合格标的后按文件顺序选择前 N 只，页面搜索不改变股票池。

只看界面和已保存的报告无需安装 PyTorch。上传 CSV 必须有 `date,return` 两列，一日一条，收益使用已扣费的小数值（0.01 = 1%）。重复日期、空数据或非有限收益会被拒绝。演示数据有明确标注。

## 真实训练与 GPU

```bash
python -m pip install -r requirements.txt -r requirements-ui.txt
python -m streamlit run app.py
```

PyTorch 能识别 CUDA 时自动使用 GPU，否则使用 CPU。NVIDIA 用户应先安装与驱动兼容的 CUDA 版 PyTorch；本次实跑使用 cu126，示例：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

其他硬件请参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。不需要为了本项目额外安装 torchvision、AKShare 或 XGBoost。解释性分析和回归绘图的可选依赖位于 `requirements-extras.txt`。

同一流程也可从命令行运行：

```bash
python run_experiment.py --start 2023-09-18 --end 2026-09-17 --stocks 25 --epochs 20 --seed 42
python verify_experiment.py runs/<实验编号>
```

每次生成独立实验目录，重新训练，不自动复用旧模型。原始行情按股票、日期范围和复权方式缓存于 `cache/market/`，可复用同一下载快照。`main.py` 和 `backtest_main.py` 保留兼容入口；独立回测时在实验目录执行 `python ../../backtest_main.py`。

关闭浏览器不会停止后台任务。停止按钮仅控制当前会话启动的进程；完成结果会持久保留。本工作台仅面向本机单用户，默认监听 127.0.0.1。

## 评估协议 v3 与修复

1. 按交易日切分 70% 训练、15% 验证、15% 测试，两处边界均保留 5 个交易日的标签禁入期。
2. 样本仅读取目标日前的历史窗口；缺失值不从未来回填。特征去极值与标准化按当日横截面处理，单序列场景使用扩展历史统计。
3. 批处理保留同日横截面；排序损失只比较同一天的股票，并排除相同标签的配对。日期元数据改为可批处理的字符串。
4. 只根据验证损失早停、选择模型；测试区间不参与模型选择。检查点保存特征顺序与模型配置，训练与独立回测共用批量推理函数。
5. 默认持有前 10 只、5 个交易日调仓、单股上限 10%。持仓权重随价格漂移；权重上限无法满仓时余款留现金；缺信号日继续按交易日计收益。
6. 买卖均按成交名义金额计入 0.1% 成本和 0.05% 滑点，初始建仓也收费。信号采用收盘到下一收盘的研究近似；最后一个交易日不产生无后续收益的调仓。
7. 最大回撤包含初始本金。分层未来 5 日收益以非重叠窗口抽样；看板等权参考使用同股票池每日等权收益，未扣费，**不是沪深 300**。

v3 修改了排序和回测口径，v1/v2 检查点需要重新训练。旧版全样本回测的收益宣传已撤回，不能与本次结果直接比较。

## 代码结构

```text
app.py                     轻量工作台入口
ui/                        双语文案、页面、图表和任务视图
research/experiments.py     实验目录、进程与状态持久化
research/pipeline.py        数据、训练、预测、回测流程
research/inference.py       共享特征选择与批量推理
research/runner.py          分阶段任务执行与异常状态
research/reporting.py       结果、来源、环境和文件哈希
data/                      BaoStock 下载、缓存、清洗与特征
model/                     分解、注意力、时序卷积与多任务输出
training/                  数据集、按日期采样、损失和训练器
portfolio/                 权重优化、持仓模拟与指标
evaluation/                IC、分层收益和可选解释分析
reports/                   可提交 GitHub 的轻量实跑记录
runs/                      本地行情、模型、日志和实验结果
tests/                     时间完整性、损失、回测、界面与 GPU 检查
```

Python 注释和 docstring 使用英文；删去逐行复述代码的冗余注释。核心、界面和可选分析依赖分开安装。配置导入不再全局修改 CPU 线程或打印配置，运行时显式初始化线程数与随机种子。

## 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试覆盖日期批处理、同日排序、预测窗口一致性、持仓漂移、现金上限、首日交易成本、CSV 校验、中英文切换及任务失败状态。GPU 前向/反向测试在无 CUDA 环境自动跳过。CI 在 Windows 和 Linux 上运行。

## 已知限制

- 固定的当前股票池可能存在幸存者偏差；本次 25 只不能代表全市场。
- 前复权使用下载时的调整口径，未提供逐历史时点的成分股和公司行动快照。
- 收盘价回测未模拟涨跌停、停牌成交约束、税费差异或真实执行延迟。
- 仅完成单次时间切分与单个随机种子实验，尚无完整 walk-forward 稳健性检验。
- VCFormer 为沿用的模块名称；当前注意力实现是序列维标准多头注意力，不能据此声称已验证变量维专用结构的优势。
- 本次组合收益为负，应将其作为可检查的研究基线，而非收益承诺。

## 许可证

[MIT License](LICENSE) · [@ilovemiku520](https://github.com/ilovemiku520)
