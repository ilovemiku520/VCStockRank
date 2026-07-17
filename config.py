# config.py
#本项目采用cpu训练，请根据实际硬件情况进行调整
# 以下全局变量参数仅供参考，请根据实际需求和环境资源调整
import torch
from datetime import datetime, timedelta
import os
import multiprocessing
# ========== 自动调用全部 CPU 核心， 选择性注释==========
cpu_count = multiprocessing.cpu_count()
# 如果核心数太多（>64），建议限制在 64 以内避免系统卡顿
effective_cores = min(cpu_count, 64)

os.environ["OMP_NUM_THREADS"] = str(effective_cores)
os.environ["MKL_NUM_THREADS"] = str(effective_cores)

torch.set_num_threads(effective_cores)

print(f"🔧 使用 {effective_cores} 个 CPU 核心进行训练")
# ============================================
# ================= 数据配置 =================
DATA_START = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
DATA_END = datetime.now().strftime('%Y-%m-%d')
MIN_TRADING_DAYS = 180
MAX_STOCKS = 25




# ================= 模型配置 =================
class ModelConfig:
    # 数据配置
    MAX_STOCKS = MAX_STOCKS
    MIN_TRADING_DAYS = MIN_TRADING_DAYS
    DATA_START = DATA_START
    DATA_END = DATA_END

    # 序列长度（缩短以加速计算）
    SEQ_LEN = 30                     # 原 60，调试用 30

    # 模型维度（减小容量，加快训练）
    INPUT_DIM = 50                   # 因子数不变，自动调整
    HIDDEN_DIM = 64                  # 原 128，减半
    NUM_HEADS = 4                    # 原 8，减少头数
    NUM_LAYERS = 1                   # 原 2，减少层数
    DROPOUT = 0.1

    # TPA配置（保持原样）
    CNN_FILTERS = [3, 6, 12, 24]
    CNN_CHANNELS = 32

    # 训练配置（调试用轻量参数）
    BATCH_SIZE = 256                 # 增大批次，提高吞吐
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20                      # 原 100，调试仅跑 20 轮
    PATIENCE = 5                     # 原 20，更早触发早停

    # 多任务权重（保持原样）
    LAMBDA_VOL = 0.3
    LAMBDA_DECOMP = 0.05

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 滚动窗口配置（缩短验证和测试窗口，快速出结果）
    TRAIN_WINDOW = 720
    VAL_WINDOW = 10                  # 原 30，调试用 10
    TEST_WINDOW = 10                 # 原 30，调试用 10


# ================= 组合配置（保持不变）=================
class PortfolioConfig:
    TOP_K = 30
    REBALANCE_FREQ = 5
    MAX_WEIGHT = 0.1
    RISK_AVERSION = 0.5
    TRANSACTION_COST = 0.001
    SLIPPAGE = 0.0005


# ================= 回测配置（保持不变）=================
class BacktestConfig:
    START_DATE = DATA_START
    END_DATE = DATA_END
    BENCHMARK = '沪深300'
    RF_RATE = 0.02


# ================= 打印配置信息 =================
print("=" * 60)
print("配置信息 (调试模式)")
print("=" * 60)
print(f"数据时间范围: {DATA_START} 至 {DATA_END}")
print(f"回测时间范围: {BacktestConfig.START_DATE} 至 {BacktestConfig.END_DATE}")
print(f"设备: {ModelConfig.DEVICE}")
print(f"输入特征维度: {ModelConfig.INPUT_DIM}")
print(f"最大股票数: {ModelConfig.MAX_STOCKS}")
print(f"序列长度: {ModelConfig.SEQ_LEN}")
print(f"隐藏层维度: {ModelConfig.HIDDEN_DIM}")
print(f"批次大小: {ModelConfig.BATCH_SIZE}")
print(f"训练轮次: {ModelConfig.EPOCHS}")
print(f"早停耐心: {ModelConfig.PATIENCE}")
print("宏观数据: 已禁用")
print("=" * 60)


