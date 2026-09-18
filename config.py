import torch
from datetime import datetime, timedelta

DATA_START = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
DATA_END = datetime.now().strftime('%Y-%m-%d')
MIN_TRADING_DAYS = 180
MAX_STOCKS = 25

class ModelConfig:
    PROTOCOL_VERSION = 4
    SEED = 42
    CPU_THREADS = 4
    REUSE_CHECKPOINT = False
    CACHE_DIR = "cache/market"

    MAX_STOCKS = MAX_STOCKS
    MIN_TRADING_DAYS = MIN_TRADING_DAYS
    DATA_START = DATA_START
    DATA_END = DATA_END

    SEQ_LEN = 30

    INPUT_DIM = 50
    HIDDEN_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 1
    DROPOUT = 0.1

    CNN_FILTERS = [3, 6, 12, 24]
    CNN_CHANNELS = 32

    BATCH_SIZE = 256
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    PATIENCE = 5
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    LABEL_HORIZON = 5

    LAMBDA_VOL = 0.3
    LAMBDA_DECOMP = 0.05

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRAIN_WINDOW = 720
    VAL_WINDOW = 10
    TEST_WINDOW = 10

class PortfolioConfig:
    TOP_K = 10
    REBALANCE_FREQ = 5
    MAX_WEIGHT = 0.1
    RISK_AVERSION = 0.5
    TRANSACTION_COST = 0.001
    SLIPPAGE = 0.0005

class BacktestConfig:
    START_DATE = DATA_START
    END_DATE = DATA_END
    BENCHMARK = '沪深300'
    RF_RATE = 0.02
