"""Run one isolated dashboard experiment in a separate Python process."""
import json
import os
from pathlib import Path
import sys
import traceback


def run(directory):
    directory = Path(directory).resolve()
    os.chdir(directory)
    try:
        settings = json.loads((directory / 'settings.json').read_text(encoding='utf-8'))
        from config import ModelConfig
        from main import MultiModalStrategy

        config = ModelConfig()
        for key, value in settings.items():
            setattr(config, key, value)
        strategy = MultiModalStrategy(config)
        if strategy.load_data() is None:
            raise RuntimeError('数据下载或因子构建失败，请检查网络、日期范围和股票池。')
        if strategy.train_model() is None:
            raise RuntimeError('训练失败，可能没有足够的有效交易日。')
        if not strategy.generate_predictions():
            raise RuntimeError('未生成有效的测试集预测。')
        result = strategy.run_backtest()
        if result is None or result.returns.empty:
            raise RuntimeError('回测没有生成收益，请检查价格和预测日期。')
        result.returns.rename_axis('date').rename('return').to_csv('backtest_returns.csv')
        strategy.evaluate_predictions()
        status = {'state': 'complete', 'message': '训练、回测与评估已完成。'}
    except Exception as error:
        traceback.print_exc()
        status = {'state': 'failed', 'message': str(error)}
    (directory / 'status.json').write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')
    return 0 if status['state'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(run(sys.argv[1]))
