# evaluation/shap_analysis.py
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """
    SHAP值分析（使用代理模型）
    用于解释深度学习模型的预测
    """

    def __init__(self, model, feature_names=None, n_samples=1000):
        """
        Parameters:
        -----------
        model : trained model (deep learning model)
        feature_names : list, 特征名称
        n_samples : int, 用于SHAP分析的样本数
        """
        self.model = model
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.shap_values = None
        self.shap_results = {}

    def analyze(self, X, y=None, feature_names=None):
        """

        Parameters:
        -----------
        X : DataFrame or numpy array, 特征数据
        y : array-like, 目标值（可选）
        feature_names : list, 特征名称

        Returns:
        --------
        计算SHAP值
        shap_values : SHAP values
        """
        if feature_names is not None:
            self.feature_names = feature_names

        # 转换为numpy
        if isinstance(X, pd.DataFrame):
            X_np = X.values
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
        else:
            X_np = np.asarray(X)

        # 采样
        if len(X_np) > self.n_samples:
            indices = np.random.choice(len(X_np), self.n_samples, replace=False)
            X_sample = X_np[indices]
        else:
            X_sample = X_np

        # 使用代理模型（XGBoost）来近似深度学习模型
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            import shap

            # 生成预测（作为目标）
            # 这里需要模型的前向传播来得到预测得分
            # 假设模型有predict方法
            if hasattr(self.model, 'predict'):
                # 对于PyTorch模型，需要转换
                if hasattr(self.model, 'eval'):
                    import torch
                    self.model.eval()
                    with torch.no_grad():
                        if isinstance(X_sample, np.ndarray):
                            X_tensor = torch.FloatTensor(X_sample)
                        else:
                            X_tensor = X_sample
                        # 根据模型输出调整
                        preds = self.model(X_tensor)
                        if isinstance(preds, dict):
                            y_pred = preds['rank_score'].cpu().numpy().flatten()
                        else:
                            y_pred = preds.cpu().numpy().flatten()
                else:
                    y_pred = self.model.predict(X_sample)
            else:
                # 如果没有predict方法，使用已有的y
                if y is not None:
                    if len(y) >= len(X_sample):
                        y_pred = y[:len(X_sample)]
                    else:
                        y_pred = np.random.randn(len(X_sample))
                else:
                    # 生成随机目标（仅用于演示）
                    print("Warning: No prediction method found, using random targets")
                    y_pred = np.random.randn(len(X_sample))

            # 训练XGBoost代理模型
            X_train, X_test, y_train, y_test = train_test_split(
                X_sample, y_pred, test_size=0.2, random_state=42
            )

            proxy_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            proxy_model.fit(X_train, y_train)

            # 计算SHAP值
            explainer = shap.TreeExplainer(proxy_model)
            shap_values = explainer.shap_values(X_test)

            # 存储
            self.shap_values = shap_values
            self.proxy_model = proxy_model
            self.X_test = X_test
            self.y_test = y_test

            # 汇总结果
            self.shap_results = {
                'mean_abs_shap': np.abs(shap_values).mean(axis=0),
                'std_shap': shap_values.std(axis=0),
                'feature_names': self.feature_names,
                'proxy_r2': proxy_model.score(X_test, y_test)
            }

            print(f"Proxy model R²: {self.shap_results['proxy_r2']:.4f}")

            return self.shap_results

        except ImportError:
            print("XGBoost or SHAP not installed. Install with: pip install xgboost shap")
            return {}

    def plot_global_importance(self, save_path=None):
        """
        绘制全局特征重要性条形图
        """
        if not self.shap_results:
            print("No SHAP results available. Run analyze() first.")
            return

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 获取特征重要性
            importance = self.shap_results['mean_abs_shap']
            feature_names = self.shap_results['feature_names']

            # 排序
            sorted_idx = np.argsort(importance)[::-1]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(sorted_idx)), importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Global Feature Importance (SHAP)')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            print("matplotlib not installed")

    def plot_summary(self, save_path=None):
        """
        绘制SHAP summary plot
        """
        if self.shap_values is None:
            print("No SHAP values available")
            return

        try:
            import shap
            import matplotlib.pyplot as plt

            # 创建summary plot
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                feature_names=self.shap_results['feature_names'],
                show=False
            )

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            print("SHAP library not installed")