# training/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RankingLoss(nn.Module):
    """
    排序损失函数
    支持多种排序损失：Pairwise Hinge, ListNet, LambdaRank
    """

    def __init__(self, loss_type='pairwise_hinge', margin=0.1):
        """
        Parameters:
        -----------
        loss_type : str, 'pairwise_hinge', 'pairwise_logistic', 'listnet'
        margin : float, margin for hinge loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.margin = margin

    def forward(self, scores, targets):
        """
        Parameters:
        -----------
        scores : (batch_size,) 预测得分
        targets : (batch_size,) 真实标签（超额收益）

        Returns:
        --------
        loss : scalar
        """
        batch_size = scores.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=scores.device)

        if self.loss_type == 'pairwise_hinge':
            return self._pairwise_hinge_loss(scores, targets)
        elif self.loss_type == 'pairwise_logistic':
            return self._pairwise_logistic_loss(scores, targets)
        elif self.loss_type == 'listnet':
            return self._listnet_loss(scores, targets)
        elif self.loss_type == 'lambdarank':
            return self._lambdarank_loss(scores, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _pairwise_hinge_loss(self, scores, targets):
        """Pairwise Hinge Loss"""
        # 生成所有配对
        i_idx, j_idx = torch.triu_indices(len(scores), len(scores), offset=1)

        if i_idx.size(0) == 0:
            return torch.tensor(0.0, device=scores.device)

        # 获取配对的得分和标签
        scores_i = scores[i_idx]
        scores_j = scores[j_idx]
        targets_i = targets[i_idx]
        targets_j = targets[j_idx]

        # 判断排序是否正确
        correct_order = (targets_i > targets_j).float()

        # Hinge Loss
        loss = torch.max(
            torch.zeros_like(scores_i),
            self.margin - (scores_i - scores_j) * (2 * correct_order - 1)
        )

        return loss.mean()

    def _pairwise_logistic_loss(self, scores, targets):
        """Pairwise Logistic Loss"""
        i_idx, j_idx = torch.triu_indices(len(scores), len(scores), offset=1)

        if i_idx.size(0) == 0:
            return torch.tensor(0.0, device=scores.device)

        scores_i = scores[i_idx]
        scores_j = scores[j_idx]
        targets_i = targets[i_idx]
        targets_j = targets[j_idx]

        # 计算配对概率
        prob = torch.sigmoid(scores_i - scores_j)

        # 真实标签
        y = (targets_i > targets_j).float()

        # Logistic Loss
        loss = F.binary_cross_entropy(prob, y)

        return loss

    def _listnet_loss(self, scores, targets):
        """ListNet Loss (top-1 probability)"""
        # 使用softmax计算得分分布
        scores_softmax = F.softmax(scores, dim=0)
        targets_softmax = F.softmax(targets, dim=0)

        # 交叉熵损失
        loss = -torch.sum(targets_softmax * torch.log(scores_softmax + 1e-8))

        return loss

    def _lambdarank_loss(self, scores, targets):
        """LambdaRank Loss"""
        # 计算排序
        sorted_indices = torch.argsort(targets, descending=True)
        sorted_scores = scores[sorted_indices]
        sorted_targets = targets[sorted_indices]

        # 计算NDCG增益
        gains = torch.pow(2.0, sorted_targets) - 1.0
        discounts = torch.log2(torch.arange(1, len(gains) + 1, device=gains.device).float() + 1.0)
        dcg = torch.sum(gains / discounts)

        # 计算IDCG
        ideal_gains = torch.sort(gains, descending=True)[0]
        ideal_dcg = torch.sum(ideal_gains / discounts)

        # NDCG
        ndcg = dcg / (ideal_dcg + 1e-8)

        # Lambda权重
        lambdas = torch.zeros_like(scores)

        for i in range(len(scores)):
            for j in range(len(scores)):
                if i != j:
                    if targets[i] > targets[j]:
                        lambdas[i] += (scores[i] - scores[j]).abs() * (1.0 / (1 + torch.exp(scores[i] - scores[j])))
                    else:
                        lambdas[i] -= (scores[i] - scores[j]).abs() * (1.0 / (1 + torch.exp(scores[i] - scores[j])))

        # LambdaRank损失
        loss = -torch.sum(lambdas * torch.log(torch.sigmoid(scores)))

        return loss


class MultiTaskLoss(nn.Module):
    """
    多任务损失
    组合排序损失和波动率损失
    """

    def __init__(self, ranking_loss_weight=1.0, vol_loss_weight=0.1,
                 regime_loss_weight=0.1, decomp_loss_weight=0.05):
        super().__init__()
        self.ranking_loss_weight = ranking_loss_weight
        self.vol_loss_weight = vol_loss_weight
        self.regime_loss_weight = regime_loss_weight
        self.decomp_loss_weight = decomp_loss_weight

        # 排序损失
        self.ranking_loss = RankingLoss(loss_type='pairwise_hinge')

        # 波动率损失
        self.vol_loss = nn.MSELoss()

        # 分类损失（市场状态）
        self.regime_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Parameters:
        -----------
        outputs : dict from model
            - 'rank_score': (batch, 1)
            - 'vol_pred': (batch, 1)
            - 'regime_probs': (batch, n_regimes)
        targets : dict
            - 'rank_target': (batch,)
            - 'vol_target': (batch,)
            - 'regime_target': (batch,) optional
        """
        total_loss = 0
        loss_dict = {}

        # 1. 排序损失
        if 'rank_score' in outputs and 'rank_target' in targets:
            rank_loss = self.ranking_loss(
                outputs['rank_score'].squeeze(-1),
                targets['rank_target']
            )
            loss_dict['rank_loss'] = rank_loss.item()
            total_loss += self.ranking_loss_weight * rank_loss

        # 2. 波动率损失
        if 'vol_pred' in outputs and 'vol_target' in targets:
            vol_loss = self.vol_loss(
                outputs['vol_pred'].squeeze(-1),
                targets['vol_target']
            )
            loss_dict['vol_loss'] = vol_loss.item()
            total_loss += self.vol_loss_weight * vol_loss

        # 3. 市场状态损失（如果有）
        if 'regime_probs' in outputs and 'regime_target' in targets:
            regime_loss = self.regime_loss(
                outputs['regime_probs'],
                targets['regime_target']
            )
            loss_dict['regime_loss'] = regime_loss.item()
            total_loss += self.regime_loss_weight * regime_loss

        # 4. 时序分解损失（如果有）
        if 'decomposition' in outputs and 'decomposition_loss' in outputs['decomposition']:
            decomp_loss = outputs['decomposition']['decomposition_loss']
            loss_dict['decomp_loss'] = decomp_loss.item()
            total_loss += self.decomp_loss_weight * decomp_loss

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class AdaptiveLossWeighting(nn.Module):
    """
    自适应损失加权（创新）
    根据损失值动态调整各任务权重
    """

    def __init__(self, n_tasks=2, initial_weights=None, temperature=1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.temperature = temperature

        if initial_weights is None:
            initial_weights = torch.ones(n_tasks) / n_tasks

        self.weights = nn.Parameter(torch.log(initial_weights))

    def forward(self, losses):
        """
        Parameters:
        -----------
        losses : list of task losses

        Returns:
        --------
        weighted_loss : scalar
        """
        # Softmax weights
        weights = F.softmax(self.weights / self.temperature, dim=0)

        # 加权求和
        weighted_loss = torch.sum(weights * torch.stack(losses))

        return weighted_loss