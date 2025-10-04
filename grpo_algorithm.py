import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
import numpy as np
from config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRPO_CLIP_EPS,
    GRPO_GAMMA,
    GRPO_LAMBDA,
    GRPO_EPOCHS,
    GRPO_ENTROPY_COEF,
    GRPO_C,
    GROUP_SIZE
)


class GRPO:
    """Group Relative Policy Optimization (GRPO) 算法实现"""

    def __init__(self, policy: nn.Module):
        self.policy = policy
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # GRPO参数
        self.clip_eps = GRPO_CLIP_EPS
        self.gamma = GRPO_GAMMA
        self.lmbda = GRPO_LAMBDA
        self.epochs = GRPO_EPOCHS
        self.entropy_coef = GRPO_ENTROPY_COEF
        self.c = GRPO_C  # 群体正则化系数

        # 记录训练历史
        self.loss_history = []
        self.policy_loss_history = []
        self.entropy_history = []

    def compute_advantages(self, rewards: List[float], dones: List[bool]) -> torch.Tensor:
        """
        计算优势估计（GAE）
        在GRPO中，我们不使用critic，而是使用简化的优势计算
        """
        advantages = []
        last_advantage = 0

        # 反向计算优势
        for t in reversed(range(len(rewards))):
            delta = rewards[t] - last_advantage
            last_advantage = rewards[t] + self.gamma * (1 - dones[t]) * last_advantage
            advantages.insert(0, last_advantage)

        # 标准化优势
        advantages = torch.tensor(advantages, device=DEVICE, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    # def compute_advantages(self, rewards: List[float], dones: List[bool]) -> torch.Tensor:
    #     """
    #     GRPO 原文中的优势计算方式：
    #     按组计算优势：advantage_i = reward_i - mean(rewards)
    #     然后对组内优势进行标准化。
    #     不使用 critic 模型，只是 sample 多个成员的 reward。
    #     """
    #     rewards = torch.tensor(rewards, device=DEVICE, dtype=torch.float32)
    #     # 计算 baseline：组内平均 reward
    #     baseline = rewards.mean()
    #     advantages = rewards - baseline  # 相对优势
    #
    #     # 标准化：zero-mean, unit-variance
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #
    #     return advantages

    def update(self, groups: List[List[Tuple[Dict, Dict, float, float, bool]]]) -> Dict[str, float]:
        """
        使用GRPO更新策略网络
        groups: 群体列表，每个群体包含多个轨迹片段
                每个轨迹片段是(状态, 动作, 新策略对数概率, 旧策略对数概率, 奖励, 是否结束)的元组
        """
        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy = 0.0

        # 对每个群体进行更新
        for group in groups:
            # 提取群体数据
            states, actions, new_log_probs, old_log_probs, rewards, dones = zip(*group)

            # 转换为张量
            new_log_probs = torch.tensor(new_log_probs, device=DEVICE, dtype=torch.float32)
            old_log_probs = torch.tensor(old_log_probs, device=DEVICE, dtype=torch.float32)
            rewards = torch.tensor(rewards, device=DEVICE, dtype=torch.float32)
            dones = torch.tensor(dones, device=DEVICE, dtype=torch.float32)

            # 计算优势
            advantages = self.compute_advantages(rewards, dones)

            # 多轮更新
            for _ in range(self.epochs):
                # 重新计算新策略的对数概率（确保使用最新的策略参数）
                new_log_probs_recomputed = []
                for state, action in zip(states, actions):
                    # with torch.no_grad():
                    log_prob = self.policy.get_log_prob(state, action)
                    new_log_probs_recomputed.append(log_prob)
                # new_log_probs_recomputed = torch.tensor(new_log_probs_recomputed, device=DEVICE, dtype=torch.float32)
                new_log_probs_recomputed = torch.stack(new_log_probs_recomputed)

                # 计算概率比率
                ratios = torch.exp(new_log_probs_recomputed - old_log_probs)

                # 计算群体平均比率和优势
                group_avg_ratio = ratios.mean()
                group_avg_advantage = advantages.mean()

                # 计算相对优势和相对比率（GRPO核心）
                relative_advantages = advantages - group_avg_advantage
                relative_ratios = ratios / (group_avg_ratio + 1e-8)

                # 计算策略损失
                surr1 = relative_ratios * relative_advantages
                surr2 = torch.clamp(relative_ratios, 1 - self.clip_eps, 1 + self.clip_eps) * relative_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 群体正则化项（GRPO特色）
                group_regularization = self.c * torch.square(ratios - 1).mean()
                policy_loss += group_regularization

                # 计算熵（鼓励探索）
                entropy = -(torch.exp(new_log_probs_recomputed) * new_log_probs_recomputed).mean()

                # 总损失
                loss = policy_loss - self.entropy_coef * entropy

                # 反向传播和优化
                self.optimizer.zero_grad()
                # print("loss requires_grad:", loss.requires_grad)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

                self.optimizer.step()

                # 累积损失
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_entropy += entropy.item()

        # 计算平均损失
        num_updates = len(groups) * self.epochs
        avg_loss = total_loss / num_updates
        avg_policy_loss = total_policy_loss / num_updates
        avg_entropy = total_entropy / num_updates

        # 记录历史
        self.loss_history.append(avg_loss)
        self.policy_loss_history.append(avg_policy_loss)
        self.entropy_history.append(avg_entropy)

        return {
            "total_loss": avg_loss,
            "policy_loss": avg_policy_loss,
            "entropy": avg_entropy
        }

    def get_metrics(self) -> Dict[str, List[float]]:
        """获取训练指标历史"""
        return {
            "loss": self.loss_history,
            "policy_loss": self.policy_loss_history,
            "entropy": self.entropy_history
        }
