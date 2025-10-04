import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple, Any
from config import (
    DEVICE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_HEADS,
    DROPOUT
)


class TransformerPolicy(nn.Module):
    """基于Transformer的策略网络，用于超参数优化任务"""

    def __init__(self, config_space: Dict):
        super().__init__()
        self.config_space = config_space
        self.param_names = list(config_space.keys())
        self.param_info = config_space

        # 1. 任务描述编码器（解析超参数空间和元信息）
        self.task_encoder = self._build_task_encoder()

        # 2. 轨迹优化轨迹编码器（处理历史配置和性能）
        self.trajectory_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=HIDDEN_SIZE,
                nhead=NUM_HEADS,
                dim_feedforward=HIDDEN_SIZE * 4,
                dropout=DROPOUT,
                batch_first=True
            ),
            num_layers=NUM_LAYERS
        )

        # 3. 输出头（生成超参数分布）
        self.output_heads = self._build_output_heads()

        # 位置编码（增强序列时序信息）
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, HIDDEN_SIZE))

        # 初始化权重
        self._initialize_weights()
        self.num_features = len(self.param_names) + 1  # 参数数 + 性能值
        self.step_projector = nn.Linear(self.num_features, HIDDEN_SIZE)

    def _build_task_encoder(self) -> nn.ModuleDict:
        """构建        构建任务描述编码器
        解析超参数空间约束和元信息，将结构化信息转换为向量
        """
        module_dict = nn.ModuleDict()

        # 为每个超参数构建专用编码器
        for param_name, spec in self.param_info.items():
            if spec["type"] in ["continuous", "integer"]:
                # 连续/整数参数：编码上下界信息
                module_dict[f"param_{param_name}"] = nn.Linear(2, HIDDEN_SIZE // len(self.param_names))
            elif spec["type"] == "categorical":
                # 类别参数：编码选项集合
                module_dict[f"param_{param_name}"] = nn.Embedding(
                    num_embeddings=len(spec["choices"]),
                    embedding_dim=HIDDEN_SIZE // len(self.param_names)
                )

        # 元信息编码器（处理数据集特性等）
        module_dict["meta_encoder"] = nn.Sequential(
            nn.Linear(5, HIDDEN_SIZE // 4),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE // 4, HIDDEN_SIZE // 4)
        )

        # 任务信息聚合层
        module_dict["task_aggregator"] = nn.Sequential(
            nn.Linear(len(self.param_names) * (HIDDEN_SIZE // len(self.param_names)) + HIDDEN_SIZE // 4, HIDDEN_SIZE),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_SIZE)
        )

        return module_dict

    def _build_output_heads(self) -> nn.ModuleDict:
        """构建输出头，为不同类型的超参数生成概率分布参数"""
        output_heads = nn.ModuleDict()

        for param_name, spec in self.param_info.items():
            if spec["type"] in ["continuous", "integer"]:
                # 连续/整数参数：输出均值和标准差
                output_heads[param_name] = nn.Sequential(
                    nn.Linear(HIDDEN_SIZE, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2),  # [均值, 对数标准差]
                    nn.Tanh()  # 将输出限制在[-1, 1]便于后续映射到参数范围
                )
            elif spec["type"] == "categorical":
                # 类别参数：输出每个选项的logits
                output_heads[param_name] = nn.Sequential(
                    nn.Linear(HIDDEN_SIZE, 64),
                    nn.ReLU(),
                    nn.Linear(64, len(spec["choices"])),
                    nn.Softmax(dim=-1)  # 转换为概率分布
                )

        return output_heads

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def encode_task_description(self, task_desc: Dict, meta_info: Dict) -> torch.Tensor:
        """
        将任务描述（超参数空间）和元信息编码为向量
        Args:
            task_desc: 包含超参数空间约束的字典
            meta_info: 包含数据集特性的元信息字典
        Returns:
            编码后的任务向量 [1, 1, HIDDEN_SIZE]
        """
        # 1. 编码超参数空间约束
        param_embeddings = []
        for param_name, spec in task_desc["configuration_space"].items():
            if spec["type"] in ["continuous", "integer"]:
                # 归一化上下界到[0, 1]
                bounds = torch.tensor(
                    [spec["lower"], spec["upper"]],
                    device=DEVICE, dtype=torch.float32
                )
                # 标准化：(x - min) / (max - min)
                bounds_norm = (bounds - bounds[0]) / (bounds[1] - bounds[0] + 1e-8)
                bounds_norm = bounds_norm.to(DEVICE)
                emb = self.task_encoder[f"param_{param_name}"](bounds_norm)
                param_embeddings.append(emb)

            elif spec["type"] == "categorical":
                # 使用第一个类别的嵌入代表整个类别空间特征
                emb = self.task_encoder[f"param_{param_name}"](
                    torch.tensor(0, device=DEVICE, dtype=torch.long)
                )
                param_embeddings.append(emb)

        # 2. 编码元信息（提取关键特征）
        meta_features = torch.tensor([
            meta_info.get("n_samples", 0) / 10000,  # 样本数（归一化）
            meta_info.get("n_features", 0) / 100,  # 特征数（归一化）
            meta_info.get("n_classes", 0) / 10,  # 类别数（归一化）
            meta_info.get("dataset_id", 0) / 1000,  # 数据集ID（归一化）
            1.0 if meta_info.get("task_type") == "classification" else 0.0  # 任务类型（分类/回归）
        ], device=DEVICE, dtype=torch.float32)
        meta_emb = self.task_encoder["meta_encoder"](meta_features)
        param_embeddings.append(meta_emb)

        # 3. 聚合所有任务特征
        task_emb = torch.cat(param_embeddings, dim=0).unsqueeze(0).unsqueeze(0)
        task_emb = self.task_encoder["task_aggregator"](task_emb)

        return task_emb  # 形状: [1, 1, HIDDEN_SIZE]

    def encode_trajectory(self, trajectory: Dict) -> torch.Tensor:
        """
        编码完整的优化轨迹（任务描述 + 历史配置 + 性能结果）
        Args:
            trajectory: 包含任务描述、元信息、配置历史和结果的字典
        Returns:
            编码后的轨迹序列 [1, seq_len, HIDDEN_SIZE]
        """
        # 1. 编码任务描述（作为序列的第一个元素）
        task_emb = self.encode_task_description(
            trajectory["task_description"],
            trajectory["meta_information"]
        )  # [1, 1, HIDDEN_SIZE]

        # 2. 编码历史优化步骤
        step_embeddings = []
        for config, result in zip(trajectory["configuration"], trajectory["result"]):
            # 编码超参数配置
            config_features = []
            for param_name in self.param_names:
                value = config[param_name]
                spec = self.param_info[param_name]

                # 归一化参数值到[-1, 1]范围
                if spec["type"] in ["continuous", "integer"]:
                    norm_val = 2 * (value - spec["lower"]) / (spec["upper"] - spec["lower"] + 1e-8) - 1
                    config_features.append(torch.tensor(norm_val, device=DEVICE))
                elif spec["type"] == "categorical":
                    idx = spec["choices"].index(value)
                    norm_val = 2 * idx / (len(spec["choices"]) - 1 + 1e-8) - 1 if len(spec["choices"]) > 1 else 0
                    config_features.append(torch.tensor(norm_val, device=DEVICE))

            # 编码性能指标（验证集损失）
            val_perf = result["validation"]["function_value"]
            # 归一化性能值（假设损失范围在0-10之间）
            norm_perf = 2 * (min(max(val_perf, 0), 10) / 10) - 1  # 映射到[-1, 1]
            config_features.append(torch.tensor(norm_perf, device=DEVICE))

            # 转换为向量并映射到隐藏维度
            config_tensor = torch.stack(config_features).float().unsqueeze(0)  # [1, num_params+1]
            step_emb = self.step_projector(config_tensor).unsqueeze(1)
            step_embeddings.append(step_emb)

        # 3. 构建完整序列（任务描述 + 历史步骤）
        if step_embeddings:
            steps_tensor = torch.cat(step_embeddings, dim=1)  # [1, num_steps, HIDDEN_SIZE]
            full_seq = torch.cat([task_emb, steps_tensor], dim=1)  # 任务描述作为序列首元素
        else:
            full_seq = task_emb  # 初始状态：只有任务描述

        # 添加位置编码
        full_seq += self.positional_encoding[:, :full_seq.size(1), :]

        return full_seq  # 形状: [1, seq_len, HIDDEN_SIZE]

    def forward(self, trajectory: Dict) -> Tuple[Dict, float, Dict]:
        """
        前向传播：生成超参数配置并计算对数概率
        Args:
            trajectory: 当前优化轨迹
        Returns:
            sampled_config: 采样的超参数配置
            total_log_prob: 总对数概率
            dist_params: 分布参数（用于记录和调试）
        """
        # 1. 编码轨迹
        seq_emb = self.encode_trajectory(trajectory)
        encoded_seq = self.trajectory_encoder(seq_emb)
        last_hidden = encoded_seq[:, -1, :]  # 取最后一个时间步的隐藏状态

        # 2. 为每个超参数生成分布并采样
        sampled_config = {}
        total_log_prob = 0.0
        dist_params = {}

        for param_name in self.param_names:
            spec = self.param_info[param_name]
            head_output = self.output_heads[param_name](last_hidden)

            if spec["type"] == "continuous":
                # 连续参数：使用正态分布
                mean_raw, log_std_raw = torch.chunk(head_output, 2, dim=-1)

                # 将输出映射到参数范围
                scale = spec["upper"] - spec["lower"]
                mean = spec["lower"] + (mean_raw + 1) * scale / 2  # 从[-1,1]映射到[lower, upper]
                std = torch.exp(log_std_raw) * (scale / 10)  # 限制标准差大小（避免采样范围过大）

                # 创建分布并采样
                dist = Normal(mean, std)
                sample = dist.rsample()  # 重参数化采样
                sampled_value = torch.clamp(sample, spec["lower"], spec["upper"]).item()

                # 记录对数概率和分布参数
                total_log_prob += dist.log_prob(sample).item()
                sampled_config[param_name] = sampled_value
                dist_params[param_name] = {
                    "type": "normal",
                    "mean": mean.item(),
                    "std": std.item()
                }

            elif spec["type"] == "integer":
                # 整数参数：正态分布采样后取整
                mean_raw, log_std_raw = torch.chunk(head_output, 2, dim=-1)

                # 将输出映射到参数范围
                scale = spec["upper"] - spec["lower"]
                mean = spec["lower"] + (mean_raw + 1) * scale / 2
                std = torch.exp(log_std_raw) * (scale / 10)

                # 创建分布并采样
                dist = Normal(mean, std)
                sample = dist.rsample()
                sampled_value = torch.clamp(torch.round(sample), spec["lower"], spec["upper"]).item()
                sampled_value = int(sampled_value)  # 转换为整数

                # 记录对数概率和分布参数
                total_log_prob += dist.log_prob(sample).item()
                sampled_config[param_name] = sampled_value
                dist_params[param_name] = {
                    "type": "normal_discrete",
                    "mean": mean.item(),
                    "std": std.item()
                }

            elif spec["type"] == "categorical":
                # 类别参数：使用类别分布
                probs = head_output  # 已通过softmax
                dist = Categorical(probs=probs)
                sample_idx = dist.sample()
                sampled_value = spec["choices"][sample_idx.item()]

                # 记录对数概率和分布参数
                total_log_prob += dist.log_prob(sample_idx).item()
                sampled_config[param_name] = sampled_value
                dist_params[param_name] = {
                    "type": "categorical",
                    "probs": {k: v.item() for k, v in zip(spec["choices"], probs[0])}
                }

        return sampled_config, total_log_prob, dist_params

    def get_log_prob(self, trajectory: Dict, config: Dict) -> float:
        """
        计算给定配置在当前轨迹下的对数概率（用于强化学习更新）
        Args:
            trajectory: 当前优化轨迹
            config: 超参数配置
        Returns:
            配置的对数概率总和
        """
        # 编码轨迹
        seq_emb = self.encode_trajectory(trajectory)
        encoded_seq = self.trajectory_encoder(seq_emb)
        last_hidden = encoded_seq[:, -1, :]

        total_log_prob = 0.0

        for param_name, value in config.items():
            spec = self.param_info[param_name]
            head_output = self.output_heads[param_name](last_hidden)

            if spec["type"] == "continuous":
                mean_raw, log_std_raw = torch.chunk(head_output, 2, dim=-1)
                scale = spec["upper"] - spec["lower"]
                mean = spec["lower"] + (mean_raw + 1) * scale / 2
                std = torch.exp(log_std_raw) * (scale / 10)

                dist = Normal(mean, std)
                value_tensor = torch.tensor(value, device=DEVICE, dtype=torch.float32).unsqueeze(0)
                total_log_prob += dist.log_prob(value_tensor)

            elif spec["type"] == "integer":
                mean_raw, log_std_raw = torch.chunk(head_output, 2, dim=-1)
                scale = spec["upper"] - spec["lower"]
                mean = spec["lower"] + (mean_raw + 1) * scale / 2
                std = torch.exp(log_std_raw) * (scale / 10)

                dist = Normal(mean, std)
                value_tensor = torch.tensor(value, device=DEVICE, dtype=torch.float32).unsqueeze(0)
                total_log_prob += dist.log_prob(value_tensor)

            elif spec["type"] == "categorical":
                probs = head_output
                dist = Categorical(probs=probs)
                sample_idx = torch.tensor(spec["choices"].index(value), device=DEVICE)
                total_log_prob += dist.log_prob(sample_idx)

        return total_log_prob

    def save_model(self, path: str) -> None:
        """保存模型参数"""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config_space": self.config_space
        }, path)
        print(f"模型已保存至: {path}")

    def load_model(self, path: str) -> None:
        """加载模型参数"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.config_space = checkpoint["config_space"]
        self.param_names = list(self.config_space.keys())
        self.param_info = self.config_space
        print(f"模型已从: {path} 加载")
