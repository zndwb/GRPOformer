import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math
from config import (
    DEVICE,
    # HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_HEADS,
    DROPOUT
)
from ConfigSpace import Configuration

class TransformerPolicyT5(nn.Module):
    def __init__(self, config_space: Dict):
        super().__init__()
        self.config_space = config_space
        self.param_names = list(config_space.keys())

        # Load T5 tokenizer and encoder model
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)
        self.hidden_size = self.t5.config.d_model # use configured hidden size

        # Output heads for each hyperparameter
        self.output_heads = self._build_output_heads().to(DEVICE)

    def _trajectory_to_text(self, trajectory: Dict) -> str:
        lines = ["Task description:"]
        td = trajectory["task_description"]["configuration_space"]
        for k, spec in td.items():
            lines.append(f"{k}: {spec}")
        ob=trajectory["task_description"]["objective"]
        lines.append("objective:")
        lines.append(ob)

        lines.append("History:")

        configurations = trajectory["configuration"]
        results = trajectory["result"]
        # 兼容单个 dict 或多个配置组成的 list
        if isinstance(configurations, dict):
            configurations = [configurations]
        if isinstance(results, dict):
            results = [results]

        for cfg, res in zip(configurations, results):
            cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
            perf = res["validation"]["val_accuarcy"]
            lines.append(f"cfg={{ {cfg_str} }}, perf={perf}")

        return "\n".join(lines)

    def _build_output_heads(self):
        output_heads = nn.ModuleDict()
        for param_name, spec in self.config_space.items():
            if spec["type"] in ["continuous", "integer"]:
                output_heads[param_name] = nn.Sequential(
                    nn.Linear(self.t5.config.d_model, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(self.hidden_size, 2),
                    nn.Tanh()
                )
            elif spec["type"] == "categorical":
                output_heads[param_name] = nn.Sequential(
                    nn.Linear(self.t5.config.d_model, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(self.hidden_size, len(spec["choices"])),
                    nn.Softmax(dim=-1)
                )
        return output_heads

    def encode_with_t5(self, trajectory: Dict) -> torch.Tensor:
        prompt = self._trajectory_to_text(trajectory)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
        encoder_outputs = self.t5.encoder(**inputs)
        hidden_states = encoder_outputs.last_hidden_state
        last_hidden = hidden_states[:, 0, :]  # take first token
        return last_hidden

    def forward(self, trajectory: Dict) -> Tuple[Dict, float, Dict]:
        # === 1. 提取 log 信息 ===
        log_info = {}
        config_space_raw = trajectory.get("task_description", {}).get("configuration_space", {})
        for param_name, param_spec in config_space_raw.items():
            log_info[param_name] = param_spec.get("log", False)

        # === 2. 对 trajectory["configs"] 中 log 参数做 log2 变换，用于 encode ===
        transformed_trajectory = trajectory.copy()
        original_config= trajectory.get("configuration")
        if original_config is None:
            raise ValueError("Missing 'configuration' in trajectory.")
        transformed_config_dict_all=[]
        # 创建新配置的 dict（注意要从 Configuration 中提取）
        for g in range(len(original_config)):
            transformed_config_dict = original_config[g].copy()
            # 处理 log 转换
            for param_name, value in transformed_config_dict.items():
                if log_info.get(param_name, False):
                    if value <= 0:
                        raise ValueError(f"Cannot take log2 of non-positive value: {param_name}={value}")
                    transformed_config_dict[param_name] = math.log2(value)
            transformed_config_dict_all.append(transformed_config_dict)
        # 更新 trajectory
        transformed_trajectory["configuration"] = transformed_config_dict_all
        # === 3. Encode 输入 ===
        last_hidden = self.encode_with_t5(transformed_trajectory)

        sampled_config = {}
        total_log_prob = 0.0
        dist_params = {}

        for param_name in self.param_names:
            spec = self.config_space[param_name]
            head_output = self.output_heads[param_name](last_hidden)

            if spec["type"] == "continuous":
                mean_raw, log_std_raw = torch.chunk(head_output, 2, dim=-1)

                if log_info.get(param_name, False):
                    # log2 建模
                    log_lower = math.log2(spec["lower"])
                    log_upper = math.log2(spec["upper"])
                    log_scale = log_upper - log_lower

                    mean_log = log_lower + (mean_raw + 1) * log_scale / 2
                    std_log = torch.exp(log_std_raw) * (log_scale / 10)

                    dist = Normal(mean_log, std_log)
                    sample_log = dist.rsample()
                    sampled_log = torch.clamp(sample_log, log_lower, log_upper)
                    sampled_value = 2 ** sampled_log.item()

                    total_log_prob += dist.log_prob(sample_log).item()
                    sampled_config[param_name] = sampled_value
                    dist_params[param_name] = {
                        "type": "normal_log",
                        "mean_log2": mean_log.item(),
                        "std_log2": std_log.item(),
                        "log_bounds": [log_lower, log_upper]
                    }

                else:
                    scale = spec["upper"] - spec["lower"]
                    mean = spec["lower"] + (mean_raw + 1) * scale / 2
                    std = torch.exp(log_std_raw) * (scale / 10)

                    dist = Normal(mean, std)
                    sample = dist.rsample()
                    sampled_value = torch.clamp(sample, spec["lower"], spec["upper"]).item()

                    total_log_prob += dist.log_prob(sample).item()
                    sampled_config[param_name] = sampled_value
                    dist_params[param_name] = {"type": "normal", "mean": mean.item(), "std": std.item()}

            elif spec["type"] == "integer":
                mean_raw, log_std_raw = torch.chunk(head_output, 2, dim=-1)

                if log_info.get(param_name, False):
                    log_lower = math.log2(spec["lower"])
                    log_upper = math.log2(spec["upper"])
                    log_scale = log_upper - log_lower

                    mean_log = log_lower + (mean_raw + 1) * log_scale / 2
                    std_log = torch.exp(log_std_raw) * (log_scale / 10)

                    dist = Normal(mean_log, std_log)
                    sample_log = dist.rsample()
                    sampled_log = torch.clamp(sample_log, log_lower, log_upper)
                    sampled_value = int(round(2 ** sampled_log.item()))
                    sampled_value = int(torch.clamp(torch.tensor(sampled_value), spec["lower"], spec["upper"]).item())

                    total_log_prob += dist.log_prob(sample_log).item()
                    sampled_config[param_name] = sampled_value
                    dist_params[param_name] = {
                        "type": "normal_log_discrete",
                        "mean_log2": mean_log.item(),
                        "std_log2": std_log.item(),
                        "log_bounds": [log_lower, log_upper]
                    }

                else:
                    scale = spec["upper"] - spec["lower"]
                    mean = spec["lower"] + (mean_raw + 1) * scale / 2
                    std = torch.exp(log_std_raw) * (scale / 10)

                    dist = Normal(mean, std)
                    sample = dist.rsample()
                    sampled_value = int(torch.clamp(torch.round(sample), spec["lower"], spec["upper"]).item())

                    total_log_prob += dist.log_prob(sample).item()
                    sampled_config[param_name] = sampled_value
                    dist_params[param_name] = {"type": "normal_discrete", "mean": mean.item(), "std": std.item()}

            elif spec["type"] == "categorical":
                probs = head_output
                dist = Categorical(probs=probs)
                idx = dist.sample()
                sampled_value = spec["choices"][idx.item()]

                total_log_prob += dist.log_prob(idx).item()
                sampled_config[param_name] = sampled_value
                dist_params[param_name] = {
                    "type": "categorical",
                    "probs": {k: v.item() for k, v in zip(spec["choices"], probs[0])}
                }

        return sampled_config, total_log_prob, dist_params

    def get_log_prob(self, trajectory: Dict, config: Dict) -> float:
        """
        计算给定配置在当前策略下的对数概率
        Args:
            config: 已知的超参数配置（目标）
            trajectory: 优化轨迹（用于上下文）
        Returns:
            total_log_prob: 总对数概率
        """
        # === 1. 提取 log 信息 ===
        log_info = {}
        config_space_raw = trajectory.get("task_description", {}).get("configuration_space", {})
        for param_name, param_spec in config_space_raw.items():
            log_info[param_name] = param_spec.get("log", False)

        # === 2. 对 trajectory["configs"] 中 log 参数做 log2 变换，用于 encode ===
        transformed_trajectory = trajectory.copy()
        original_config: Configuration = trajectory.get("configuration")
        if original_config is None:
            raise ValueError("Missing 'configuration' in trajectory.")
        transformed_config_dict_all = []
        # 创建新配置的 dict（注意要从 Configuration 中提取）
        for g in range(len(original_config)):
            transformed_config_dict = original_config[g].copy()
            # 处理 log 转换
            for param_name, value in transformed_config_dict.items():
                if log_info.get(param_name, False):
                    if value <= 0:
                        raise ValueError(f"Cannot take log2 of non-positive value: {param_name}={value}")
                    transformed_config_dict[param_name] = math.log2(value)
            transformed_config_dict_all.append(transformed_config_dict)
        # 更新 trajectory
        transformed_trajectory["configuration"] = transformed_config_dict_all

        last_hidden = self.encode_with_t5(transformed_trajectory)
        total_log_prob = 0.0

        # === 4. 针对每个参数计算 log_prob ===
        for param_name in self.param_names:
            spec = self.config_space[param_name]
            head_output = self.output_heads[param_name](last_hidden)

            if spec["type"] == "continuous":
                mean_raw, log_std_raw = torch.chunk(head_output, 2, dim=-1)
                if log_info.get(param_name, False):
                    # 在 log2 空间内建模
                    log_lower = math.log2(spec["lower"])
                    log_upper = math.log2(spec["upper"])
                    scale = log_upper - log_lower
                    mean_log = log_lower + (mean_raw + 1) * scale / 2
                    std_log = torch.exp(log_std_raw) * (scale / 10)
                    dist = Normal(mean_log, std_log)

                    # 注意：输入的 config[param_name] 也要转换到 log2 空间
                    raw_value = config[param_name]
                    if raw_value <= 0:
                        raise ValueError(f"Cannot take log2 of non-positive config value: {param_name}={raw_value}")
                    value = torch.tensor(math.log2(raw_value), device=mean_log.device)
                else:
                    scale = spec["upper"] - spec["lower"]
                    mean = spec["lower"] + (mean_raw + 1) * scale / 2
                    std = torch.exp(log_std_raw) * (scale / 10)
                    dist = Normal(mean, std)
                    value = torch.tensor(config[param_name], device=mean.device)

                total_log_prob += dist.log_prob(value)

            elif spec["type"] == "integer":
                mean_raw, log_std_raw = torch.chunk(head_output, 2, dim=-1)
                if log_info.get(param_name, False):
                    # 在 log2 空间内建模
                    log_lower = math.log2(spec["lower"])
                    log_upper = math.log2(spec["upper"])
                    scale = log_upper - log_lower
                    mean_log = log_lower + (mean_raw + 1) * scale / 2
                    std_log = torch.exp(log_std_raw) * (scale / 10)
                    dist = Normal(mean_log, std_log)

                    raw_value = config[param_name]
                    if raw_value <= 0:
                        raise ValueError(f"Cannot take log2 of non-positive config value: {param_name}={raw_value}")
                    value = torch.tensor(math.log2(raw_value), device=mean_log.device, dtype=torch.float32)
                else:
                    scale = spec["upper"] - spec["lower"]
                    mean = spec["lower"] + (mean_raw + 1) * scale / 2
                    std = torch.exp(log_std_raw) * (scale / 10)
                    dist = Normal(mean, std)
                    value = torch.tensor(config[param_name], device=mean.device, dtype=torch.float32)

                total_log_prob += dist.log_prob(value)

            elif spec["type"] == "categorical":
                probs = head_output
                dist = Categorical(probs=probs)
                idx = spec["choices"].index(config[param_name])
                total_log_prob += dist.log_prob(torch.tensor(idx, device=probs.device))

        return total_log_prob

    def save_model(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.output_heads.state_dict(), os.path.join(save_dir, "output_heads.pt"))
        self.t5.save_pretrained(os.path.join(save_dir, "t5_model"))
        self.tokenizer.save_pretrained(os.path.join(save_dir, "t5_model"))
        torch.save(self.config_space, os.path.join(save_dir, "config_space.pt"))

    @classmethod
    def load_model(cls, load_dir: str):
        config_space = torch.load(os.path.join(load_dir, "config_space.pt"))
        model = cls(config_space=config_space)
        model.output_heads.load_state_dict(torch.load(os.path.join(load_dir, "output_heads.pt"), map_location=DEVICE))
        model.t5 = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(load_dir, "t5_model")).to(DEVICE)
        model.tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_dir, "t5_model"))
        return model

