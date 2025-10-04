import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from HPOBench.hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from HPOBench.hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from HPOBench.hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
from HPOBench.hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from HPOBench.hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from hpobench.util.rng_helper import get_rng
from HPOBench.hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from collections import deque
import random
import logging
from tqdm import tqdm


# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"使用设备: {DEVICE}")

# --------------------------
# 配置参数
# --------------------------
ttttt=5
CONFIG = {
    "model_type": "rf",  # 可选: "xgboost", "svm", "nn", "lr", "rf"
    "task_id": "167155",  # HPOBench 任务ID
    "n_iter": 50,  # 总迭代次数
    "samples_per_iter": 10,  # 每个迭代采样超参数数量
    "max_seq_len": 1024,  # OptFormer 最大序列长度
    "t5_model_name": "google/flan-t5-small",  # T5 基础模型
    "quantization_level": 1000,  # 数值量化等级
    "acquisition_fn": "EI",  # 采集函数
    "num_candidates": 30,  # 采集函数候选数量
    "gamma": 0.95,  # 折扣因子
    "random_seed": 42  # 随机种子
}


# --------------------------
# 1. OptFormer 核心组件
# --------------------------
class OptFormerSerializer:
    """研究序列化组件：将元数据和优化轨迹转为文本序列"""

    def __init__(self, quantization_level=1000):
        self.Q = quantization_level  # 量化等级
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    def _normalize_quantize(self, value, min_val, max_val):
        """数值归一化+量化"""
        if max_val == min_val:
            return 0
        x_norm = (value - min_val) / (max_val - min_val)
        x_norm = np.clip(x_norm, 0.0, 1.0)
        return int(x_norm * self.Q)

    def _denormalize_dequantize(self, quantized_val, min_val, max_val):
        """量化值反归一化"""
        x_norm = quantized_val / self.Q
        return x_norm * (max_val - min_val) + min_val

    def serialize_metadata(self, metadata):
        """序列化元数据"""
        meta_str = f'<name>:{metadata["name"]},<metric>:{metadata["metric"]},<goal>:{metadata["goal"]},<algorithm>:{metadata["algorithm"]}'
        # 追加超参数空间信息
        for param in metadata["parameters"]:
            param_str = f'&<name>:{param["name"]},<type>:{param["type"]}'
            if param["type"] in ["DOUBLE", "INTEGER"]:
                param_str += f',<min>:{param["min"]},<max>:{param["max"]},<scale>:{param["scale"]}'
            elif param["type"] == "CATEGORICAL":
                param_str += f',<categories>:{",".join(param["categories"])}'
            meta_str += param_str
        return meta_str

    def serialize_history(self, history, param_info):
        """序列化优化轨迹"""
        history_str = ""
        # 计算函数值y的范围（用于量化）
        y_list = [h[1] for h in history]
        y_min, y_max = min(y_list), max(y_list) if len(y_list) > 0 else (0.0, 1.0)

        for param_dict, y in history:
            # 序列化当前试验的超参数
            param_tokens = []
            for p_info in param_info:
                p_name = p_info["name"]
                p_val = param_dict[p_name]
                p_type = p_info["type"]

                if p_type in ["DOUBLE", "INTEGER"]:
                    # 数值型：归一化+量化
                    quantized = self._normalize_quantize(p_val, p_info["min"], p_info["max"])
                    param_tokens.append(str(quantized))
                elif p_type == "CATEGORICAL":
                    # 分类型：用类别索引
                    param_tokens.append(str(p_info["categories"].index(p_val)))

            # 序列化函数值y
            quantized_y = self._normalize_quantize(y, y_min, y_max)
            # 拼接试验：参数+*+y+|
            trial_str = " ".join(param_tokens) + f" * {quantized_y} | "
            history_str += trial_str

        return history_str, (y_min, y_max)

    def deserialize_param(self, quantized_tokens, param_info):
        """反序列化超参数"""
        param_dict = {}
        for idx, p_info in enumerate(param_info):
            p_name = p_info["name"]
            p_type = p_info["type"]
            quantized_val = int(quantized_tokens[idx])

            if p_type in ["DOUBLE", "INTEGER"]:
                # 数值型：反量化
                raw_val = self._denormalize_dequantize(quantized_val, p_info["min"], p_info["max"])
                param_dict[p_name] = int(raw_val) if p_type == "INTEGER" else raw_val
            elif p_type == "CATEGORICAL":
                # 分类型：索引→类别
                param_dict[p_name] = p_info["categories"][quantized_val]
        return param_dict

    def deserialize_y(self, quantized_y, y_min, y_max):
        """反序列化函数值y"""
        return self._denormalize_dequantize(quantized_y, y_min, y_max)


class OptFormer(nn.Module):
    """OptFormer模型（基于T5编码器-解码器）"""

    def __init__(self, t5_model_name):
        super().__init__()
        # 使用指定方式加载T5模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(DEVICE)
        self.device = DEVICE

    def forward(self, input_seq, target_seq=None):
        """前向传播：输入序列→预测目标序列"""
        # 编码输入序列
        input_ids = self.tokenizer(
            input_seq,
            max_length=CONFIG["max_seq_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)

        if target_seq is not None:
            # 训练模式：计算预测损失
            target_ids = self.tokenizer(
                target_seq,
                max_length=CONFIG["max_seq_len"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)

            # 计算损失
            loss = self.t5_model(
                input_ids=input_ids,
                labels=target_ids
            ).loss

            # 过滤分隔符损失
            sep_tokens = self.tokenizer(["*", "|"], add_special_tokens=False)["input_ids"]
            sep_token_ids = [token for sublist in sep_tokens for token in sublist]
            mask = torch.ones_like(target_ids, dtype=torch.bool)
            for sep_id in sep_token_ids:
                mask &= (target_ids != sep_id)
            masked_loss = loss * mask.float().mean()
            return masked_loss

        else:
            # 推理模式：生成下一个试验的预测
            generated_ids = self.t5_model.generate(
                input_ids=input_ids,
                max_length=CONFIG["max_seq_len"],
                num_return_sequences=1,
                early_stopping=True
            )
            generated_seq = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_seq


class AcquisitionFunction:
    """采集函数实现"""

    @staticmethod
    def ei(y_pred_dist, y_best_quantized, Q=CONFIG["quantization_level"]):
        """Expected Improvement：计算期望改进值"""
        ei_val = 0.0
        for y_quant in range(Q):
            if y_quant <= y_best_quantized:
                continue
            improvement = y_quant - y_best_quantized
            prob = y_pred_dist.get(y_quant, 0.0)
            ei_val += improvement * prob
        return ei_val

    @staticmethod
    def pi(y_pred_dist, y_best_quantized, Q=CONFIG["quantization_level"]):
        """Probability of Improvement：计算改进概率"""
        pi_val = 0.0
        for y_quant in range(Q):
            if y_quant > y_best_quantized:
                pi_val += y_pred_dist.get(y_quant, 0.0)
        return pi_val

    @staticmethod
    def ucb(y_pred_dist, alpha=0.9, Q=CONFIG["quantization_level"]):
        """Upper Confidence Bound：计算上置信界"""
        sorted_y = sorted(y_pred_dist.items(), key=lambda x: x[0])
        cumulative_prob = 0.0
        ucb_val = 0.0
        for y_quant, prob in sorted_y:
            cumulative_prob += prob
            if cumulative_prob >= alpha:
                ucb_val = y_quant
                break
        return ucb_val

    @staticmethod
    def ts(y_pred_dist, Q=CONFIG["quantization_level"]):
        """Thompson Sampling：随机采样函数值"""
        y_quants = list(y_pred_dist.keys())
        probs = list(y_pred_dist.values())
        if not y_quants:
            return 0
        return np.random.choice(y_quants, p=probs)

    @classmethod
    def compute(cls, fn_name, y_pred_dist, y_best_quantized):
        """统一调用接口"""
        if fn_name == "EI":
            return cls.ei(y_pred_dist, y_best_quantized)
        elif fn_name == "PI":
            return cls.pi(y_pred_dist, y_best_quantized)
        elif fn_name == "UCB":
            return cls.ucb(y_pred_dist)
        elif fn_name == "TS":
            return cls.ts(y_pred_dist)
        else:
            raise ValueError(f"不支持的采集函数：{fn_name}")


# --------------------------
# 2. 超参数优化核心逻辑
# --------------------------
def optformer_hpo(config):
    # 1. 初始化随机种子
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    # 2. 加载对应模型的HPOBench基准任务
    model_type = config["model_type"].lower()
    logger.info(f"加载模型：{model_type}，任务ID：{config['task_id']}")
    rng = get_rng(config["random_seed"])

    # 选择基准任务
    if model_type == "xgboost":
        benchmark = XGBoostBenchmark(task_id=config["task_id"], rng=rng)
    elif model_type == "svm":
        benchmark = SVMBenchmark(task_id=config["task_id"], rng=rng)
    elif model_type == "nn":
        benchmark = NNBenchmark(task_id=config["task_id"], rng=rng)
    elif model_type == "lr":
        benchmark = LRBenchmark(task_id=config["task_id"], rng=rng)
    elif model_type == "rf":
        benchmark = RandomForestBenchmark(task_id=config["task_id"], rng=rng)
    elif model_type=="histgb":
        benchmark = HistGBBenchmark(task_id=config["task_id"], rng=rng)
    else:
        raise ValueError(f"不支持的模型类型：{model_type}，请选择 xgboost/svm/nn/lr/rf")

    # 3. 构建元数据
    search_space = benchmark.get_configuration_space()
    param_info = []
    for param in search_space.get_hyperparameters():
        p_dict = {
            "name": param.name,
            "type": param.__class__.__name__.upper(),
            "scale": getattr(param, "scale", "LINEAR")
        }
        if p_dict["type"] in ["DOUBLE", "INTEGER"]:
            p_dict["min"] = param.lower
            p_dict["max"] = param.upper
        elif p_dict["type"] == "CATEGORICAL":
            p_dict["categories"] = param.choices
        param_info.append(p_dict)

    metadata = {
        "name": f"{model_type}_task_{config['task_id']}",
        "metric": "accuracy",
        "goal": "maximize",
        "algorithm": "optformer",
        "parameters": param_info
    }

    # 4. 初始化组件
    serializer = OptFormerSerializer(quantization_level=config["quantization_level"])
    optformer = OptFormer(t5_model_name="google/flan-t5-small")
    optimizer = Adam(optformer.parameters(), lr=1e-5, weight_decay=1e-5)

    # 5. 初始化历史数据和最佳准确率
    logger.info("初始化随机超参数样本...")
    init_hparams = [search_space.sample_configuration().get_dictionary() for _ in range(ttttt)]
    history = []  # 存储(参数dict, 准确率)

    for hp in tqdm(init_hparams, desc="评估初始超参数"):
        try:
            # 根据模型类型设置不同的fidelity参数
            if model_type in ["xgboost", "rf","histgb"]:
                result = benchmark.objective_function(hp, rng=42,
                                                      fidelity={"n_estimators": 128, 'subsample': 0.1})
            elif model_type in ["nn", "lr"]:
                result = benchmark.objective_function(hp, rng=42,
                                                      fidelity={"iter": 100, 'subsample': 0.1})
            elif model_type == "svm":
                result = benchmark.objective_function(hp, rng=42,
                                                      fidelity={'subsample': 0.1})
            acc = result['val_scores']['acc']
            history.append((hp, acc))
        except Exception as e:
            logger.warning(f"评估超参数失败: {e}")

    if not history:
        raise ValueError("初始化超参数评估全部失败，请检查配置")

    # 初始化最佳准确率
    best_acc = max([h[1] for h in history])
    best_hp = [h[0] for h in history if h[1] == best_acc][0]
    logger.info(f"初始最佳准确率: {best_acc:.4f}")

    # 6. 存储每次迭代的最佳准确率（共50个元素）
    iter_best_accs = []

    # 7. 迭代优化
    logger.info("开始超参数优化迭代...")
    for iter_idx in range(config["n_iter"]):
        logger.info(f"\n=== 迭代 {iter_idx + 1}/{config['n_iter']} ===")
        iter_accuracies = []

        # 序列化当前状态（元数据+历史轨迹）
        meta_str = serializer.serialize_metadata(metadata)
        history_str, (y_min, y_max) = serializer.serialize_history(history, param_info)
        input_seq = f"{meta_str} <history> {history_str}"

        # 生成候选超参数
        candidates = []
        for _ in range(config["num_candidates"]):
            # 使用OptFormer生成下一个候选
            optformer.eval()
            with torch.no_grad():
                generated_seq = optformer(input_seq)

            # 解析生成的序列
            try:
                # 分割参数和函数值
                if "*" in generated_seq:
                    param_part, y_part = generated_seq.split("*", 1)
                    param_tokens = param_part.strip().split()
                    y_quant = int(y_part.strip().split("|")[0])

                    # 确保参数数量匹配
                    if len(param_tokens) == len(param_info):
                        param_dict = serializer.deserialize_param(param_tokens, param_info)
                        candidates.append((param_dict, y_quant))
            except Exception as e:
                logger.warning(f"解析生成序列失败: {e}，序列: {generated_seq}")
                continue

        # 如果生成的候选不足，补充随机候选
        if len(candidates) < config["samples_per_iter"]:
            need = config["samples_per_iter"] - len(candidates)
            logger.info(f"生成候选不足，补充 {need} 个随机候选")
            for _ in range(need):
                hp = search_space.sample_configuration().get_dictionary()
                candidates.append((hp, 0))  # 随机候选的y_quant设为0

        # 计算采集函数值并选择最佳候选
        y_best_quantized = serializer._normalize_quantize(best_acc, y_min, y_max)
        scored_candidates = []

        for param_dict, y_quant in candidates:
            # 简单的分布估计：假设生成的y_quant为均值，添加高斯噪声
            y_pred_dist = {y_quant: 0.8, y_quant + 1: 0.1, y_quant - 1: 0.1}
            score = AcquisitionFunction.compute(
                config["acquisition_fn"],
                y_pred_dist,
                y_best_quantized
            )
            scored_candidates.append((param_dict, score))

        # 按采集函数分数排序，选择前samples_per_iter个
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_candidates = [c[0] for c in scored_candidates[:config["samples_per_iter"]]]

        # 评估选中的候选超参数
        pbar = tqdm(selected_candidates, desc=f"迭代 {iter_idx + 1} 进度")
        for hp in pbar:
            try:
                # 根据模型类型设置不同的fidelity参数
                if model_type in ["xgboost", "rf","histgb"]:
                    result = benchmark.objective_function(hp, rng=42,
                                                          fidelity={"n_estimators": 128, 'subsample': 0.1})
                elif model_type in ["nn", "lr"]:
                    result = benchmark.objective_function(hp, rng=42,
                                                          fidelity={"iter": 100, 'subsample': 0.1})
                elif model_type == "svm":
                    result = benchmark.objective_function(hp, rng=42,
                                                          fidelity={'subsample': 0.1})
                current_acc = result['val_scores']['acc']
                iter_accuracies.append(current_acc)
                history.append((hp, current_acc))

                # 更新最佳准确率
                if current_acc > best_acc:
                    best_acc = current_acc
                    best_hp = hp

                pbar.set_postfix({"当前准确率": f"{current_acc:.4f}", "最佳准确率": f"{best_acc:.4f}"})
            except Exception as e:
                logger.warning(f"评估超参数失败: {e}")
                continue

        # 训练OptFormer（行为克隆）
        if len(history) > 1:
            optformer.train()
            optimizer.zero_grad()

            # 使用最后一个试验作为目标
            last_trial = history[-1]
            last_param, last_y = last_trial
            # 序列化最后一个试验作为目标序列
            _, (last_y_min, last_y_max) = serializer.serialize_history([last_trial], param_info)
            last_y_quant = serializer._normalize_quantize(last_y, last_y_min, last_y_max)

            # 构建目标序列
            param_tokens = []
            for p_info in param_info:
                p_val = last_param[p_info["name"]]
                if p_info["type"] in ["DOUBLE", "INTEGER"]:
                    quantized = serializer._normalize_quantize(p_val, p_info["min"], p_info["max"])
                    param_tokens.append(str(quantized))
                elif p_info["type"] == "CATEGORICAL":
                    param_tokens.append(str(p_info["categories"].index(p_val)))

            target_seq = " ".join(param_tokens) + f" * {last_y_quant} | "

            # 计算损失并反向传播
            loss = optformer(input_seq, target_seq)
            loss.backward()
            optimizer.step()
            logger.info(f"迭代 {iter_idx + 1} 训练损失: {loss.item():.4f}")

        # 记录当前迭代的最佳准确率
        iter_best_accs.append(best_acc)

        # 打印迭代统计
        if iter_accuracies:
            iter_avg = np.mean(iter_accuracies)
            iter_current_best = np.max(iter_accuracies)
            logger.info(
                f"迭代 {iter_idx + 1} 统计: 平均准确率 = {iter_avg:.4f}, 本迭代最佳 = {iter_current_best:.4f}, 历史最佳 = {best_acc:.4f}")
        else:
            logger.warning(f"迭代 {iter_idx + 1} 没有有效评估结果")

    # 输出最终结果
    logger.info("\n优化完成!")
    logger.info(f"模型类型: {model_type}")
    logger.info(f"最终最佳准确率: {best_acc:.4f}")
    logger.info(f"50次迭代的最佳准确率变化: {iter_best_accs}")

    return best_acc, iter_best_accs


if __name__ == "__main__":
    # 可以在这里修改模型类型
    # CONFIG["model_type"] = "svm"

    # 运行超参数优化
    final_best_accuracy, iteration_best_accuracies = optformer_hpo(CONFIG)

    # 打印最终结果
    print("\n===== 优化结果 =====")
    print(f"模型类型: {CONFIG['model_type']}")
    print(f"最终最佳准确率: {final_best_accuracy:.4f}")
    print("50次迭代的最佳准确率变化数组:")
    print(iteration_best_accuracies)
