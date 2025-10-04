import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
# 使用PyTorch原生Transformer替代HuggingFace的实现
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from HPOBench.hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from HPOBench.hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from HPOBench.hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
from HPOBench.hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from HPOBench.hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from hpobench.util.rng_helper import get_rng
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from collections import deque
import random
import logging
from tqdm import tqdm
from HPOBench.hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------
# 配置参数
# --------------------------
CONFIG = {
    "model_type": "xgboost",  # 可选: "xgboost", "svm", "nn", "lr", "rf"
    "task_id": 167097,  # HPOBench任务ID
    "n_iter": 50,  # 总迭代次数
    "samples_per_iter":10,  # 每个迭代采样的超参数数量
    "max_seq_len": 31,  # BOFormer序列长度
    "hidden_dim": 128,  # Transformer隐藏层维度
    "n_heads": 4,  # 注意力头数
    "n_layers": 8,  # Transformer编码器层数
    "lr": 1e-5,  # 学习率
    "weight_decay": 1e-5,  # 权重衰减
    "buffer_size": 1024,  # 轨迹回放缓冲大小
    "batch_size": 16,  # 训练批次大小
    "gamma": 0.95,  # 折扣因子
    "target_update_freq": 10,  # 目标网络更新频率
    "random_seed": 42  # 随机种子
}


# --------------------------
# BOFormer 模型定义
# --------------------------
class BOFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # 位置编码
        self.pos_encoding = nn.Embedding(max_seq_len, hidden_dim)

        # 输入嵌入层
        self.obs_emb = nn.Linear(input_dim, hidden_dim)

        # 使用PyTorch原生Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True  # 注意：PyTorch 1.10+支持batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Q值预测头 - 预测准确率
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, pos_ids):
        """
        x_seq: 序列输入 [batch_size, seq_len, input_dim]
        pos_ids: 位置ID [batch_size, seq_len]
        """
        # 输入嵌入 + 位置编码
        emb = self.obs_emb(x_seq)  # [B, L, H]
        pos_emb = self.pos_encoding(pos_ids)  # [B, L, H]
        x = emb + pos_emb  # [B, L, H]

        # Transformer编码
        x = self.transformer(x)  # [B, L, H]

        # 预测最后一步的Q值（准确率相关）
        q_val = self.q_head(x[:, -1, :])  # [B, 1]
        return q_val


# --------------------------
# 优先轨迹回放缓冲
# --------------------------
class PrioritizedTrajectoryBuffer:
    def __init__(self, buffer_size, alpha=0.6):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha  # 优先级权重

    def add(self, trajectory, td_error):
        """添加轨迹和对应的优先级"""
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.buffer.append(trajectory)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        """基于优先级采样批次轨迹"""
        if len(self.buffer) < batch_size:
            return None, None, None  # 缓冲区数据不足

        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # 重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        """更新轨迹优先级"""
        for i, idx in enumerate(indices):
            self.priorities[idx] = (abs(td_errors[i]) + 1e-6) ** self.alpha


# --------------------------
# 超参数优化核心逻辑
# --------------------------
def boformer_hpo(config):
    # 设置随机种子
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    # 根据选择的模型类型初始化对应的HPOBench基准任务
    model_type = config["model_type"].lower()
    logger.info(f"选择的模型类型: {model_type}，任务ID: {config['task_id']}")
    rng = get_rng(config["random_seed"])

    # 选择对应的benchmark
    if model_type == "xgboost":
        benchmark = XGBoostBenchmark(task_id=config["task_id"])
    elif model_type == "svm":
        benchmark = SVMBenchmark(task_id=config["task_id"])
    elif model_type == "nn":
        benchmark = NNBenchmark(task_id=config["task_id"])
    elif model_type == "lr":
        benchmark = LRBenchmark(task_id=config["task_id"])
    elif model_type == "rf":
        benchmark = RandomForestBenchmark(task_id=config["task_id"])
    elif model_type =="histgb":
        benchmark= HistGBBenchmark(task_id=config["task_id"])
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，请选择 'xgboost', 'svm', 'nn', 'lr' 或 'rf'")

    # 获取超参数搜索空间
    search_space = benchmark.get_configuration_space()
    n_hparams = len(search_space.get_hyperparameters())
    logger.info(f"超参数维度: {n_hparams}")

    # 初始化GP代理模型 - 建模准确率
    def init_gp_model(X, y):
        """初始化并训练GP模型，建模准确率"""
        if len(X) < 2:  # 至少需要2个样本才能训练GP
            return None

        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[1]))
        # 直接使用准确率作为目标值
        gp = SingleTaskGP(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        return gp

    # 初始化BOFormer和训练组件
    input_dim = n_hparams + 2 + 1 + 1 + 1  # 超参数 + GP后验 + 最佳准确率 + 时间占比 + Q值
    model = BOFormer(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq_len=config["max_seq_len"]
    )
    target_model = BOFormer(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq_len=config["max_seq_len"]
    )
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    mse_loss = nn.MSELoss()

    # 优先轨迹缓冲
    buffer = PrioritizedTrajectoryBuffer(buffer_size=config["buffer_size"])

    # 初始化历史数据和最佳准确率
    logger.info("初始化随机超参数样本...")
    init_hparams = [search_space.sample_configuration().get_dictionary() for _ in range(5)]
    init_X = np.array([list(hp.values()) for hp in init_hparams])
    init_y = []

    for hp in tqdm(init_hparams, desc="评估初始超参数"):
        # 不同模型可能需要不同的fidelity参数，这里做兼容处理
        if model_type in ["xgboost", "rf","histgb"]:
            result = benchmark.objective_function(hp, rng=42,
                                                  fidelity={"n_estimators": 128, 'subsample': 0.1})
        elif model_type in ["nn" , "lr"]:
            result = benchmark.objective_function(hp, rng=42, fidelity={"iter": 100, 'subsample': 0.1})
        elif model_type == "svm":
            result = benchmark.objective_function(hp, rng=42, fidelity={'subsample': 0.1})

        init_y.append(result['val_scores']['acc'])

    init_y = np.array(init_y).reshape(-1, 1)
    best_acc = np.max(init_y)  # 初始最佳准确率
    logger.info(f"初始最佳准确率: {best_acc:.4f}")

    # 初始化GP模型
    gp_model = init_gp_model(init_X, init_y)

    # 历史轨迹
    history = []
    for i in range(len(init_X)):
        if gp_model is not None:
            with torch.no_grad():
                posterior = gp_model(torch.tensor(init_X[i:i + 1], dtype=torch.float32))
                mu = posterior.mean.item()
                sigma = posterior.variance.sqrt().item()
        else:
            mu = 0.0
            sigma = 1.0

        obs = np.array([
            *init_X[i], mu, sigma, best_acc,
            i / (config["n_iter"] * config["samples_per_iter"]), 0.0
        ])
        history.append(obs)

    # 存储每次迭代的最佳准确率（共50个元素）
    iter_best_accs = []
    total_steps = 0

    # 迭代优化 - 每个迭代处理samples_per_iter个超参数
    logger.info("开始超参数优化迭代...")
    for iter_idx in range(config["n_iter"]):
        logger.info(f"\n=== 迭代 {iter_idx + 1}/{config['n_iter']} ===")
        iter_accuracies = []

        # 每个迭代采样指定数量的超参数并评估
        candidate_hparams = [search_space.sample_configuration().get_dictionary()
                             for _ in range(config["samples_per_iter"])]
        candidate_X = np.array([list(hp.values()) for hp in candidate_hparams])
        candidate_Q = []

        # 计算每个候选超参数的Q值
        model.eval()
        with torch.no_grad():
            for x in candidate_X:
                # 计算GP后验（均值+方差）
                if gp_model is not None:
                    x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
                    posterior = gp_model(x_tensor)
                    mu = posterior.mean.item()
                    sigma = posterior.variance.sqrt().item()
                else:
                    mu = 0.0
                    sigma = 1.0

                # 构建序列输入
                seq = history[-config["max_seq_len"]:]
                if len(seq) < config["max_seq_len"]:
                    pad = np.zeros((config["max_seq_len"] - len(seq), input_dim))
                    seq = np.concatenate([pad, seq], axis=0)

                current_step = iter_idx * config["samples_per_iter"] + len(candidate_Q)
                total_steps_possible = config["n_iter"] * config["samples_per_iter"]

                new_obs = np.array([
                    *x, mu, sigma, best_acc,
                    current_step / total_steps_possible, 0.0
                ])

                seq = np.concatenate([seq, [new_obs]], axis=0)[1:]

                # 预测Q值
                pos_ids = torch.arange(config["max_seq_len"]).reshape(1, -1)
                seq_tensor = torch.tensor(seq.reshape(1, config["max_seq_len"], input_dim), dtype=torch.float32)
                q_val = model(seq_tensor, pos_ids).item()
                candidate_Q.append(q_val)

        # 逐个评估所有候选超参数
        pbar = tqdm(range(config["samples_per_iter"]), desc=f"迭代 {iter_idx + 1} 进度")
        for run_idx in pbar:
            total_steps += 1

            selected_hp = candidate_hparams[run_idx]
            selected_x = candidate_X[run_idx]
            selected_q = candidate_Q[run_idx]

            # 评估超参数，获取准确率，根据模型类型设置不同的fidelity
            if model_type in ["xgboost", "rf","histgb"]:
                result = benchmark.objective_function(selected_hp, rng=42,
                                                      fidelity={"n_estimators": 128, 'subsample': 0.1})
            elif model_type in ["nn", "lr"]:
                result = benchmark.objective_function(selected_hp, rng=42, fidelity={"iter": 100, 'subsample': 0.1})
            elif model_type == "svm":
                result = benchmark.objective_function(selected_hp, rng=42,fidelity={'subsample': 0.1})

            current_acc = result['val_scores']['acc']
            iter_accuracies.append(current_acc)

            # 更新最佳准确率（只升不降）
            if current_acc > best_acc:
                best_acc = current_acc
                best_hp = selected_hp

            pbar.set_postfix({"当前准确率": f"{current_acc:.4f}", "最佳准确率": f"{best_acc:.4f}"})

            # 更新历史和GP模型
            if gp_model is not None:
                with torch.no_grad():
                    x_tensor = torch.tensor(selected_x.reshape(1, -1), dtype=torch.float32)
                    posterior = gp_model(x_tensor)
                    mu = posterior.mean.item()
                    sigma = posterior.variance.sqrt().item()
            else:
                mu = 0.0
                sigma = 1.0

            # 构建新的观测向量
            current_step = iter_idx * config["samples_per_iter"] + run_idx
            new_obs = np.array([
                *selected_x, mu, sigma, best_acc,
                current_step / total_steps_possible, selected_q
            ])
            history.append(new_obs)

            # 更新GP模型
            new_X = np.concatenate([init_X, selected_x.reshape(1, -1)], axis=0)
            new_y = np.concatenate([init_y, np.array([current_acc]).reshape(1, -1)], axis=0)
            gp_model = init_gp_model(new_X, new_y)
            init_X, init_y = new_X, new_y

            # 收集轨迹并训练BOFormer
            if len(history) >= config["max_seq_len"]:
                trajectory = history[-config["max_seq_len"]:]

                # 计算TD误差
                model.train()
                with torch.no_grad():
                    # 准备输入
                    trajectory_tensor = torch.tensor(
                        trajectory, dtype=torch.float32
                    ).reshape(1, config["max_seq_len"], input_dim)
                    pos_ids = torch.arange(config["max_seq_len"]).reshape(1, -1)

                    # 预测Q值
                    q_pred = model(trajectory_tensor, pos_ids)

                    # 计算目标Q值
                    next_obs = trajectory[1:]
                    if len(next_obs) < config["max_seq_len"]:
                        pad = torch.zeros(1, config["max_seq_len"] - len(next_obs), input_dim)
                        next_obs_tensor = torch.cat([
                            pad,
                            torch.tensor(next_obs, dtype=torch.float32).reshape(1, -1, input_dim)
                        ], dim=1)
                    else:
                        next_obs_tensor = torch.tensor(
                            next_obs, dtype=torch.float32
                        ).reshape(1, config["max_seq_len"], input_dim)

                    target_q = current_acc + config["gamma"] * target_model(next_obs_tensor, pos_ids).item()
                    td_error = target_q - q_pred.item()

                # 添加到缓冲
                buffer.add(trajectory, td_error)

                # 从缓冲采样并训练
                batch, indices, weights = buffer.sample(config["batch_size"])
                if batch is not None and len(batch) > 0:
                    optimizer.zero_grad()

                    losses = []
                    new_td_errors = []

                    for i, traj in enumerate(batch):
                        traj_tensor = torch.tensor(
                            traj, dtype=torch.float32
                        ).reshape(1, config["max_seq_len"], input_dim)

                        q_pred = model(traj_tensor, pos_ids)

                        next_traj = traj[1:]
                        if len(next_traj) < config["max_seq_len"]:
                            pad = torch.zeros(1, config["max_seq_len"] - len(next_traj), input_dim)
                            next_traj_tensor = torch.cat([
                                pad,
                                torch.tensor(next_traj, dtype=torch.float32).reshape(1, -1, input_dim)
                            ], dim=1)
                        else:
                            next_traj_tensor = torch.tensor(
                                next_traj, dtype=torch.float32
                            ).reshape(1, config["max_seq_len"], input_dim)

                        reward = traj[-1][n_hparams + 2]
                        target_q_batch = reward + config["gamma"] * target_model(next_traj_tensor, pos_ids).item()

                        loss = mse_loss(q_pred, torch.tensor([[target_q_batch]], dtype=torch.float32))
                        losses.append(loss * weights[i])
                        new_td_errors.append(target_q_batch - q_pred.item())

                    total_loss = torch.mean(torch.stack(losses))
                    total_loss.backward()
                    optimizer.step()
                    buffer.update_priorities(indices, new_td_errors)

            # 更新目标网络
            if total_steps % config["target_update_freq"] == 0:
                target_model.load_state_dict(model.state_dict())

        # 记录当前迭代的最佳准确率（保持历史最高）
        iter_best_accs.append(best_acc)

        # 打印迭代统计
        iter_avg = np.mean(iter_accuracies)
        iter_current_best = np.max(iter_accuracies)
        logger.info(
            f"迭代 {iter_idx + 1} 统计: 平均准确率 = {iter_avg:.4f}, 本迭代最佳 = {iter_current_best:.4f}, 历史最佳 = {best_acc:.4f}")

    # 输出最终结果
    logger.info("\n优化完成!")
    logger.info(f"模型类型: {model_type}")
    logger.info(f"最终最佳准确率: {best_acc:.4f}")
    logger.info(f"50次迭代的最佳准确率变化: {iter_best_accs}")

    return best_acc, iter_best_accs


if __name__ == "__main__":
    # 可以在这里修改模型类型，可选: "xgboost", "svm", "nn", "lr", "rf"
    # CONFIG["model_type"] = "svm"  # 示例：切换到SVM模型

    # 运行超参数优化
    final_best_accuracy, iteration_best_accuracies = boformer_hpo(CONFIG)

    # 打印最终结果
    print("\n===== 优化结果 =====")
    print(f"模型类型: {CONFIG['model_type']}")
    print(f"最终最佳准确率: {final_best_accuracy:.4f}")
    print("50次迭代的最佳准确率变化数组:")
    print(iteration_best_accuracies)
