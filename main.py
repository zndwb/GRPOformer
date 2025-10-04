import numpy as np
import os
import torch
from HPOBench.hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from HPOBench.hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from HPOBench.hpobench.benchmarks.ml.nn_benchmark import NNBenchmark,NNBenchmarkBB
from trajectory_manager import TrajectoryManager
# from policy_network import TransformerPolicy
from t5_policy import TransformerPolicyT5 as TransformerPolicy
from HPOBench.hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from HPOBench.hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
from HPOBench.hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
# from HPOBench.hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark
# from grpo_algorithm import GRPO
from grpo_chain import GRPO
from typing import List
from config import (
    SEED,
    NUM_EPISODES,
    NUM_STEPS_PER_EPISODE,
    SAVE_INTERVAL,
    MODEL_SAVE_DIR,
    DEVICE,
    GROUP_SIZE,
    model
)
import config as con
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

# 设置随机种子
def set_seed(seed: int):
    """设置随机种子确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_best_val_accuracy(manager):
    best_val_acc = 0.0
    # 遍历轨迹中的所有result
    for traj_data in manager.trajectory["result"]:
        val_acc = traj_data.get("performance", {}).get("val_scores").get("acc")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return best_val_acc

def create_trajectory_managers(benchmark, num_managers: int) -> List[TrajectoryManager]:
    """创建多个轨迹管理器，形成群体"""
    return [TrajectoryManager(benchmark) for _ in range(num_managers)]

# 167149,167097,167155,167168,167079,167069
def main():
    # 设置随机种子
    set_seed(SEED)
    taskid=167097
    # 创建模型保存目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    if con.model=="nn":
        # 初始化基准测试
        benchmark = NNBenchmark(task_id=taskid)  # 使用XGBoost在OpenML任务上的基准测试
    elif con.model=="xgb":
        benchmark = XGBoostBenchmark(task_id=taskid)  # 使用XGBoost在OpenML任务上的基准测试
    elif con.model=="svm":
        benchmark = SVMBenchmark(task_id=taskid)
    elif con.model=="lr":
        benchmark=LRBenchmark(task_id=taskid)
    elif con.model=="rf":
        benchmark=RandomForestBenchmark(task_id=taskid)
    elif con.model=="histgb":
        benchmark=HistGBBenchmark(task_id=taskid)
    # 初始化多个轨迹管理器形成群体
    trajectory_managers = create_trajectory_managers(benchmark, GROUP_SIZE)
    frist_acc=trajectory_managers[0].trajectory["result"][0].get("performance")["val_scores"]["acc"]
    frist_bal_acc = trajectory_managers[0].trajectory["result"][0].get("performance")["val_scores"]["bal_acc"]
    frist_f1 = trajectory_managers[0].trajectory["result"][0].get("performance")["val_scores"]["f1"]
    frist_precision = trajectory_managers[0].trajectory["result"][0].get("performance")["val_scores"]["precision"]
    print(f"已创建包含 {GROUP_SIZE} 个成员的优化群体")

    # 初始化策略网络和GRPO算法
    # 使用第一个轨迹管理器的配置空间
    policy = TransformerPolicy(trajectory_managers[0].task_description["configuration_space"])
    grpo = GRPO(policy)
    # 每个成员的验证集最优准确率历史
    val_acc_history_per_member = [[frist_acc] for _ in range(GROUP_SIZE)]
    val_bal_acc_history_per_member = [[frist_bal_acc] for _ in range(GROUP_SIZE)]
    val_f1_history_per_member = [[frist_f1] for _ in range(GROUP_SIZE)]
    val_precision_history_per_member = [[frist_precision] for _ in range(GROUP_SIZE)]
    # 每个 episode 的全局最佳验证准确率（从4个成员中取最大）
    overall_best_val_acc_history = []
    overall_best_val_f1_history = []
    overall_best_val_bal_acc_history = []
    overall_best_val_precision_history = []
    overall_best_val_acc_history.append(frist_acc)
    overall_best_val_bal_acc_history.append(frist_bal_acc)
    overall_best_val_f1_history.append(frist_f1)
    overall_best_val_precision_history.append(frist_precision)
    total_loss_history = []
    policy_loss_history = []
    entropy_history = []

    # 主训练循环
    for episode in range(NUM_EPISODES):
        print(f"\n===== 回合 {episode + 1}/{NUM_EPISODES} =====")

        # 收集所有群体成员的轨迹数据
        all_trajectories = []

        # 每个群体成员执行一定步数
        for manager_idx, manager in enumerate(trajectory_managers):
            print(f"  群体成员 {manager_idx + 1}/{GROUP_SIZE} 执行优化步骤")
            member_trajectories = []
            old_config_lb=[]

            for step in range(NUM_STEPS_PER_EPISODE):
                best_acc = get_best_val_accuracy(manager)
                # 获取当前轨迹状态
                current_trajectory = manager.get_current_state()

                # 旧策略采样（用于计算比率）
                with torch.no_grad():
                    old_config, old_log_prob, _ = policy(current_trajectory)

                # 评估旧配置并更新轨迹
                manager.update_trajectory(old_config)

                # 获取最新结果
                latest_result = manager.get_current_state()["result"][-1]
                val_performance = latest_result["performance"]["val_scores"]["acc"]
                # test_performance = latest_result["test"]["test_scores"]["acc"]
                val_f1=latest_result["performance"]["val_scores"]["f1"]
                val_bal_acc=latest_result["performance"]["val_scores"]["bal_acc"]
                val_precision = latest_result["performance"]["val_scores"]["precision"]

                if step % 2 == 0:  # 每两步打印一次性能
                    print(f"    步骤 {step + 1}: 验证集acc={val_performance:.4f}")

                # 计算奖励（使用验证集结果）
                reward = np.clip((val_performance - best_acc) / (1 - best_acc + 1e-8), -1, 1)

                # 新策略采样（用于训练）
                _, new_log_prob, _ = policy(current_trajectory)

                # 保存轨迹数据
                member_trajectories.append((
                    current_trajectory,
                    old_config,
                    new_log_prob,
                    old_log_prob,
                    reward,
                    (step == NUM_STEPS_PER_EPISODE - 1)  # 最后一步标记为结束
                ))

            all_trajectories.extend(member_trajectories)
            # 保存该成员的轨迹
            # manager.save_trajectory(f"{episode + 1}_member_{manager_idx + 1}")

        # 将所有轨迹分成小的群体用于GRPO更新
        # 随机打乱轨迹
        np.random.shuffle(all_trajectories)
        # 分成多个小群体
        groups = [all_trajectories[i:i + GROUP_SIZE] for i in range(0, len(all_trajectories), GROUP_SIZE)]

        # 使用GRPO更新策略网络
        metrics = grpo.update(groups)
        print(f"策略更新完成 - 总损失: {metrics['total_loss']:.4f}, "
              f"策略损失: {metrics['policy_loss']:.4f}, "
              f"熵: {metrics['entropy']:.4f}")
        total_loss_history.append(metrics['total_loss'])
        policy_loss_history.append(metrics['policy_loss'])
        entropy_history.append(metrics['entropy'])

        # 打印每个群体成员的最佳性能
        for manager_idx, manager in enumerate(trajectory_managers):
            best_val_perf = manager.get_best_performance_acc()
            # best_test_perf = manager.get_best_test_performance_acc()
            best_val_bal_acc = manager.get_best_performance_bal_acc()
            best_val_f1 = manager.get_best_performance_f1()
            best_val_precision = manager.get_best_performance_precision()

            # 记录每个成员在当前 episode 的 best val accuracy
            val_acc_history_per_member[manager_idx].append(best_val_perf)
            val_bal_acc_history_per_member[manager_idx].append(best_val_bal_acc)
            val_f1_history_per_member[manager_idx].append(best_val_f1)
            val_precision_history_per_member[manager_idx].append(best_val_precision)
            print(f"  群体成员 {manager_idx + 1} 最佳性能 - 验证集: {best_val_perf:.4f}")
        # 记录本回合的全局最佳验证准确率
        episode_best_val_acc = max([val_acc_history_per_member[i][-1] for i in range(GROUP_SIZE)])
        episode_best_val_bal_acc = max([val_bal_acc_history_per_member[i][-1] for i in range(GROUP_SIZE)])
        episode_best_val_f1 = max([val_f1_history_per_member[i][-1] for i in range(GROUP_SIZE)])
        episode_best_val_precision = max([val_precision_history_per_member[i][-1] for i in range(GROUP_SIZE)])
        overall_best_val_acc_history.append(episode_best_val_acc)
        overall_best_val_bal_acc_history.append(episode_best_val_bal_acc)
        overall_best_val_f1_history.append(episode_best_val_f1)
        overall_best_val_precision_history.append(episode_best_val_precision)

        # 定期保存模型
        if (episode + 1) % SAVE_INTERVAL == 0:
            model_save_path = os.path.join(MODEL_SAVE_DIR, f"policy_episode_{episode + 1}.pt")
            policy.save_model(model_save_path)
            print(f"模型已保存至: {model_save_path}")
    print("total_loss_history:",total_loss_history)
    print("policy_loss_history:",policy_loss_history)
    print("entropy_history",entropy_history)
    # 保存最终模型
    final_model_path = os.path.join(MODEL_SAVE_DIR, "policy_final.pt")
    policy.save_model(final_model_path)
    print(f"\n最终模型已保存至: {final_model_path}")
    print("训练完成！")

    print("overall_best_val_acc_history: ", overall_best_val_acc_history)
    print("overall_best_val_bal_acc_history: ", overall_best_val_bal_acc_history)
    print("overall_best_val_f1_history: ", overall_best_val_f1_history)
    print("overall_best_val_precision_history: ", overall_best_val_precision_history)
    # ===== 绘图展示每个群体成员的验证集最优准确率变化 =====
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Best accuracy in each member", fontsize=16)

    for i in range(GROUP_SIZE):
        ax = axs[i // 2][i % 2]
        ax.plot(range(0, NUM_EPISODES + 1), val_acc_history_per_member[i], label=f'member {i + 1}')
        ax.set_title(f'member {i + 1}')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Best Val Accuracy")
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "val_accuracy_per_member.png"))
    plt.show()

    # ===== 绘图：每个episode的全局最佳验证准确率 =====
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, NUM_EPISODES + 1), overall_best_val_acc_history,  color='green',
             label='Best accuracy')
    plt.title("Best accuracy in all members", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Best Val Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "best_val_accuracy_overall.png"))
    plt.show()

    episodes = list(range(1, NUM_EPISODES + 1))

    # --- 1. Total Loss ---
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, total_loss_history, label='Total Loss')
    plt.xlabel('Episode')
    plt.ylabel('Total Loss')
    plt.title('GRPO Total Loss over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 2. Policy Loss ---
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, policy_loss_history, label='Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Policy Loss')
    plt.title('GRPO Policy Loss over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 3. Entropy ---
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, entropy_history, label='Entropy')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.title('GRPO Entropy over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
