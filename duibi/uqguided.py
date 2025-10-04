import numpy as np
from HPOBench.hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from ConfigSpace import ConfigurationSpace
from typing import List, Tuple

# ============ UQ工具函数 ============
def fit_loss_curve(history: List[float]):
    """简单拟合损失曲线，返回均值和方差"""
    epochs = np.arange(1, len(history) + 1)
    coeffs = np.polyfit(1/np.sqrt(epochs), history, 1)  # momentum 模型近似
    mu = coeffs[1]
    sigma = np.var(history[-min(5, len(history)):])  # 最近几轮方差
    return mu, sigma

def uq_compare(mu1, sigma1, mu2, sigma2):
    """计算 P( candidate1 更优于 candidate2 )"""
    from scipy.stats import norm
    z = (mu2 - mu1) / np.sqrt(sigma1 + sigma2 + 1e-8)
    return norm.cdf(z)

# ============ UQ-guided SH ============
def uq_guided_sh(bench, max_budget=50, eta=3, n_candidates=9):
    # 初始化
    cs = bench.get_configuration_space(seed=42)
    configs = [cs.sample_configuration() for _ in range(n_candidates)]

    budget = max_budget // int(np.log(n_candidates) / np.log(eta))
    history = {str(cfg): [] for cfg in configs}

    while len(configs) > 1 and budget <= max_budget:
        results = []
        for cfg in configs:
            res = bench.objective_function(configuration=cfg, fidelity={'n_estimators': budget})
            loss = res['function_value']
            history[str(cfg)].append(loss)

            mu, sigma = fit_loss_curve(history[str(cfg)])
            results.append((cfg, mu, sigma))

        # 排序并构造置信曲线
        results.sort(key=lambda x: x[1])  # 按均值排序
        keep = []
        for i in range(1, len(results) + 1):
            prob_best = np.mean([uq_compare(r[1], r[2], results[i-1][1], results[i-1][2]) for r in results])
            if prob_best > 0.6:  # 阈值，可调
                keep = results[:i]
                break

        configs = [r[0] for r in keep]
        budget *= eta

    # 最后训练到最大 budget
    best_cfg = configs[0]
    final_res = bench.objective_function(configuration=best_cfg, fidelity={'n_estimators': max_budget})
    return best_cfg, final_res['function_value']

# ============ 运行 ============
if __name__ == "__main__":
    bench = XGBoostBenchmark(task_id=167149)
    best_cfg, best_loss = uq_guided_sh(bench)
    print("Best config:", best_cfg)
    print("Final validation loss:", best_loss)
