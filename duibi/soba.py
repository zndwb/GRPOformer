import numpy as np
from HPOBench.hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from HPOBench.hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from HPOBench.hpobench.benchmarks.ml.nn_benchmark import NNBenchmark,NNBenchmarkBB
from HPOBench.hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
from HPOBench.hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from HPOBench.hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark
class SBOAOptimizer:
    def __init__(self, param_names, lb, ub, types, pop_size=5, max_iter=50):
        self.param_names = param_names
        self.lb = lb
        self.ub = ub
        self.types = types
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = len(param_names)
        self.pop = self._initialize_population()
        self.fitness = np.zeros(pop_size)
        self.best_solution = None
        self.best_fitness = float("-inf")  # 最大化
        self.iteration_results = []

    def _initialize_population(self):
        pop = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            for j in range(self.dim):
                if self.types[j] is int:
                    pop[i, j] = np.random.randint(self.lb[j], self.ub[j] + 1)
                else:
                    pop[i, j] = self.lb[j] + np.random.rand() * (self.ub[j] - self.lb[j])
        return pop

    def _levy_flight(self, dim):
        eta = 1.5
        sigma = (np.math.gamma(1 + eta) * np.sin(np.pi * eta / 2) /
                 (np.math.gamma((1 + eta) / 2) * eta * 2 ** ((eta - 1) / 2))) ** (1 / eta)
        u = np.random.normal(0, sigma ** 2, dim)
        v = np.random.normal(0, 1, dim)
        return 0.01 * u / np.abs(v) ** (1 / eta)

    def optimize(self, objective_func):
        for t in range(self.max_iter):
            # 计算适应度
            for i in range(self.pop_size):
                params = {
                    self.param_names[j]: int(self.pop[i, j]) if self.types[j] is int else float(self.pop[i, j])
                    for j in range(self.dim)
                }
                self.fitness[i] = objective_func(params)

            # 更新全局最优
            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_solution = self.pop[best_idx].copy()

            # 记录迭代结果
            best_params = {
                self.param_names[j]: int(self.best_solution[j]) if self.types[j] is int else float(self.best_solution[j])
                for j in range(self.dim)
            }
            self.iteration_results.append({
                'iteration': t + 1,
                'best_fitness': self.best_fitness,
                'best_params': best_params
            })

            print(f"迭代 {t+1}/{self.max_iter} - 当前最佳 Accuracy: {self.best_fitness:.6f}")

            # ===== 捕蛇阶段 & 避敌阶段（保持不变） =====
            new_pop = self.pop.copy()
            for i in range(self.pop_size):
                if t < self.max_iter / 3:
                    r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                    rand = np.random.rand(self.dim)
                    new_pop[i] = self.pop[i] + (self.pop[r1] - self.pop[r2]) * rand
                elif t < 2 * self.max_iter / 3:
                    rb = np.random.randn(self.dim)
                    new_pop[i] = self.best_solution + np.exp((t / self.max_iter) ** 4) * (rb - 0.5) * (
                        self.best_solution - self.pop[i]
                    )
                else:
                    levy = self._levy_flight(self.dim)
                    factor = (1 - t / self.max_iter) ** (2 * t / self.max_iter)
                    new_pop[i] = self.best_solution + factor * self.pop[i] * levy

                new_pop[i] = np.clip(new_pop[i], self.lb, self.ub)
                for j in range(self.dim):
                    if self.types[j] is int:
                        new_pop[i, j] = np.round(new_pop[i, j])

            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    rb = np.random.randn(self.dim)
                    factor = (1 - t / self.max_iter) ** 2
                    new_pop[i] = self.best_solution + (2 * rb - 1) * factor * self.pop[i]
                else:
                    r = np.random.choice(self.pop_size)
                    r2 = np.random.randn(self.dim)
                    k = np.round(1 + np.random.rand())
                    new_pop[i] = self.pop[i] + r2 * (self.pop[r] - k * self.pop[i])

                new_pop[i] = np.clip(new_pop[i], self.lb, self.ub)
                for j in range(self.dim):
                    if self.types[j] is int:
                        new_pop[i, j] = np.round(new_pop[i, j])

            self.pop = new_pop

        return self.best_solution, self.best_fitness, self.iteration_results

model="xgboost"
def main(task_id):
    if model == "nn":
        # 初始化基准测试
        benchmark = NNBenchmark(task_id=task_id)  # 使用XGBoost在OpenML任务上的基准测试
    elif model == "xgboost":
        benchmark = XGBoostBenchmark(task_id=task_id)  # 使用XGBoost在OpenML任务上的基准测试
    elif model == "svm":
        benchmark = SVMBenchmark(task_id=task_id)
    elif model == "lr":
        benchmark = LRBenchmark(task_id=task_id)
    elif model == "rf":
        benchmark = RandomForestBenchmark(task_id=task_id)
    elif model == "histgb":
        benchmark = HistGBBenchmark(task_id=task_id)
    hyperparameters = benchmark.get_configuration_space().get_hyperparameters()

    param_names = [hp.name for hp in hyperparameters]
    lb = np.array([hp.lower for hp in hyperparameters], dtype=float)
    ub = np.array([hp.upper for hp in hyperparameters], dtype=float)
    types = [int if isinstance(hp, UniformIntegerHyperparameter) else float for hp in hyperparameters]

    def objective(params):
        if model in ["xgboost", "rf", "histgb"]:
            fidelity = {"n_estimators": 128, 'subsample': 0.1}
        elif model in ["nn", "lr"]:
            fidelity = {"iter": 100, 'subsample': 0.1}
        elif model == "svm":
            fidelity = {'subsample': 0.1}
        result = benchmark.objective_function(configuration=params, rng=42, fidelity=fidelity)
        return result['val_scores']['acc']

    sboa = SBOAOptimizer(param_names, lb, ub, types, pop_size=10, max_iter=50)
    best_solution, best_fitness, results = sboa.optimize(objective)

    # 只要 Accuracy 数组
    acc_list = [r['best_fitness'] for r in results]
    print("\nAccuracy 数组输出：")
    print(acc_list)


if __name__ == "__main__":
    task_id =167097
    main(task_id)
