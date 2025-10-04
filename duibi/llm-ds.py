import os
import re
import json
from zhipuai import ZhipuAI
# 导入所有需要的基准类
from HPOBench.hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from HPOBench.hpobench.benchmarks.ml.nn_benchmark import NNBenchmark, NNBenchmarkBB
from HPOBench.hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
from HPOBench.hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from HPOBench.hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from HPOBench.hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark
from ConfigSpace import Configuration

# 使用 ZHIPUAI_KEY 初始化客户端
client = ZhipuAI(api_key="")


class GLM4HPOptimizer:
    def __init__(self, task_id, model_type, max_trials=5, trainings_per_trial=5):
        """
        初始化优化器
        :param task_id: HPOBench 任务ID
        :param model_type: 模型类型，可选值："xgboost", "nn", "lr", "rf", "svm"
        :param max_trials: 总实验次数
        :param trainings_per_trial: 每个实验的训练次数
        """
        self.task_id = task_id
        self.max_trials = max_trials
        self.trainings_per_trial = trainings_per_trial
        self.model_type = model_type  # 新增：模型类型变量
        # 根据模型类型选择对应的基准类
        self.benchmark = self._get_benchmark_by_model_type()
        self.history = []
        self.best_acc_curve = []
        self.overall_best_acc = -float('inf')
        # 动态获取当前模型的配置空间
        self.config_space = self.benchmark.get_configuration_space()
        self.param_ranges = self._parse_config_space()
        self.task_description = self._get_task_description()

    def _get_benchmark_by_model_type(self):
        """根据 model_type 选择对应的 HPOBench 基准类"""
        model_benchmark_map = {
            "xgboost": XGBoostBenchmark,
            "nn": NNBenchmark,  # 若需使用 NNBenchmarkBB，可将此处改为 NNBenchmarkBB
            "lr": LRBenchmark,
            "rf": RandomForestBenchmark,
            "svm": SVMBenchmark,
            "histgb": HistGBBenchmark
        }
        # 检查模型类型是否合法
        if self.model_type not in model_benchmark_map:
            raise ValueError(
                f"不支持的模型类型：{self.model_type}，可选类型为：{list(model_benchmark_map.keys())}"
            )
        # 返回对应基准类的实例（传入 task_id）
        return model_benchmark_map[self.model_type](task_id=self.task_id)

    def _parse_config_space(self):
        """解析当前模型的超参数搜索范围（适配所有模型）"""
        ranges = {}
        for param in self.config_space.get_hyperparameters():
            # 不同模型的超参数类型可能不同，统一处理数值型参数（HPOBench 主流模型均支持）
            if hasattr(param, 'lower') and hasattr(param, 'upper'):
                ranges[param.name] = (param.lower, param.upper)
            # 若模型有非数值参数（如SVM的kernel），可在此处扩展处理逻辑
        return ranges

    def _get_task_description(self):
        """获取任务描述（适配所有模型，补充模型类型信息）"""
        meta = self.benchmark.get_meta_information()
        dataset_name = meta.get("dataset_name", "unknown dataset")
        task_type = "classification task"  # HPOBench 上述模型均以分类任务为主
        n_samples = meta.get("n_samples", "unknown number of")
        n_features = meta.get("n_features", "unknown number of")

        # 任务描述中补充当前模型类型，让GLM-4生成更适配的超参数
        return (f"This task uses {self.model_type.upper()} model on the {dataset_name} dataset, "
                f"which is a {task_type}. It contains {n_samples} samples and {n_features} features. "
                f"Generate optimized {self.model_type.upper()} hyperparameters for this specific task.")

    def _generate_hyperparameters(self):
        """生成当前模型的超参数配置（Prompt 动态包含模型类型）"""
        prompt_lines = [
            f"Generate optimized {self.model_type.upper()} hyperparameter configurations for the following task:",
            self.task_description,
            f"\n{self.model_type.upper()} Hyperparameter ranges:",
        ]

        # 动态添加当前模型的超参数范围
        for param, (min_val, max_val) in self.param_ranges.items():
            dtype = "integer" if isinstance(min_val, int) else "float"
            prompt_lines.append(f"- {param}: {min_val} to {max_val} ({dtype})")

        # 生成当前模型的超参数示例（避免GLM-4混淆模型参数）
        prompt_lines.append(f"\nReturn only a JSON object with {self.model_type.upper()} hyperparameter names and values. Example:")
        example = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            if isinstance(min_val, int):
                example[param] = (min_val + max_val) // 2
            else:
                example[param] = round((min_val + max_val) / 2, 3)
        prompt_lines.append(json.dumps(example))

        return self._call_glm4("\n".join(prompt_lines))

    def _call_glm4(self, prompt):
        """调用GLM-4模型（逻辑不变，适配动态Prompt）"""
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )

        raw_output = response.choices[0].message.content.strip()

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            # 提取原始输出中的JSON片段
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            print(f"⚠️ GLM-4 返回的{self.model_type.upper()}超参数不是合法 JSON，原始输出：", raw_output)
            return {}

    def _validate_hyperparameters(self, params):
        """验证超参数（适配所有模型的参数范围，包括枚举型和数值型）"""
        validated = {}
        for param, range_val in self.param_ranges.items():

            # 如果是枚举型（list）
            if isinstance(range_val, list):
                val = params.get(param, range_val[0])  # 默认取第一个
                # 如果给的值不在枚举里，强制修正
                if val not in range_val:
                    # 尝试字符串转换（避免 int/float 和 str 不匹配）
                    val_str = str(val)
                    if val_str in map(str, range_val):
                        # 找到对应的原始类型值
                        for r in range_val:
                            if str(r) == val_str:
                                val = r
                                break
                    else:
                        val = range_val[0]
                validated[param] = val

            # 如果是数值范围 (min_val, max_val)
            elif isinstance(range_val, tuple) and len(range_val) == 2:
                min_val, max_val = range_val
                if isinstance(min_val, list):
                    min_val=min_val[0]
                elif isinstance(max_val, list):
                    max_val=max_val[0]

                # 获取值（默认取中间）
                val = params.get(
                    param,
                    (min_val + max_val) / 2 if isinstance(min_val, float) else (min_val + max_val) // 2
                )
                if isinstance(val, list):
                    val=val[0]


                # 如果是字符串，尝试转成数值
                if isinstance(val, str):
                    if isinstance(min_val, int):
                        val = int(float(val))  # 先转 float 再转 int，避免 "3.0"
                    else:
                        val = float(val)

                # 约束到范围内
                val = max(min_val, min(val, max_val))

                # 整数参数强制转换
                if isinstance(min_val, int):
                    val = int(val)

                validated[param] = val

            else:
                raise ValueError(f"param_ranges[{param}] 格式不支持: {range_val}")

        return validated

    def run_optimization(self):
        """运行超参数优化（逻辑不变，输出补充模型类型）"""
        print(f"在任务ID {self.task_id} 上运行 {self.model_type.upper()} 超参数优化...")
        print(f"任务描述: {self.task_description}")
        print(f"总实验次数: {self.max_trials}, 每次实验训练次数: {self.trainings_per_trial}")

        for trial in range(self.max_trials):
            print(f"\n实验 {trial + 1}/{self.max_trials}")

            # 生成并验证当前模型的超参数
            params = self._generate_hyperparameters()
            params = self._validate_hyperparameters(params)
            # 用当前模型的配置空间创建合法配置
            config_p = Configuration(self.config_space, values=params)

            # 每个实验训练多次，取最佳结果（适配所有模型的 objective_function）
            trial_accuracies = []
            for training in range(self.trainings_per_trial):
                # 不同模型的 fidelity 参数可能不同，此处兼容主流设置（可根据模型扩展）
                if self.model_type in ["xgboost", "rf", "histgb"]:
                    fidelity={"n_estimators": 128, 'subsample': 0.1}
                elif self.model_type in ["nn", "lr"]:
                    fidelity={"iter": 100, 'subsample': 0.1}
                elif self.model_type == "svm":
                    fidelity={'subsample': 0.1}
                result = self.benchmark.objective_function(
                    config_p,
                    rng=42 + training,  # 每次训练用不同随机种子（避免结果重复）
                    fidelity=fidelity
                )
                # HPOBench 所有分类模型的验证准确率均存储在 'val_scores']['acc']
                accuracy = result['val_scores']['acc']
                trial_accuracies.append(accuracy)
                print(f"  训练 {training + 1}/{self.trainings_per_trial}: 准确率 {accuracy:.4f}")

            # 记录当前实验的最佳准确率
            best_trial_acc = max(trial_accuracies)
            print(f"当前实验最佳准确率: {best_trial_acc:.4f}")

            # 更新全局最佳准确率和曲线
            if best_trial_acc > self.overall_best_acc:
                self.overall_best_acc = best_trial_acc
            self.best_acc_curve.append(self.overall_best_acc)
            self.history.append({
                "trial": trial + 1,
                "model_type": self.model_type,
                "params": params,
                "best_trial_acc": best_trial_acc
            })

        # 输出最终结果（补充模型类型）
        if self.history:
            best_trial = max(self.history, key=lambda x: x["best_trial_acc"])
            print(f"\n{self.model_type.upper()} 最佳实验 - 实验 {best_trial['trial']}: 准确率 {best_trial['best_trial_acc']:.4f}")
            print(f"\n{self.model_type.upper()} 最佳准确率变化曲线（共{len(self.best_acc_curve)}个元素）:")
            print(self.best_acc_curve)
            return self.best_acc_curve
        return None


if __name__ == "__main__":
    # --------------------------
    # 核心：通过修改 model_type 选择模型
    # 可选值："xgboost", "nn", "lr", "rf", "svm"
    # --------------------------
    TARGET_MODEL = "rf"  # 这里修改为你需要的模型类型

    # 初始化优化器（传入模型类型）
    optimizer = GLM4HPOptimizer(
        task_id=167155,
        model_type=TARGET_MODEL,  # 指定模型类型
        max_trials=50,
        trainings_per_trial=20
    )
    # 运行优化并获取最佳准确率曲线
    best_acc_curve = optimizer.run_optimization()
    print(f"\n{TARGET_MODEL.upper()} 最终最佳准确率曲线: {best_acc_curve}")
    print(best_acc_curve[-1])