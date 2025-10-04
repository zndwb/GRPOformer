import numpy as np
import json
import os
from typing import Dict, List, Any
from HPOBench.hpobench.abstract_benchmark import AbstractBenchmark
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    OrdinalHyperparameter
)
from config import MAX_TRAJECTORY_LENGTH, RESULTS_DIR,frist_config_xgboost,frist_config_nn
import config as con
from ConfigSpace import Configuration


class TrajectoryManager:
    """管理超参数优化轨迹的创建、更新和保存"""

    def __init__(self, benchmark: AbstractBenchmark):
        self.benchmark = benchmark
        self.config_space = self.benchmark.get_configuration_space(con.SEED)
        self.meta_information = self.benchmark.get_meta_information()  # 任务元信息
        self.task_description = self._create_task_description()  # 超参数取值范围
        self.trajectory = self._initialize_trajectory()

        # 创建结果保存目录
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def _create_task_description(self) -> Dict:
        """创建任务描述，包含超参数空间取值范围（使用get_configuration_space）"""
        param_info = {}
        for param in self.config_space.get_hyperparameters():
            if isinstance(param, UniformFloatHyperparameter):
                param_info[param.name] = {
                    "type": "continuous",
                    "lower": param.lower,
                    "upper": param.upper,
                    "log": param.log
                }
            elif isinstance(param, UniformIntegerHyperparameter):
                param_info[param.name] = {
                    "type": "integer",
                    "lower": param.lower,
                    "upper": param.upper
                }
            elif isinstance(param, (CategoricalHyperparameter, OrdinalHyperparameter)):
                param_info[param.name] = {
                    "type": "categorical",
                    "choices": param.choices
                }

        return {
            "configuration_space": param_info,
            "objective": "maximize"
        }

    def _initialize_trajectory(self) -> Dict:
        """初始化轨迹，使用超参数中间值作为起点"""
        initial_config = self._get_initial_config().get_dictionary()
        if con.model == "nn":
            val_result = self.benchmark.objective_function(initial_config, rng=42,
                                                           fidelity={"iter": con.iter, 'subsample': con.nn_subsample})
            # test_result = self.benchmark.objective_function_test(initial_config, rng=1)
        elif con.model == "xgb":
            val_result = self.benchmark.objective_function(initial_config, rng=42,
                                                           fidelity={"n_estimators": con.n_estimators,
                                                                     'subsample': con.xgb_subsample})
        elif con.model=="svm":
            val_result = self.benchmark.objective_function(initial_config, rng=42,
                                                           fidelity={'subsample': con.svm_subsample})
            # test_result = self.benchmark.objective_function_test(initial_config, rng=1)
        elif con.model=="lr":
            val_result = self.benchmark.objective_function(initial_config, rng=42,
                                                       fidelity={'iter': con.iter_lr, 'subsample': con.lr_subsample})
        elif con.model=="rf":
            val_result = self.benchmark.objective_function(initial_config, rng=42,
                                                           fidelity={"n_estimators": con.n_estimators_rf,'subsample': con.rf_subsample})
        elif con.model=="histab":
            val_result = self.benchmark.objective_function(initial_config, rng=42,
                                                           fidelity={"n_estimators": con.n_estimators_histgb,'subsample': con.histgb_subsample})
        # 整合结果，明确区分验证集和测试集
        initial_result = {
            "performance": val_result,
            # "test": test_result
        }
        # task=self.task_description
        # meta=self.meta_information
        return {
            "task_description": self.task_description,  # 超参数取值范围
            "meta_information": self.meta_information,  # 任务元信息
            "configuration": [initial_config],  # 超参数轨迹
            "result": [initial_result]  # 结果轨迹（包含验证和测试集）
        }

    def _get_initial_config(self) -> Dict:
        # return self.config_space.get_default_configuration()
        if con.model=="nn":
            config = con.frist_config_nn
        elif con.model=="xgb":
            config=con.frist_config_xgboost
        elif con.model=="svm":
            config=con.frist_config_svm
        elif con.model=="lr":
            config=con.frist_config_lr
        elif con.model=="rf":
            config=con.frist_config_rf
        # elif con.model=="rf":

        config_configuration=Configuration(self.config_space, values=config)
        return config_configuration


    def update_trajectory(self, new_config: Dict) -> None:
        """
        更新轨迹，添加新的配置和结果
        自动评估新配置，同时获取验证集和测试集结果
        """
        corrected_config = new_config.copy()

        for param_name, value in new_config.items():
            spec = self.task_description["configuration_space"][param_name]
            if spec["type"] in ["continuous", "integer"]:
                lower = spec["lower"]
                upper = spec["upper"]
                if value < lower:
                    # print(f"[参数修正] {param_name}: {value} < {lower}，已修正为 {lower}")
                    corrected_config[param_name] = lower
                elif value > upper:
                    # print(f"[参数修正] {param_name}: {value} > {upper}，已修正为 {upper}")
                    corrected_config[param_name] = upper
            # categorical 类型无范围限制，无需处理

        config = Configuration(self.config_space, values=corrected_config)
        if con.model=="nn":
            val_result = self.benchmark.objective_function(config,rng=42, fidelity={"iter": con.iter,'subsample':con.nn_subsample})
            # test_result = self.benchmark.objective_function_test(config, rng=1)
        elif con.model=="xgb":
            val_result = self.benchmark.objective_function(config, rng=42, fidelity={"n_estimators": con.n_estimators, 'subsample': con.xgb_subsample})
            # test_result = self.benchmark.objective_function_test(config, rng=1)
        elif con.model=="svm":
            val_result = self.benchmark.objective_function(config, rng=42,fidelity={'subsample': con.svm_subsample})
        elif con.model=="lr":
            val_result = self.benchmark.objective_function(config, rng=42, fidelity={'iter': con.iter_lr, 'subsample': con.lr_subsample})
        elif con.model=="rf":
            val_result = self.benchmark.objective_function(config, rng=42,
                                                           fidelity={"n_estimators": con.n_estimators_rf,'subsample': con.rf_subsample})
        elif con.model=="rf":
            val_result = self.benchmark.objective_function(config, rng=42,
                                                           fidelity={"n_estimators": con.n_estimators_histgb,'subsample': con.histgb_subsample})
        # 整合结果，明确区分验证集和测试集
        new_result = {
            "performance": val_result,
            # "validation": val_result,
            # "test": test_result
        }
        # 更新轨迹
        if len(self.trajectory["configuration"]) < MAX_TRAJECTORY_LENGTH:
            self.trajectory["configuration"].append(new_config)
            self.trajectory["result"].append(new_result)
        else:
            # 保持轨迹长度，移除最旧的条目
            self.trajectory["configuration"].pop(0)
            self.trajectory["result"].pop(0)
            self.trajectory["configuration"].append(new_config)
            self.trajectory["result"].append(new_result)

    def get_current_state(self) -> Dict:
        """获取当前轨迹状态"""
        return self.trajectory

    def save_trajectory(self, episode: int) -> None:
        """保存轨迹到文件"""
        save_path = os.path.join(RESULTS_DIR, f"trajectory_episode_{episode}.json")
        filtered_trajectory = {
            "configuration": self.trajectory["configuration"],
            "result": self.trajectory["result"]
        }
        with open(save_path, 'w') as f:
            json.dump(filtered_trajectory, f, indent=2, default=self._convert_to_native)
        print(f"轨迹已保存至: {save_path}")

    @staticmethod
    def _convert_to_native(obj: Any) -> Any:
        """将numpy类型转换为Python原生类型，确保JSON可序列化"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: TrajectoryManager._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [TrajectoryManager._convert_to_native(v) for v in obj]
        else:
            return obj

    def get_best_performance_acc(self) -> float:
        """获取当前轨迹中的最佳验证集性能"""
        if not self.trajectory["result"]:
            return np.inf
        return max([r["performance"]["val_scores"]["acc"] for r in self.trajectory["result"]])

    def get_best_test_performance_acc(self) -> float:
        """获取当前轨迹中的最佳测试集性能"""
        if not self.trajectory["result"]:
            return np.inf
        best_val_idx = np.argmin([r["performance"]["val_scores"]["acc"] for r in self.trajectory["result"]])
        return self.trajectory["result"][best_val_idx]["test"]["test_scores"]["acc"]

    def get_best_performance_bal_acc(self) -> float:
        """获取当前轨迹中的最佳验证集性能"""
        if not self.trajectory["result"]:
            return np.inf
        return max([r["performance"]["val_scores"]["bal_acc"] for r in self.trajectory["result"]])

    def get_best_test_performance_bal_acc(self) -> float:
        """获取当前轨迹中的最佳测试集性能"""
        if not self.trajectory["result"]:
            return np.inf
        best_val_idx = np.argmin([r["performance"]["val_scores"]["bal_acc"] for r in self.trajectory["result"]])
        return self.trajectory["result"][best_val_idx]["test"]["test_scores"]["bal_acc"]

    def get_best_performance_f1(self) -> float:
        """获取当前轨迹中的最佳验证集性能"""
        if not self.trajectory["result"]:
            return np.inf
        return max([r["performance"]["val_scores"]["f1"] for r in self.trajectory["result"]])

    def get_best_test_performance_f1(self) -> float:
        """获取当前轨迹中的最佳测试集性能"""
        if not self.trajectory["result"]:
            return np.inf
        best_val_idx = np.argmin([r["performance"]["val_scores"]["f1"] for r in self.trajectory["result"]])
        return self.trajectory["result"][best_val_idx]["test"]["test_scores"]["f1"]

    def get_best_performance_precision(self) -> float:
        """获取当前轨迹中的最佳验证集性能"""
        if not self.trajectory["result"]:
            return np.inf
        return max([r["performance"]["val_scores"]["precision"] for r in self.trajectory["result"]])

    def get_best_test_performance_precision(self) -> float:
        """获取当前轨迹中的最佳测试集性能"""
        if not self.trajectory["result"]:
            return np.inf
        best_val_idx = np.argmin([r["performance"]["val_scores"]["precision"] for r in self.trajectory["result"]])
        return self.trajectory["result"][best_val_idx]["test"]["test_scores"]["precision"]

    def get_best_configuration(self) -> Dict:
        """获取当前轨迹中的最佳配置（基于验证集）"""
        if not self.trajectory["result"]:
            return {}
        best_idx = np.argmin([r["performance"]["val_scores"]["acc"] for r in self.trajectory["result"]])
        return self.trajectory["configuration"][best_idx]
