
from HPOBench.hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from HPOBench.hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from HPOBench.hpobench.benchmarks.ml.svm_benchmark_old import SupportVectorMachine
from HPOBench.hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from HPOBench.hpobench.benchmarks.ml.nn_benchmark import NNBenchmarkBB,NNBenchmark
from HPOBench.hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark
from HPOBench.hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
from HPOBench.hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from HPOBench.hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark
# from HPOBench.hpobench.benchmarks.ml.pybnn import BayesianNeuralNetworkBenchmark
from HPOBench.hpobench.benchmarks.nas.nasbench_1shot1 import NASBench1shot1SearchSpace1Benchmark
model_type="histgb"
taskid=167097
#211693 211694 167069
if model_type == "xgboost":
    b = XGBoostBenchmark(task_id=taskid)
elif model_type == "svm":
    b = SVMBenchmark(task_id=taskid)
elif model_type == "nn":
    b = NNBenchmark(task_id=taskid)
elif model_type == "lr":
    b = LRBenchmark(task_id=taskid)
elif model_type == "rf":
    b = RandomForestBenchmark(task_id=taskid)
elif model_type=="histgb":
    b = HistGBBenchmark(task_id=taskid)
elif model_type=="pybnn":
    b = NASBench1shot1SearchSpace1Benchmark(task_id=taskid)
# b=SupportVectorMachine(task_id=167149)
# b = SupportVectorMachine(task_id=31)
config = b.get_configuration_space(seed=1).sample_configuration()
if model_type in ["xgboost", "rf"]:
    result_dict = b.objective_function(config, rng=42,
                                          fidelity={"n_estimators": 128, 'subsample': 0.1})
elif model_type in ["nn", "lr"]:
    result_dict = b.objective_function(config, rng=42, fidelity={"iter": 100, 'subsample': 0.1})
elif model_type == "svm":
    result_dict = b.objective_function(config, rng=42, fidelity={'subsample': 0.1})
elif model_type == "histgb":
    result_dict = b.objective_function(config, rng=42,
                                       fidelity={"n_estimators": 10, 'subsample': 0.5})
else:
    result_dict = b.objective_function(config, rng=42)
# result_test = b.objective_function_test(configuration=config, rng=42)
print(config)
print(result_dict)
# print(result_test)


