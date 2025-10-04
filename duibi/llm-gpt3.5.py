import os
import json
from openai import OpenAI
from HPOBench.hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark

# Set API key (for local testing only, do NOT expose publicly)
client = OpenAI(
    api_key= ""
)

class GPT3HPOptimizer:
    def __init__(self, task_id, max_trials=5):
        self.task_id = task_id
        self.max_trials = max_trials
        self.benchmark = XGBoostBenchmark(task_id=task_id)
        self.history = []
        self.config_space = self.benchmark.get_configuration_space()
        self.param_ranges = self._parse_config_space()
        self.task_description = self._get_task_description()  # Task info in English

    def _parse_config_space(self):
        """Parse hyperparameter search ranges"""
        ranges = {}
        for param in self.config_space.get_hyperparameters():
            if hasattr(param, 'lower') and hasattr(param, 'upper'):
                ranges[param.name] = (param.lower, param.upper)
        return ranges

    def _get_task_description(self):
        """Get task metadata in English from benchmark"""
        meta = self.benchmark.get_meta_information()
        dataset_name = meta.get("dataset_name", "unknown dataset")
        task_type = "classification task"  # XGBoostBenchmark focuses on classification
        n_samples = meta.get("n_samples", "unknown number of")
        n_features = meta.get("n_features", "unknown number of")

        return (f"This task is based on the {dataset_name} dataset, which is a {task_type}. "
                f"It contains {n_samples} samples and {n_features} features. "
                "Generate XGBoost hyperparameters optimized for this specific task.")


    def _generate_hyperparameters(self):
        """Create English-only prompt for GPT-3"""
        prompt_lines = [
            "Generate optimized XGBoost hyperparameter configurations for the following task:",
            self.task_description,
            "\nHyperparameter ranges:",
        ]

        # Add hyperparameter ranges with types
        for param, (min_val, max_val) in self.param_ranges.items():
            dtype = "integer" if isinstance(min_val, int) else "float"
            prompt_lines.append(f"- {param}: {min_val} to {max_val} ({dtype})")

        # Add output format instructions and example
        prompt_lines.append("\nReturn only a JSON object with hyperparameter names and values. Example:")
        example = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            if isinstance(min_val, int):
                example[param] = (min_val + max_val) // 2
            else:
                example[param] = round((min_val + max_val) / 2, 3)
        prompt_lines.append(json.dumps(example))

        return self._call_gpt("\n".join(prompt_lines))

    def _call_gpt(self, prompt):
        """Call GPT-3 with English prompt"""
        # 新版本API调用方式
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        # 新版本获取响应的方式
        return json.loads(response.choices[0].message.content)

    def _validate_hyperparameters(self, params):
        """Validate hyperparameters against ranges"""
        validated = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            val = params.get(param, (min_val + max_val) / 2 if isinstance(min_val, float) else (min_val + max_val) // 2)
            validated[param] = max(min_val, min(val, max_val))
            if isinstance(min_val, int):
                validated[param] = int(validated[param])
        return validated

    def run_optimization(self):
        print(f"Running hyperparameter optimization on task ID {self.task_id}...")
        print(f"Task description: {self.task_description}")

        for trial in range(self.max_trials):
            print(f"\nTrial {trial + 1}/{self.max_trials}")
            params = self._generate_hyperparameters()
            params = self._validate_hyperparameters(params)
            # print(f"Hyperparameters: {params}")

            try:
                result = self.benchmark.objective_function(
                    config=params,
                    rng=42,fidelity={'n_estimators': 128, 'subsample': 0.1}
                )
                accuracy = result['val_scores']['acc']
                print(f"Validation accuracy: {accuracy:.4f}")
                self.history.append({"trial": trial + 1, "params": params, "accuracy": accuracy})
            except Exception as e:
                print(f"Evaluation failed: {e}")

        if self.history:
            best = max(self.history, key=lambda x: x["accuracy"])
            print(f"\nBest result - Trial {best['trial']}: Accuracy {best['accuracy']:.4f}")
            return best
        return None


if __name__ == "__main__":
    optimizer = GPT3HPOptimizer(task_id=167097, max_trials=5)
    optimizer.run_optimization()
