# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


def parse_results(file_path):
    with open(file_path) as f:
        try:
            lines = f.readlines()
            last_line = lines[-1].strip()
            start_idx = last_line.index("{")
            end_idx = last_line.rindex("}")
            result = ast.literal_eval(last_line.strip()[start_idx : end_idx + 1])
        except Exception:
            result = None
        return result


def process_data(data_root_path):
    full_results = defaultdict(list)
    for root, dirs, files in tqdm(os.walk(data_root_path)):
        if "run_instance.log" in files:
            instance_id, _, folder_name = root.split(os.path.sep)[-3:]
            assert folder_name == "eval_output" and len(dirs) == 0
            result_path = os.path.join(root, "run_instance.log")
            result = parse_results(result_path)
            if result is not None:
                full_results[instance_id].append(
                    {
                        "resolved": result.get("resolved", False),
                        "eval_result_path": result_path,
                    }
                )
    return full_results


# full_results = process_data(
#     "/mnt/hdfs/yyding/swe-run/20260119_033209"
# )
output_path = [
    # "/mnt/hdfs/yyding/swe-data/swe-bench-verified/qwen3-coder-30b-a3b-n32/analysis.json",
    "recipe/plothub/analysis_temp.json",
]
# for path in output_path:
#     with open(path, "w") as f:
#         json.dump(full_results, f, indent=4)


def calculate_pass_k(analysis_json_path, k):
    with open(analysis_json_path) as f:
        full_results = json.load(f)

    def choose(n, m):
        val = 1
        for i in range(m):
            val *= n - i
            val /= i + 1
        return val

    pass_k = [0] * (k + 1)
    for result in full_results.values():
        correct_list = [item["resolved"] for item in result]
        n = len(correct_list)
        correct_count = sum(correct_list)
        for i in range(1, k + 1):
            if n - correct_count <= 0 or i > n - correct_count:
                pass_k[i] += 1
            else:
                pass_k[i] += 1 - choose(n - correct_count, i) / choose(n, i)

    pass_k = np.array(pass_k[1:]) / len(full_results)
    return pass_k


plt.rcParams.update(
    {
        "font.size": 14,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)
k = 32
fig, ax = plt.subplots(figsize=(6, 4))
ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
x = np.arange(k) + 1
y = calculate_pass_k("recipe/plothub/analysis_temp.json", k)
print(list(y))
ax.plot(x, y)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
# plt.savefig("recipe/plothub/output/test_time_scaling_temp.png", dpi=500)
