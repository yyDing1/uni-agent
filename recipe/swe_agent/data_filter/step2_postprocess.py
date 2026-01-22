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

import json
import os

from utils import process_data

dataset = "swe-bench-live"
filter_model = "qwen3-coder-30b-a3b"
log_root_dir = f"/mnt/hdfs/yyding/swe-data/{dataset}/step2-model-eval"

log_dir = f"{log_root_dir}/eval-log/{filter_model}"
eval_report_path = f"{log_root_dir}/eval_report.json"
filtered_result_path = f"{log_root_dir}/result_list.txt"
final_data_path = f"{log_root_dir}/final_data.jsonl"

# 1. process eval log
if not os.path.exists(eval_report_path):
    full_results = process_data(log_dir)
    with open(eval_report_path, "w") as f:
        json.dump(full_results, f, indent=4)
else:
    with open(eval_report_path) as f:
        full_results = json.load(f)

# 2. filter instance with at only one sample succeed
if not os.path.exists(filtered_result_path):
    instances = full_results.keys()
    success_instances = []
    for instance_id in instances:
        assert len(full_results[instance_id]) == 16
        hav_success = [r["result"]["resolved"] for r in full_results[instance_id]]
        if sum(hav_success) > 0:
            success_instances.append(instance_id)
    print(f"total {len(instances)} instances, {len(success_instances)} instances have success")

    # with open(filtered_result_path, "w") as f:
    #     for instance_id, instance_results in full_results.items():
    #         if sum([1 for r in instance_results if r["succeed"]]) > 0:
    #             f.write(instance_id + "\n")
