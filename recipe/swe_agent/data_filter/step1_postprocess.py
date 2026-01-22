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

from datasets import load_dataset
from utils import process_data

dataset = "swe-bench-live"
log_root_dir = f"/mnt/hdfs/yyding/swe-data/{dataset}/step1-eval-gt"

log_dir = f"{log_root_dir}/eval-log"
eval_report_path = f"{log_root_dir}/eval-report.json"
filtered_result_path = f"{log_root_dir}/result_list.txt"
final_data_path = f"{log_root_dir}/step1-output.parquet"
max_exec_time = 300.0

# 1. process eval log
if not os.path.exists(eval_report_path):
    full_results = process_data(log_dir)
    with open(eval_report_path, "w") as f:
        json.dump(full_results, f, indent=4)
else:
    print(f"{eval_report_path} already exists, skip")
    with open(eval_report_path) as f:
        full_results = json.load(f)

# 2. filter success instance with exec_time < max_exec_time
if not os.path.exists(filtered_result_path):
    instances = full_results.keys()
    success_list = []
    fail_list = []
    for instance_id in instances:
        assert len(full_results[instance_id]) == 1
        result = full_results[instance_id][0]["result"]
        if result["resolved"] and result["eval_completed"]:
            success_list.append((instance_id, 100))
        else:
            fail_list.append((instance_id, 100))

    print(f"success_cnt = {len(success_list)}")
    print(f"total = {len(success_list + fail_list)}")
    print(f"success instance (0 - 60s) = {sum([1 for _, t in success_list if t < 60.0])}")
    print(f"success instance (60 - 120s) = {sum([1 for _, t in success_list if 60.0 <= t < 120.0])}")
    print(f"success instance (120 - 240s) = {sum([1 for _, t in success_list if 120.0 <= t < 240.0])}")
    print(f"success instance (240s+) = {sum([1 for _, t in success_list if t >= 240.0])}")

    filtered_result = [instance_id for instance_id, exec_time in success_list if exec_time < max_exec_time]
    print(f"filtered_instance_cnt = {len(filtered_result)}")
    with open(filtered_result_path, "w") as f:
        for instance_id in filtered_result:
            f.write(f"{instance_id}\n")
else:
    print(f"{filtered_result_path} already exists, skip")
    with open(filtered_result_path) as f:
        filtered_result = [line.strip() for line in f.readlines()]

# 3. filter final data
if not os.path.exists(final_data_path):
    raw_dataset_path = "/mnt/hdfs/yyding/swe-data/r2e-gym-subset/r2e_gym_subset.parquet"
    dataset = load_dataset("parquet", data_files=raw_dataset_path, split="train")
    final_dataset = dataset.filter(lambda x: x["extra_info"]["tools_kwargs"]["instance_id"] in filtered_result)
    final_dataset.to_parquet(final_data_path)
else:
    print(f"{final_data_path} already exists, skip")
    final_dataset = load_dataset("parquet", data_files=final_data_path, split="train")
