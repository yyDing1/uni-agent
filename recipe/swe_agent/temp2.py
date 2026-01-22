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
import glob
import os

from tqdm import tqdm

# data_path = "/tmp/trajectory/step0-validate"
data_path = "/mnt/hdfs/yyding/swe-logs/qwen3-coder-30b-a3b-n8-new-scaffold"

success_list = []
fail_list = []
wa_count = 0
tle_count = 0


def parse(line: str) -> dict:
    try:
        start_idx = line.index("{")
        end_idx = line.rindex("}")
        return ast.literal_eval(line.strip()[start_idx : end_idx + 1])
    except Exception:
        return None


log_paths = glob.glob(os.path.join(data_path, "**", "run_instance.log"), recursive=True)

for log_path in tqdm(log_paths):
    instance_id = os.path.basename(os.path.dirname(log_path))

    with open(log_path) as f:
        try:
            lines = f.readlines()
            last_line = lines[-1].strip()
            result = parse(last_line)
            assert isinstance(result, dict)
        except Exception:
            continue
        if "execution_time" not in result:
            result["execution_time"] = 10000
        if "resolved" not in result:
            result["resolved"] = False
            print("resolved not in result:", log_path)
            continue
        if result["resolved"]:
            success_list.append((instance_id, "AC   ", result["execution_time"]))
        else:
            if not result["eval_completed"]:
                tle_count += 1
                fail_list.append((instance_id, "TLE  ", result["execution_time"]))
            else:
                wa_count += 1
                fail_list.append((instance_id, "WA   ", result["execution_time"]))

print(
    f"{'-' * 20}\n"
    f"success:  {len(success_list)}\n"
    f"fail:     {len(fail_list)}\n"
    f"total:    {len(success_list + fail_list)}\n"
    f"Accuracy: {len(success_list) / (len(success_list + fail_list)):.2%}\n"
    f"{'-' * 20}\n"
    f"fail:     {len(fail_list)}\n"
    f"wa:       {wa_count}\n"
    f"tle:      {tle_count}\n"
    f"{'-' * 20}\n"
)

cnt = 0
for example in success_list:
    if example[-1] <= 240:
        cnt += 1
print(cnt)

# print(wa_count, tle_count)
# fail_list = sorted(fail_list, key=lambda x: x[0])
# for item in fail_list:
#     print(item[1], item[0], item[-1])


# full_list = success_list + fail_list
# full_list = sorted(full_list, key=lambda x: x[-1])
# for item in full_list[-30:]:
#     print(item[1], item[0], item[-1])
