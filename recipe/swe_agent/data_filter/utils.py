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
import os
import re
from collections import defaultdict

from tqdm import tqdm


def parse_result_from_file(file_path: str, ignore_error=False) -> dict:
    pattern = r"Final report for\s+([^:]+):\s+(.*)"
    with open(file_path) as f:
        try:
            line = f.readlines()[-1].strip()
            m = re.search(pattern, line)
            instance_id = m.group(1)
            result_dict = ast.literal_eval(m.group(2))
            return instance_id, result_dict
        except Exception as e:
            if ignore_error:
                return None, None
            else:
                raise e


def process_data(data_root_path):
    full_results = defaultdict(list)
    for root, dirs, files in tqdm(os.walk(data_root_path)):
        if "run_instance.log" in files:
            instance_id, folder_name = root.split(os.path.sep)[-2:]
            # print(instance_id, folder_name)
            # assert folder_name == "eval_output" and len(dirs) == 0
            result_path = os.path.join(root, "run_instance.log")
            instance_id, result = parse_result_from_file(result_path)
            if result is not None:
                full_results[instance_id].append(
                    {
                        "result": result,
                        "eval_result_path": result_path,
                    }
                )
    return full_results
