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

from pydantic import BaseModel
from r2egym.repo_analysis.execution_log_parser import decolor_dict_keys, parse_log_fn


class R2EGymTestSpec(BaseModel):
    metadata: dict

    @property
    def instance_id(self):
        return self.metadata["instance_id"]

    @property
    def gold_patch(self):
        return self.metadata["patch"]

    @property
    def eval_script(self):
        # r2e-gym eval scripts have exist in the docker image
        pass

    def _get_logs_eval(self, eval_output: str):
        instance = self.metadata
        repo = instance["repo"]
        return parse_log_fn(repo)(eval_output)

    def get_eval_report(self, eval_output: str):
        eval_report = {
            "resolved": False,
            "found_eval_status": False,
            "test_status": None,
        }

        # step 1: get logs eval
        parsed_status = self._get_logs_eval(eval_output)
        parsed_status = decolor_dict_keys(parsed_status)
        if parsed_status:
            eval_report["found_eval_status"] = True

        # step 2: get eval tests report
        expected_json = self.metadata["expected_output_json"]
        expected_status = json.loads(expected_json)
        expected_status = decolor_dict_keys(expected_status)

        parsed_status = {k.split(" - ")[0]: parsed_status[k] for k in sorted(parsed_status.keys())}
        expected_status = {k.split(" - ")[0]: expected_status[k] for k in sorted(expected_status.keys())}
        eval_report["test_status"] = {
            "parsed_status": parsed_status,
            "expected_status": expected_status,
        }
        if len(parsed_status) != len(expected_status):
            eval_report["resolved"] = False
        else:
            match = True
            for k in parsed_status.keys():
                if not k:
                    continue
                if k not in expected_status:
                    match = False
                    break
                if parsed_status[k] != expected_status[k]:
                    match = False
                    break
            eval_report["resolved"] = match
        return eval_report
