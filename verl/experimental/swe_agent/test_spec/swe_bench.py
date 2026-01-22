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
from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.create_scripts import make_eval_script_list


class SWEBenchTestSpec(BaseModel):
    metadata: dict

    @property
    def instance_id(self):
        return self.metadata["instance_id"]

    @property
    def gold_patch(self):
        return self.metadata["patch"]

    @property
    def eval_script(self):
        instance = self.metadata
        repo = instance["repo"]
        version = instance.get("version")
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
        env_name = "testbed"
        repo_directory = f"/{env_name}"
        base_commit = instance["base_commit"]
        test_patch = instance["test_patch"]
        eval_script_list = make_eval_script_list(
            instance=instance,
            specs=specs,
            env_name=env_name,
            repo_directory=repo_directory,
            base_commit=base_commit,
            test_patch=test_patch,
        )
        eval_script = "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list) + "\n"
        return eval_script

    def _get_logs_eval(self, eval_output: str):
        instance = self.metadata
        repo = instance["repo"]
        log_parser = MAP_REPO_TO_PARSER[repo]
        if START_TEST_OUTPUT in eval_output and END_TEST_OUTPUT in eval_output:
            test_content = eval_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
            status_map = log_parser(test_content, None)
            return status_map, True
        else:
            status_map = {}
            return status_map, False

    def get_eval_report(self, eval_output: str):
        eval_report = {
            "resolved": False,
            "found_eval_status": False,
            "test_status": None,
        }

        # step 1: get logs eval
        status_map, found = self._get_logs_eval(eval_output)
        eval_report["found_eval_status"] = found
        if not found:
            return eval_report

        # step 2: get eval tests report
        eval_ref = {
            "instance_id": self.instance_id,
            "FAIL_TO_PASS": json.loads(self.metadata.get("FAIL_TO_PASS", "[]")),
            "PASS_TO_PASS": json.loads(self.metadata.get("PASS_TO_PASS", "[]")),
        }
        repo = self.metadata["repo"]
        eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
        report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
        eval_report["test_status"] = report
        if get_resolution_status(report) == ResolvedStatus.FULL.value:
            eval_report["resolved"] = True
        return eval_report
