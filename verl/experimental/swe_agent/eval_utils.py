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

import uuid
from pathlib import Path

from verl.experimental.swe_agent.async_logging import add_file_handler, cleanup_handlers, get_logger
from verl.experimental.swe_agent.scaffold import SWEAgentEnv, SWEEnvConfig

from .test_spec import SWETestSpec


async def run_instance(
    test_spec: SWETestSpec,
    patch: str,
    env_config: SWEEnvConfig,
    log_dir: Path | None = None,
    eval_timeout: int = 300,
):
    instance_id = test_spec.instance_id

    if log_dir is None:
        log_dir = Path("/tmp/eval") / "gold" / instance_id

    # step 0: setup logger & env
    run_id = str(uuid.uuid4())
    env = SWEAgentEnv.from_config(env_config, run_id=run_id)
    log_file = log_dir / "run_instance.log"
    logger = get_logger("swea-eval", run_id=run_id)
    add_file_handler(log_file, run_id, level="info")
    await env.start(eval_only=True)

    final_report = {
        "patch_exists": False,
        "patch_applied_completed": False,
        "eval_completed": False,
        "eval_execution_time": None,
        "resolved": False,
    }

    # step 1: check patch exists
    final_report["patch_exists"] = True if patch else False
    if not final_report["patch_exists"]:
        logger.info(f"Patch does not exist for {instance_id}")

    # step 2: apply patch
    if final_report["patch_exists"]:
        patch_file_local = log_dir / "patch.diff"
        applied_patch_success = await env.apply_patch(
            patch=patch,
            patch_file_local=patch_file_local,
        )
        final_report["patch_applied_completed"] = applied_patch_success
        if not final_report["patch_applied_completed"]:
            logger.info(f"Patch applying failed for {instance_id}, exit evaluation")

    # step 3: run eval_script
    if final_report["patch_applied_completed"]:
        test_file_local = log_dir / "run_tests.sh"
        test_output_file = log_dir / "test_output.txt"
        report_file = log_dir / "report.json"
        eval_report, eval_completed = await env.evaluate(
            test_spec=test_spec,
            eval_timeout=eval_timeout,
            test_file_local=test_file_local,
            test_output_file=test_output_file,
            report_file=report_file,
        )
        final_report["eval_completed"] = eval_completed
        if not final_report["eval_completed"]:
            logger.info(f"Evaluation failed for {instance_id}, exit evaluation")

    if final_report["eval_completed"]:
        final_report["eval_execution_time"] = eval_report["execution_time"]
        final_report["resolved"] = eval_report["resolved"]

    await env.close()

    logger.info(f"Final report for {instance_id}: {final_report}")
    cleanup_handlers(run_id)
    return final_report
