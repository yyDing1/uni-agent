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

import asyncio
import logging
import os
import uuid
from pathlib import Path

import ray
from datasets import load_dataset

from recipe.swe_agent.fass_deployment import VefaasDeploymentConfig
from recipe.swe_agent.swe_eval.run_evaluation import run_instance
from recipe.swe_agent.swe_eval.test_spec import make_test_spec
from recipe.swe_agent.swe_scaffold import SWEEnvConfig

run_id = str(uuid.uuid4())

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


async def run_sample(sample):
    dataset_id = sample["extra_info"]["tools_kwargs"]["dataset_id"]
    instance_id = sample["extra_info"]["tools_kwargs"]["instance_id"]
    metadata = sample["extra_info"]["tools_kwargs"]["metadata"]

    vefass_config = VefaasDeploymentConfig(
        type="vefaas",
        command="curl -fsSL https://pjw-test-empty.tos-cn-beijing.ivolces.com/bin/tos_swe_rex.sh | bash -s -- {token}",
        timeout=600.0,
        startup_timeout=600.0,
        dataset_id=dataset_id,
        instance_id=instance_id,
    )
    env_config_dict = {
        "repo": {
            "repo_name": "testbed",
            "base_commit": metadata.get("base_commit"),
            "reset": True if dataset_id == "swe-bench-verified" else False,
        },
        "tools": [
            {"name": "str_replace_editor"},
            {"name": "execute_bash"},
            {"name": "submit"},
        ],
        "deployment": vefass_config,
        "action_timeout": 300,
    }
    env_config = SWEEnvConfig(**env_config_dict)

    test_spec = make_test_spec(dataset_id, metadata)
    gold_patch = test_spec.gold_patch
    result = await run_instance(
        test_spec=test_spec,
        patch=gold_patch,
        env_config=env_config,
        log_dir=Path("/tmp/eval_gt_temp") / instance_id,
        eval_timeout=300.0,
    )
    return result


@ray.remote
class TestEvalActor:
    _semaphore = asyncio.Semaphore(64)

    async def run_sample(self, samples):
        tasks = [self.run_single(sample) for sample in samples]
        return await asyncio.gather(*tasks)

    async def run_single(self, sample):
        async with self._semaphore:
            return await run_sample(sample)


def main():
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                # "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
    )

    data_path = os.path.expanduser("/tmp/step1-eval-gt/step1-output.parquet")
    worker_num = 8
    samples = load_dataset("parquet", data_files=data_path, split="train").to_list()
    workers = [TestEvalActor.remote() for _ in range(worker_num)]

    futures = []
    chunk_size = len(samples) // worker_num if len(samples) % worker_num == 0 else len(samples) // worker_num + 1
    for idx in range(0, len(samples), chunk_size):
        worker = workers[idx // chunk_size]
        chunk = samples[idx : idx + chunk_size]
        futures.append(worker.run_sample.remote(chunk))

    results_subset = ray.get(futures)
    results = [item for sublist in results_subset for item in sublist]

    all_num = len(results)
    success_num = len([item for item in results if item["resolved"]])
    fail_wa_num = len([item for item in results if not item["resolved"] and item["eval_completed"]])
    fail_tle_num = len([item for item in results if not item["resolved"] and not item["eval_completed"]])
    logger.info(
        f"all_num: {all_num}, success_num: {success_num}, fail_wa_num: {fail_wa_num}, fail_tle_num: {fail_tle_num}"
    )


if __name__ == "__main__":
    main()
