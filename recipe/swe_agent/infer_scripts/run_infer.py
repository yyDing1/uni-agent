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

import argparse
import logging
import os

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import DictConfig

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/trainer/config")
    print(config_dir)
    with initialize_config_dir(config_dir=config_dir):
        config = compose(config_name="ppo_trainer")
    n = 1
    model_path = "/opt/tiger/verl-swe/Qwen3-Coder-30B-A3B-Instruct"
    config.actor_rollout_ref.rollout.multi_turn.format = "qwen3_coder"
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = "recipe/swe_agent/config/template_default.yaml"
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 100
    config.actor_rollout_ref.rollout.agent.num_workers = 1
    config.actor_rollout_ref.rollout.val_kwargs.temperature = 0.7
    config.actor_rollout_ref.rollout.val_kwargs.top_p = 0.8

    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = 8

    config.data.return_raw_chat = True
    config.data.max_prompt_length = 4096
    config.data.max_response_length = 65536
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 65536
    config.actor_rollout_ref.rollout.n = n
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 4
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.8
    config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1
    # config.actor_rollout_ref.rollout.enforce_eager = True

    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    return config


def run_inference(log_path: str = None):
    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_USE_V1": "1",
        }
    }
    if log_path:
        runtime_env["env_vars"]["LOG_DIR"] = log_path
    ray.init(runtime_env=runtime_env)

    # =========================== 1. Init rollout manager ===========================

    config = init_config()
    agent_loop_manager = init_agent_loop_manager(config)

    # =========================== 2. Generate sequences  ===========================

    # data_path = os.path.expanduser("/home/tiger/data/swe_agent/r2e_gym_subset.parquet")
    data_path = "/mnt/hdfs/yyding/data/swe-agent/swe_bench_verified.parquet"
    samples = load_dataset("parquet", data_files=data_path, split="train").to_list()[:4]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([sample["prompt"] for sample in samples], dtype=object),
            "agent_name": np.array([sample["agent_name"] for sample in samples], dtype=object),
            "tools_kwargs": np.array([sample["extra_info"]["tools_kwargs"] for sample in samples], dtype=object),
        },
    ).repeat(config.actor_rollout_ref.rollout.n)
    batch.meta_info = {
        "global_steps": -100,
        "validate": True,
    }
    output = agent_loop_manager.generate_sequences(batch)
    rm_scores = output.batch["rm_scores"].sum(dim=-1).tolist()
    print("Mean RM Score:", np.mean(rm_scores))
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()
    run_inference(log_path=args.log_dir)
