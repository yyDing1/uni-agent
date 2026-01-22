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
import os
import uuid
from pathlib import Path
from typing import Any

import ray
import yaml

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.experimental.swe_agent.async_logging import add_file_handler, cleanup_handlers, get_logger
from verl.experimental.swe_agent.eval_utils import run_instance
from verl.experimental.swe_agent.scaffold import SWEAgent, SWEAgentEnv, SWEChatModel, SWEEnvConfig, SWETemplateConfig
from verl.experimental.swe_agent.test_spec import make_test_spec


class SWEAgentLoop(AgentLoopBase):
    _semaphore = asyncio.Semaphore(64)

    async def run(self, sampling_params: dict[str, Any], trajectory_info: dict, **kwargs) -> AgentLoopOutput:
        async with self._semaphore:
            return await self._run(sampling_params, trajectory_info, **kwargs)

    async def _run(self, sampling_params: dict[str, Any], trajectory_info: dict, **kwargs) -> AgentLoopOutput:
        # step 1: init
        run_id = str(uuid.uuid4())
        self.logger = get_logger("swe-agent-loop", run_id=run_id)
        agent_loop_config_path = self.config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        assert agent_loop_config_path is not None, "agent_loop_config_path is None"
        resolved_path = resolve_config_path(agent_loop_config_path)
        config_dict = yaml.safe_load(Path(resolved_path).read_text())[0]

        # init chat model & env & template
        chat_model = self._init_chat_model(sampling_params)
        dataset_id = kwargs["tools_kwargs"]["dataset_id"]
        instance_id = kwargs["tools_kwargs"]["instance_id"]
        metadata = kwargs["tools_kwargs"]["metadata"]

        env_config_dict = config_dict["env"]
        env_config_dict["repo"] = {
            "repo_name": "testbed",
            "base_commit": metadata.get("base_commit", "HEAD"),
            "reset": True if dataset_id == "swe-bench-verified" else False,
        }
        env_config_dict["deployment"].update({"dataset_id": dataset_id, "instance_id": instance_id})
        env_config = SWEEnvConfig(**env_config_dict)
        env = SWEAgentEnv.from_config(env_config, run_id=run_id)
        template_dict = config_dict["template"]
        template = SWETemplateConfig(**template_dict)

        # init agent
        log_dir = os.getenv("LOG_DIR", None)
        if not log_dir:
            step_idx = trajectory_info["step"]
            validate = trajectory_info["validate"]
            log_dir = Path("/tmp/trajectory") / f"step{step_idx}-{'validate' if validate else 'train'}"

        output_dir = Path(log_dir) / instance_id / run_id
        self.logger.info(f"running logs to {output_dir}")
        agent = SWEAgent(run_id=run_id, env=env, model=chat_model, template=template, output_dir=output_dir)

        add_file_handler(output_dir / "run.log", run_id)

        ray_task_id = ray.get_runtime_context().get_task_id()
        self.logger.info(f"ray_task_id: {ray_task_id}")
        self.logger.info(f"start running instance {instance_id} with run_id {run_id}")
        self.logger.info(f"model: {self.config.actor_rollout_ref.model.path}")
        self.logger.info(f"sampling_params: {sampling_params}")
        self.logger.info(f"multi_turn_config: {self.config.actor_rollout_ref.rollout.multi_turn}")
        self.logger.info(f"agent_config: {self.config.actor_rollout_ref.rollout.agent}")

        # step 2: run agent loop
        max_iter = self.config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        if dataset_id == "swe-bench-verified":
            traj_info = await agent.run(max_iter=max_iter, problem_statement=metadata["problem_statement"])
            dataset_id = kwargs["tools_kwargs"]["dataset_id"]
            metadata = kwargs["tools_kwargs"]["metadata"]
            test_spec = make_test_spec(dataset_id=dataset_id, metadata=metadata)

            patch = traj_info["submission_patch"]
            eval_output_dir = output_dir / "eval_output"
            eval_env_config = env_config.model_copy()
            eval_report = await run_instance(
                test_spec=test_spec,
                patch=patch,
                env_config=eval_env_config,
                log_dir=eval_output_dir,
                eval_timeout=300.0,
            )
            agent_loop_output = self.convert_to_agent_output(
                num_turns=traj_info["num_turns"],
                rollout_cache=traj_info["rollout_cache"],
                reward_score=eval_report["resolved"],
            )
        elif dataset_id == "r2e-gym-subset":
            test_spec = make_test_spec(dataset_id=dataset_id, metadata=metadata)
            traj_info = await agent.run(
                max_iter=max_iter, problem_statement=metadata["problem_statement"], test_spec=test_spec
            )
            exit_reason = traj_info["exit_reason"]
            if exit_reason == "finished":
                return self.convert_to_agent_output(
                    num_turns=traj_info["num_turns"],
                    rollout_cache=traj_info["rollout_cache"],
                    reward_score=traj_info["reward_score"],
                )
            else:
                return self._agent_loop_error_output()

        # step 5: cleanup logger handlers
        cleanup_handlers(run_id=run_id)
        return agent_loop_output

    def _init_chat_model(self, sampling_params: dict[str, Any]) -> SWEChatModel:
        model_name = self.config.actor_rollout_ref.model.path
        temperature = sampling_params["temperature"]
        top_p = sampling_params["top_p"]
        rollout_config = self.config.actor_rollout_ref.rollout
        max_model_len = (
            rollout_config.max_model_len
            if rollout_config.max_model_len
            else rollout_config.prompt_length + rollout_config.response_length
        )
        chat_model = SWEChatModel(
            model_name=model_name,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            max_model_len=max_model_len,
            tool_parser=rollout_config.multi_turn.format,
            max_parallel_calls=rollout_config.multi_turn.max_parallel_calls,
            temperature=temperature,
            top_p=top_p,
        )
        return chat_model

    def _agent_loop_error_output(self):
        return AgentLoopOutput(
            prompt_ids=[self.tokenizer.pad_token_id] * 512,
            response_ids=[self.tokenizer.pad_token_id] * 512,
            response_mask=[0] * 512,
            reward_score=0,
            num_turns=0,
            metrics={},
        )

    def convert_to_agent_output(self, num_turns: int, rollout_cache: dict, reward_score: bool) -> AgentLoopOutput:
        self.logger.info(f"num_turns: {num_turns}")
        if rollout_cache is None:
            self.logger.critical("rollout_cache is None")
            return self._agent_loop_error_output()

        prompt_ids = rollout_cache["prompt_ids"]
        response_mask = rollout_cache["response_mask"]

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: -len(response_mask)]

        max_prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        max_response_length = self.config.actor_rollout_ref.rollout.response_length

        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]
            self.logger.warning(
                f"prompt_ids length {len(prompt_ids)} exceeds max_prompt_length {max_prompt_length} "
                "truncate prompt_ids length"
            )
        if len(response_ids) > max_response_length:
            response_ids = response_ids[:max_response_length]
            response_mask = response_mask[:max_response_length]
            self.logger.warning(
                f"response_ids length {len(response_ids)} exceeds max_response_length {max_response_length} "
                "truncate response_ids length"
            )

        self.logger.info(f"prompt_ids length: {len(prompt_ids)}")
        self.logger.info(f"response_ids length: {len(response_ids)}")
        self.logger.info(f"reward_score: {reward_score}")

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward_score=reward_score,
            num_turns=num_turns,
            metrics={},
        )
        return output
