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

import time
from pathlib import Path

import orjson
from pydantic import BaseModel

from verl.experimental.swe_agent.async_logging import get_logger
from verl.experimental.swe_agent.test_spec import SWETestSpec
from verl.tools.schemas import OpenAIFunctionToolCall

from .env import ActionIncorrectSyntaxError, ActionTimeoutError, SWEAgentEnv
from .model import MaxTokenExceededError, SWEChatModel
from .template import SWETemplateConfig
from .tool_parser import FunctionCallFormatError


class StepOutput(BaseModel):
    step_idx: int

    response: str = ""
    thought: str = ""
    action: str = ""
    observation: str = ""
    execution_time: float | None = None
    done: bool = False
    exit_reason: str = ""


def fast_deepcopy(obj):
    return orjson.loads(orjson.dumps(obj))


class SWEAgent:
    def __init__(
        self,
        run_id: str,
        env: SWEAgentEnv,
        model: SWEChatModel,
        template: SWETemplateConfig,
        output_dir: Path,
    ):
        self.env = env
        self.model = model
        self.template = template
        self.output_dir = output_dir
        self.logger = get_logger("agent", run_id)

    async def step(self, step_idx: int):
        # step index start from 1
        step_output = StepOutput(step_idx=step_idx)
        self.logger.info(f"{'=' * 25} STEP {step_idx} {'=' * 25}")

        # step 1: prepare template
        steps_remaining = self.max_iter - step_idx + 1
        if steps_remaining > 0:
            stepcount_message = f"Steps Remaining: {steps_remaining}"
        else:
            stepcount_message = "This is your last step, make sure to submit your final answer."
        self.messages[-1]["content"] += f"\n{stepcount_message}"
        self.logger.info(f"🤖 MODEL INPUT\n{self.messages[-1]['content']}")

        # step 2: generate response and update rollout cache
        messages = fast_deepcopy(self.messages)
        rollout_cache = fast_deepcopy(self.rollout_cache)
        try:
            model_output, rollout_cache, generation_info = await self.model.query(
                messages=messages,
                rollout_cache=rollout_cache,
            )
            step_output.response = model_output
            self.logger.info(
                f"Prompt Tokens: {generation_info['prompt_tokens']}, "
                f"Completion Tokens: {generation_info['completion_tokens']}"
            )
            self.logger.debug(f"Model Output:\n{model_output}")
        except MaxTokenExceededError as e:
            self.logger.error(str(e))
            step_output.exit_reason = "token_limit"
            step_output.done = True
            return step_output

        # step 3: parse model response to actions
        tool_commands = self.env.commands
        self.rollout_cache = rollout_cache
        self.messages.append({"role": "assistant", "content": model_output})  # tool call message
        try:
            content, tool_calls = await self.model.parse_action_xml(model_output=model_output, tools=tool_commands)
        except FunctionCallFormatError as e:
            self.messages.append({"role": "user", "content": str(e)})  # error message
            step_output.exit_reason = "format_error"
            model_output_preview = "\n".join(model_output.splitlines()[:20])
            self.logger.error(
                f"Fail to parse thought and action from model output.\n"
                f"Error Message: {str(e)}\n"
                f"Model Output (first 20 lines): {model_output_preview}"
            )
            return step_output

        # step 4: run action in the environment
        tool_call: OpenAIFunctionToolCall = tool_calls[0]
        action_cmd = self.env.get_tool_bash_command(tool_call)
        step_output.thought = content
        step_output.action = action_cmd
        self.logger.info(f"💭 THOUGHT:\n{content}")
        self.logger.info(f"🎬 ACTION:\n{action_cmd}")
        execution_t0 = time.perf_counter()
        try:
            observation = await self.env.run_action(action_cmd)
            tool_message = {"role": "user", "content": observation}
            self.messages.append(tool_message)  # tool response message
            step_output.observation = observation
        except ActionTimeoutError as e:
            self.logger.error(str(e))
            self.messages.append({"role": "user", "content": str(e)})
            step_output.exit_reason = "timeout_error"
            self.logger.info(f"Existing timeout budget: {self.timeout_budget}")
            if self.timeout_budget > 0:
                self.timeout_budget -= 1
                return step_output
            else:
                step_output.done = True
                return step_output
        except ActionIncorrectSyntaxError as e:
            self.logger.error(str(e))
            self.messages.append({"role": "user", "content": str(e)})
            step_output.exit_reason = "syntax_error"
            return step_output

        # step 5: finalize step output
        execution_time = time.perf_counter() - execution_t0
        step_output.execution_time = execution_time
        if observation.strip() == "<<<Finished>>>":
            step_output.done = True
            step_output.exit_reason = "finished"
        else:
            step_output.done = False
            step_output.exit_reason = "completed"

        return step_output

    async def run(
        self, problem_statement: str, max_iter: int = 50, timeout_budget: int = 10, test_spec: SWETestSpec = None
    ):
        await self.env.start()
        self.max_iter = max_iter
        self.timeout_budget = timeout_budget
        self.trajectory: list[StepOutput] = []
        tool_commands = self.env.commands
        system_prompt = self.template.get_system_prompt(tools=tool_commands)
        self.logger.info(f"SYSTEM:\n{system_prompt}")
        instance_prompt = self.template.instance_template.format(problem_statement=problem_statement)
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        self.rollout_cache: dict[str, str | list[int]] | None = None

        done = False
        step_idx = 0
        while not done:
            # we start from 1
            step_idx += 1
            try:
                step_output = await self.step(step_idx=step_idx)
                self.trajectory.append(step_output)
                done = step_output.done
                if step_idx >= max_iter:
                    self.logger.error(f"Exit due to max step limit: {max_iter}")
                    step_output = StepOutput(step_idx=step_idx, exit_reason="max_step_limit")
                    self.trajectory.append(step_output)
                    break
            except Exception as e:
                # this should not happen, if it happens, we should fix the code
                self.logger.critical(f"Exit due to unknown error: {str(e)}")
                step_output = StepOutput(step_idx=step_idx, exit_reason="unknown_error")
                self.trajectory.append(step_output)
                break

        patch_content = await self.env.get_patch()
        patch_output_file = self.output_dir / "patch.diff"
        patch_output_file.write_text(patch_content)
        message_output_file = self.output_dir / "messages.json"
        message_output_file.write_bytes(orjson.dumps(self.messages, option=orjson.OPT_INDENT_2))
        trajectory_output_file = self.output_dir / "trajectory.json"
        trajectory_output_file.write_bytes(
            orjson.dumps([step.model_dump() for step in self.trajectory], option=orjson.OPT_INDENT_2)
        )
        trajectory_info = {
            "submission_patch": patch_content,
            "exit_reason": self.trajectory[-1].exit_reason if len(self.trajectory) > 0 else None,
            "num_turns": len(self.trajectory),
            "rollout_cache": self.rollout_cache,
        }
        if test_spec is not None:
            eval_report, eval_completed = await self.env.evaluate(
                test_spec=test_spec,
                test_output_file=self.output_dir / "eval_output.txt",
                report_file=self.output_dir / "eval_report.json",
                eval_timeout=300,
            )
            trajectory_info["reward_score"] = eval_completed and eval_report["resolved"]

        info_output_file = self.output_dir / "info.json"
        info_output_file.write_bytes(orjson.dumps(trajectory_info, option=orjson.OPT_INDENT_2))
        await self.env.close()
        return trajectory_info
