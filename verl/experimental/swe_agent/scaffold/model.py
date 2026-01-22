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
import uuid
from functools import cached_property
from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema

from .tool_parser import FunctionCallFormatError, XMLToolParser


class MaxTokenExceededError(Exception):
    pass


class SWEChatModel:
    model_name: str
    """The name of the model"""

    client: Any
    """AsyncLLM server manager"""

    tokenizer: Any
    """Tokenizer for the model"""

    max_model_len: int
    """Max model context length"""

    tool_parser: str = "hermes"
    """Tool parser for the model"""

    max_parallel_calls: int = 1
    """Max parallel tool calls"""

    temperature: float = 1.0
    """Temperature for sampling"""

    top_p: float = 1.0
    """Top p for sampling"""

    top_k: int = -1
    """Top k for sampling"""

    repetition_penalty: float = 1.0
    """Repetition penalty for sampling"""

    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)
        self.loop = asyncio.get_running_loop()

    async def _preprocess(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, list[int]] | None,
    ):
        """Preprocess messages for chat completion.
        Args:
            messages (list[dict[str, str]]): List of messages.

        Returns:
            tuple[str, list[int], list[int]]: Request id, prompt ids, response mask.
        """

        assert messages[-1]["role"] == "user", f"Last message must be user, but got {messages[-1]['role']}"

        # Case 1: initial chat completion: [system], user
        if len(messages) == 1 or messages[-2]["role"] == "system":
            assert rollout_cache is None, "rollout_cache must be None for initial chat completion"
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                ),
            )
            rollout_cache = {"request_id": str(uuid.uuid4()), "prompt_ids": prompt_ids, "response_mask": []}
            return rollout_cache

        # deepcopy history_metadata
        assert "request_id" in rollout_cache
        assert "prompt_ids" in rollout_cache
        assert "response_mask" in rollout_cache

        # encode tool response
        tool_responses = messages[-1:]
        tool_response_ids = await self.loop.run_in_executor(
            None,
            lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        tool_response_ids = tool_response_ids[len(self.system_prompt) :]

        # append tool response to prompt
        rollout_cache["prompt_ids"] += tool_response_ids
        rollout_cache["response_mask"] += [0] * len(tool_response_ids)

        return rollout_cache

    async def query(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, str | list[int]] | None,
        **kwargs,
    ) -> list[dict] | dict:
        rollout_cache = await self._preprocess(messages, rollout_cache)
        request_id = rollout_cache["request_id"]
        prompt_ids = rollout_cache["prompt_ids"]

        if len(prompt_ids) >= self.max_model_len:
            raise MaxTokenExceededError(
                f"prompt_ids length {len(rollout_cache['prompt_ids'])} exceeds max_model_len {self.max_model_len}\n"
                f"Last tool response: {messages[-1]['content']}"
            )

        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
        if "sampling_params" in kwargs:
            sampling_params.update(kwargs["sampling_params"])

        token_output = await self.client.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        generation_info = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_output.token_ids),
        }
        response_ids = token_output.token_ids
        rollout_cache["prompt_ids"] += response_ids
        rollout_cache["response_mask"] += [1] * len(response_ids)
        response_str = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(response_ids))

        if len(rollout_cache["prompt_ids"]) >= self.max_model_len:
            raise MaxTokenExceededError(
                f"prompt_ids length {len(rollout_cache['prompt_ids'])} exceeds max_model_len {self.max_model_len}\n"
                f"Generated response:\n{response_str}"
            )
        return response_str, rollout_cache, generation_info

    @cached_property
    def system_prompt(self):
        # used to remove system prompt prefix when encoding tool response
        try:
            self._system_prompt = self.tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        except Exception:
            # Qwen3-Coder-30B-A3B-Instruct
            self._system_prompt = []
        return self._system_prompt

    async def parse_action_xml(
        self,
        model_output: list[int],
        tools: list[dict],
    ) -> dict:
        """Postprocess model_output when chat completion is done.

        Args:
            request_id (str): Unique request id.
            prompt_ids (list[int]): Input prompt token ids in this chat completion.
            response_mask (list[int]): Response mask before this chat completion.
            response_ids (list[int]): LLM generated token ids in this chat completion.

        Returns:
            CompletionResponse: Postprocessed message.
        """
        tool_parser = XMLToolParser()
        tools = [OpenAIFunctionToolSchema(**tool) for tool in tools]
        content, tool_calls = tool_parser.extract_tool_calls(model_output, tools)

        if len(tool_calls) == 0:
            raise FunctionCallFormatError("No function call found in the response.")
        elif len(tool_calls) > self.max_parallel_calls:
            raise FunctionCallFormatError(
                f"Number of tool calls {len(tool_calls)} exceeds max_parallel_calls {self.max_parallel_calls}."
            )

        return content, tool_calls
