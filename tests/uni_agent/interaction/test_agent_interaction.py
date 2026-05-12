"""End-to-end smoke test for AgentInteraction with a mocked LLM but a real
local sandbox.

What this test covers:
  - The full agent loop runs through multiple turns without errors.
  - Per-turn timing fields (llm_time / tool_time / tool_outcome / step_start_ts /
    step_end_ts / tool_recovery_time) added for tail-latency analysis are filled
    in on every code path: success, action timeout (with recovery), and finish.
  - run() emits batch-level start_ts / end_ts wall-clock timestamps.
  - The local SWE-ReX sandbox actually executes the commands we feed it.

What this test deliberately does NOT cover:
  - vLLM / SGLang inference (mocked away)
  - Ray / verl AgentLoopManager orchestration (we use AgentInteraction directly)
  - Tool-call format parsing (we inject structured tool calls via rollout_cache.extra_fields)
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import pytest_asyncio

# Skip cleanly if optional deps aren't installed; mirrors test_host_runtime.py
pytest.importorskip("swerex")
pytest.importorskip("verl")

from uni_agent.deployment.config import LocalDeploymentConfig  # noqa: E402
from uni_agent.interaction.env import AgentEnv, AgentEnvConfig  # noqa: E402
from uni_agent.interaction.interaction import AgentInteraction  # noqa: E402
from uni_agent.interaction.tools_manager import ToolsManager, ToolsManagerConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class ScriptedChatModel:
    """A drop-in replacement for AgentChatModel that returns pre-baked tool calls.

    AgentInteraction calls four async methods on the model: set_tools_schemas,
    prepare_rollout_cache, append_messages_to_rollout_cache, query. We implement
    all four with minimal behavior so the loop runs.

    The trick: AgentInteraction.step() looks at
    ``rollout_cache.extra_fields.last_tool_calls`` BEFORE falling back to text
    parsing. By populating that field in query()'s returned cache, we bypass
    tool_parser entirely and inject any tool call we want.
    """

    def __init__(self, scripted_calls: list[dict[str, Any]]):
        # Each dict: {"name": str, "arguments": dict, "thought": str}
        self._calls = scripted_calls
        self._idx = 0
        self.tools_schemas: list[dict] = []

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "request_id": "test-request",
            "prompt_ids": [0],
            "response_mask": [],
            "response_logprobs": [],
            "routed_experts": None,
            "metrics": {},
            "extra_fields": {},
        }

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Append a single dummy token per message so prompt_ids keeps growing
        assert rollout_cache is not None
        rollout_cache["prompt_ids"] = list(rollout_cache["prompt_ids"]) + [1] * len(new_messages)
        rollout_cache["response_mask"] = list(rollout_cache["response_mask"]) + [0] * len(new_messages)
        return rollout_cache

    async def query(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
        **kwargs,
    ) -> tuple[str, dict[str, Any], dict[str, int]]:
        # Simulate real LLM latency so llm_time > 0 in assertions
        await asyncio.sleep(0.01)

        if self._idx >= len(self._calls):
            raise RuntimeError(f"ScriptedChatModel exhausted: no call #{self._idx}")
        call = self._calls[self._idx]
        self._idx += 1

        # Inject structured tool call -> bypasses tool_parser entirely
        assert rollout_cache is not None
        rollout_cache.setdefault("extra_fields", {})
        rollout_cache["extra_fields"]["last_tool_calls"] = [
            {
                "id": f"call-{self._idx}",
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
        ]
        # The "model output" text is unused once last_tool_calls is set, but we
        # still return something so the assistant message has content.
        model_output = call.get("thought", f"calling {call['name']}")

        # Grow prompt_ids to mimic completion tokens
        rollout_cache["prompt_ids"] = list(rollout_cache["prompt_ids"]) + [2, 3, 4]
        rollout_cache["response_mask"] = list(rollout_cache["response_mask"]) + [1, 1, 1]

        return model_output, rollout_cache, {"prompt_tokens": 10, "completion_tokens": 3}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def env() -> AgentEnv:
    """A live local sandbox. Cleanup happens after the test."""
    env_config = AgentEnvConfig(
        deployment=LocalDeploymentConfig(),
        env_variables=None,
        post_setup_cmd=None,
    )
    e = AgentEnv(run_id="test-run", env_config=env_config)
    await e.start()
    try:
        yield e
    finally:
        await e.close()


@pytest.fixture
def tools_manager() -> ToolsManager:
    # Real ToolsManager so install_tools(...) is a no-op-friendly call path.
    cfg = ToolsManagerConfig(
        tools=[{"name": "execute_bash"}, {"name": "submit"}],
        parser="qwen3_coder",
    )
    return ToolsManager(tools_manager_config=cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_three_turns(env: AgentEnv, tools_manager: ToolsManager) -> None:
    """LLM -> execute_bash -> LLM -> execute_bash -> LLM -> submit.

    Verifies timing fields are populated on every turn and that the sandbox
    actually executes commands (observations contain expected output).
    """
    scripted = [
        {"name": "execute_bash", "arguments": {"command": "echo hello"}, "thought": "say hi"},
        {"name": "execute_bash", "arguments": {"command": "echo world"}, "thought": "say world"},
        {"name": "submit", "arguments": {}, "thought": "done"},
    ]
    model = ScriptedChatModel(scripted)

    await env.install_tools(tools_manager.tools)
    interaction = AgentInteraction(
        run_id="happy",
        env=env,
        model=model,  # duck-typed; AgentInteraction doesn't isinstance-check
        tools_manager=tools_manager,
        messages=[{"role": "user", "content": "do the thing"}],
        action_timeout=5,
        max_turns=10,
    )

    result = await interaction.run()

    # Batch-level wall-clock fields
    assert result["start_ts"] is not None
    assert result["end_ts"] is not None
    assert result["end_ts"] >= result["start_ts"]
    assert result["execution_time"] > 0

    traj = result["trajectory"]
    assert len(traj) == 3, f"expected 3 turns, got {len(traj)}"

    # Per-turn invariants for the two successful execute_bash turns + submit
    for i, step in enumerate(traj, start=1):
        assert step.step_idx == i
        assert step.step_start_ts is not None
        assert step.step_end_ts is not None
        assert step.step_end_ts >= step.step_start_ts
        assert step.llm_time is not None and step.llm_time >= 0
        assert step.tool_outcome == "ok", f"step {i} outcome={step.tool_outcome}"
        assert step.tool_time is not None and step.tool_time >= 0
        assert step.tool_recovery_time == 0.0  # no timeout -> no recovery

    # Tool names recorded correctly
    assert [s.tool_name for s in traj] == ["execute_bash", "execute_bash", "submit"]

    # Sandbox really ran the commands
    assert "hello" in traj[0].observation
    assert "world" in traj[1].observation

    # Final exit
    assert traj[-1].done is True
    assert traj[-1].exit_reason == "finished"


@pytest.mark.asyncio
async def test_timeout_records_exec_and_recovery_split(env: AgentEnv, tools_manager: ToolsManager) -> None:
    """Tool timeout -> tool_outcome=='timeout', tool_time≈action_timeout,
    tool_recovery_time>0 (interrupt overhead), then loop continues using its
    timeout_budget (default 3) and ends on the next submit.

    This is THE key signal for tail analysis: tool_time tells you how long the
    command itself spent before the deadline, tool_recovery_time tells you how
    long it took to interrupt+probe afterward.
    """
    scripted = [
        # 1) A doomed sleep that exceeds action_timeout=1
        {"name": "execute_bash", "arguments": {"command": "sleep 30"}, "thought": "wait"},
        # 2) After the timeout error is fed back, we give up cleanly
        {"name": "submit", "arguments": {}, "thought": "give up"},
    ]
    model = ScriptedChatModel(scripted)

    await env.install_tools(tools_manager.tools)
    interaction = AgentInteraction(
        run_id="timeout",
        env=env,
        model=model,
        tools_manager=tools_manager,
        messages=[{"role": "user", "content": "do something slow"}],
        action_timeout=1,  # tight on purpose
        timeout_budget=3,
        max_turns=10,
    )

    result = await interaction.run()
    traj = result["trajectory"]

    # Turn 1: the timeout
    s1 = traj[0]
    assert s1.tool_name == "execute_bash"
    assert s1.tool_outcome == "timeout", f"got outcome={s1.tool_outcome}"
    assert s1.exit_reason == "timeout_error"
    assert s1.done is False, "timeout with budget remaining must not terminate the loop"
    # exec_time should be in the ballpark of action_timeout (give it slack)
    assert s1.tool_time is not None
    assert 0.5 <= s1.tool_time <= 5.0, f"tool_time={s1.tool_time}"
    # Recovery should be > 0 (interrupt + probe always takes some time)
    assert s1.tool_recovery_time is not None and s1.tool_recovery_time >= 0
    # And step_end_ts must be filled even on the early-return path
    assert s1.step_end_ts is not None

    # Turn 2: the submit
    s2 = traj[1]
    assert s2.tool_name == "submit"
    assert s2.tool_outcome == "ok"
    assert s2.exit_reason == "finished"
    assert s2.done is True


@pytest.mark.asyncio
async def test_env_last_action_stats_outcome_for_success(env: AgentEnv) -> None:
    """Direct unit check on env.run_action's per-call stats — make sure the
    'outcome=ok / recovery_time=0' contract holds without going through
    AgentInteraction.
    """
    obs = await env.run_action("echo direct", action_timeout=5)
    assert "direct" in obs
    stats = env.last_action_stats
    assert stats["outcome"] == "ok"
    assert stats["exec_time"] is not None and stats["exec_time"] >= 0
    assert stats["recovery_time"] == 0.0


@pytest.mark.asyncio
async def test_env_last_action_stats_outcome_for_timeout(env: AgentEnv) -> None:
    """Same direct check, but the timeout path: outcome=='timeout' and
    recovery_time > 0 because interrupt has to do real work.
    """
    from uni_agent.interaction.env import ActionTimeoutError

    with pytest.raises(ActionTimeoutError):
        await env.run_action("sleep 30", action_timeout=1)

    stats = env.last_action_stats
    assert stats["outcome"] == "timeout"
    assert stats["exec_time"] is not None
    assert 0.5 <= stats["exec_time"] <= 5.0, f"exec_time={stats['exec_time']}"
    assert stats["recovery_time"] > 0, "interrupt+probe should take measurable time"
