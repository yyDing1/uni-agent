import json
import re
import time
import uuid
from pathlib import Path

from r2egym.repo_analysis.execution_log_parser import decolor_dict_keys, parse_log_fn

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await


@register_reward_spec("r2e_gym")
class R2EGymRewardSpec(AbstractRewardSpec):
    def __init__(self, *, run_id: str, metadata: dict, env: AgentEnv, eval_timeout: int = 300):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout

    @auto_await
    async def apply_gold_patch(self) -> str:
        gold_patch = self.metadata["patch"]
        await self._apply_patch(gold_patch)

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[dict | None, bool]:
        """Run eval script in container via env.communicate (no execute). Returns (eval_report, success)."""
        result = {
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "resolved": False,
        }

        eval_script_container = "/root/run_tests.sh"
        execution_t0 = time.perf_counter()
        try:
            cmd_str = f"bash {eval_script_container}"
            output = await self.env.communicate(cmd_str, timeout=self.eval_timeout, check="ignore")

            execution_time = time.perf_counter() - execution_t0
            result["eval_completed"] = True
            result["eval_execution_time"] = execution_time

            # Remove ANSI escape codes and \r
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

            eval_report = self._get_eval_report(output)
            result["eval_report"] = eval_report
            self.logger.info(f"Eval report: {eval_report}")
            result["resolved"] = eval_report["resolved"]
        except Exception as e:
            self.logger.error(f"Failed to evaluate: {e}")
        return result["resolved"], result

    @auto_await
    async def _apply_patch(self, patch: str) -> None:
        """Apply a patch string to the env. Tries multiple apply strategies in order."""
        if not patch or not patch.strip():
            self.logger.info("Empty patch, nothing to apply.")
            return
        patch_path = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
        await self.env.write_file(patch_path, patch)
        commands = [
            f"cd /testbed && git apply --whitespace=fix {patch_path.as_posix()}",
            f"cd /testbed && git apply --reject --whitespace=nowarn {patch_path.as_posix()}",
            f"cd /testbed && patch --batch --fuzz=5 -p1 -i {patch_path.as_posix()}",
        ]
        last_error: Exception | None = None
        for cmd in commands:
            try:
                await self.env.communicate(cmd, check="raise")
                self.logger.info("Applied patch successfully!")
                return
            except RuntimeError as e:
                last_error = e
                continue
        raise RuntimeError("Failed to apply patch with any command") from last_error

    def _get_logs_eval(self, eval_output: str):
        instance = self.metadata
        repo = instance["repo"]
        return parse_log_fn(repo)(eval_output)

    def _get_eval_report(self, eval_output: str):
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
