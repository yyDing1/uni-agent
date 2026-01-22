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

import importlib
import json
import re
import shlex
import time
import uuid
from pathlib import Path, PurePath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError
from swerex.runtime.abstract import (
    BashAction,
    BashInterruptAction,
    Command,
    ReadFileRequest,
    WriteFileRequest,
)

from verl.experimental.swe_agent.async_logging import get_logger
from verl.experimental.swe_agent.fass_deployment import VefaasDeploymentConfig
from verl.experimental.swe_agent.test_spec import SWETestSpec
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionToolCall

tools_module = importlib.import_module(".tools", package=__package__)


class ActionTimeoutError(Exception):
    pass


class ActionIncorrectSyntaxError(Exception):
    pass


class SWERepoConfig(BaseModel):
    repo_name: str
    base_commit: str | None = Field(default="HEAD")
    reset: bool = True
    model_config = ConfigDict(extra="forbid")


class SWEToolConfig(BaseModel):
    name: str

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "tools" / self.name

    @property
    def command(self) -> dict:
        tool_command = getattr(tools_module, f"{self.name}_tool", None)
        if not tool_command:
            raise ValueError(f"Tool {self.name} not found")
        return tool_command


class SWEEnvConfig(BaseModel):
    repo: SWERepoConfig = Field(description="Repository configuration")
    tools: list[SWEToolConfig] = Field(description="Tools configuration")
    deployment: VefaasDeploymentConfig = Field(description="Vefaas deployment configuration")
    action_timeout: int = Field(default=120, description="Timeout for each action in seconds")
    model_config = ConfigDict(extra="forbid")


class SWEAgentEnv:
    def __init__(
        self,
        run_id: str,
        repo: SWERepoConfig,
        tools: list[SWEToolConfig],
        deployment: VefaasDeploymentConfig,
        action_timeout: int,
    ):
        """
        This class represents the environment in which we solve the tasks.

        Args:
            deployment: SWE-ReX deployment instance (only support vefass deployment)
            repo: Repository configuration object, or anything following the `Repo` protocol
        """
        super().__init__()
        self.run_id = run_id
        self.repo = repo
        self.tools = tools
        self.deployment = deployment.get_deployment(run_id)
        self.action_timeout = action_timeout
        self.logger = get_logger("env", run_id)

    @classmethod
    def from_config(cls, config: SWEEnvConfig, run_id: str = None) -> Self:
        """Create an environment instance from a configuration object.
        This is the recommended way to create an environment instance, unless you need
        more flexibility.
        """
        if not run_id:
            run_id = str(uuid.uuid4())
        return cls(
            run_id=run_id,
            repo=config.repo,
            tools=config.tools,
            deployment=config.deployment,
            action_timeout=config.action_timeout,
        )

    async def start(self, eval_only: bool = False) -> None:
        """Start the environment"""

        self.logger.info("Beginning environment startup...")

        # step 1: start the deployment and set environment variables
        await self.deployment.start()
        self.logger.info("Remote Runtime Initialized")

        if eval_only:
            return self

        # step 2: reset the environment to a clean state
        await self.set_env_variables(
            {
                "PIP_PROGRESS_BAR": "off",
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "TQDM_DISABLE": "1",
                "GIT_PAGER": "cat",
            }
        )
        if self.repo.reset:
            reset_commands = [
                f"cd /{self.repo.repo_name}",
                "git restore .",
                "git reset --hard",
                f"git checkout {self.repo.base_commit}",
                "git clean -fdq",
            ]
            await self.communicate(
                input=" && ".join(reset_commands),
                check="raise",
                error_msg="Failed to clean repository",
                # Sometimes this is slow because it rebuilds some index
                timeout=60,
            )
            self.logger.info(f"Repository {self.repo.repo_name} reset to commit {self.repo.base_commit}")

        # step 3: install tools
        self.commands = []
        for tool in self.tools:
            tool_name = tool.name
            local_tool_path = tool.local_path
            assert local_tool_path.is_file(), f"Tool {tool_name} not found"

            container_tool_path = Path(f"/usr/local/bin/{tool.name}")
            await self.deployment.copy_to_container(
                src=local_tool_path,
                tgt=container_tool_path,
            )
            await self.communicate(f"chmod +x {container_tool_path.as_posix()}", check="raise")
            await self.communicate(f"which {tool_name}", check="raise", error_msg=f"Failed to install tool {tool_name}")
            self.commands.append(tool.command)
            self.logger.info(f"Tool {tool_name} installed")

        self.logger.info("Agent environment startup completed")
        return self

    async def close(self) -> None:
        """Shutdown SWE-ReX deployment etc."""
        self.logger.info("Beginning environment shutdown...")
        await self.deployment.stop()
        self.logger.info("Environment shutdown completed")

    def get_tool_bash_command(self, tool_call: OpenAIFunctionToolCall) -> str:
        function: OpenAIFunctionCallSchema = tool_call.function
        func_name: str = function.name
        func_params: dict = function.arguments

        if func_name in ["finish", "submit"]:
            return "echo '<<<Finished>>>'"

        if func_name == "execute_bash":
            return func_params.get("command", "")

        # Start building the command
        cmd_parts = [shlex.quote(func_name)]

        # If there's a 'command' parameter, put that next
        base_command = func_params.get("command")
        if base_command is not None:
            cmd_parts.append(shlex.quote(base_command))

        # Append all other parameters
        for param_key, param_value in func_params.items():
            if param_key == "command":
                continue

            # Safely quote the param_value
            param_value_quoted = shlex.quote(str(param_value))
            cmd_parts.append(f"--{param_key}")
            cmd_parts.append(param_value_quoted)

        return " ".join(cmd_parts)

    async def run_action(self, action_cmd: str, max_observation_length: int = 160000) -> str:
        try:
            observation = await self.communicate(input=action_cmd, timeout=self.action_timeout, check="ignore")
            if observation.strip() == "":
                observation = "Your command ran successfully and did not produce any output."
            elif observation.strip() == "<<<Finished>>>":
                observation = "<<<Finished>>>"
            elif len(observation) > max_observation_length:
                observation = (
                    f"Observation:\n{observation[:max_observation_length]}<response clipped>\n"
                    f"<NOTE>Observations should not exceeded {max_observation_length} characters. "
                    f"{max_observation_length - len(observation)} characters were elided. "
                    "Please try a different command that produces less output or "
                    "use head/tail/grep/redirect the output to a file. Do not use interactive pagers.</NOTE>"
                )
            else:
                observation = f"Observation:\n{observation}"
            return observation
        except CommandTimeoutError:
            # interrupt_session
            try:
                await self.interrupt_session()
            except Exception:
                self.logger.critical("Failed to interrupt session after command timeout")
            error_message = (
                f"The command '{action_cmd}' was cancelled because it took more than {self.action_timeout} seconds. "
                "Please try a different command that completes more quickly. Note: A common source of this error is "
                "if the command is interactive or requires user input (it is impossible to receive user input "
                "in the current environment, so the command will never complete)."
            )
            raise ActionTimeoutError(error_message) from None
        except BashIncorrectSyntaxError as e:
            # this should not happen, so add critical logs here
            self.logger.error("Action command has incorrect syntax")
            error_message = (
                "Your bash command contained syntax errors and was NOT executed. "
                "Please fix the syntax errors and try again. This can be the result "
                "of not adhering to the syntax for multi-line commands. Here is the output of `bash -n`:\n"
                f"{e.extra_info['bash_stdout']}\n{e.extra_info['bash_stderr']}"
            )
            raise ActionIncorrectSyntaxError(error_message) from None

    async def get_patch(self) -> str:
        """Get the patch from the environment"""
        command = "git add -A && git diff --no-color --cached > /root/patch.diff"
        try:
            await self.deployment.runtime.execute(
                Command(command=["bash", "-c", command], cwd=f"/{self.repo.repo_name}", check=True)
            )
            patch = await self.read_file("/root/patch.diff")
            return patch
        except Exception as e:
            self.logger.critical(f"Failed to get patch: {str(e)}")
            return ""

    async def apply_patch(
        self,
        patch: str,
        patch_file_local: Path | None = None,
        patch_file_container: Path = Path("/root/patch.diff"),
    ) -> str:
        if not patch_file_local:
            fid = str(uuid.uuid4())
            patch_file_local = Path(f"/root/patch_{fid}.diff")

        patch_file_local.write_text(patch)
        self.logger.info(f"Intermediate patch file written to {patch_file_local}")
        await self.deployment.copy_to_container(src=patch_file_local, tgt=patch_file_container)
        git_apply_commands = [
            ["git", "apply", "--whitespace=fix", patch_file_container.as_posix()],
            ["git", "apply", "--reject", "--whitespace=nowarn", patch_file_container.as_posix()],
            ["patch", "--batch", "--fuzz=5", "-p1", "-i", patch_file_container.as_posix()],
        ]
        applied_patch = False
        for git_apply_cmd in git_apply_commands:
            val = await self.deployment.runtime.execute(
                Command(command=git_apply_cmd, cwd=f"/{self.repo.repo_name}", timeout=120, user="root")
            )
            if val.exit_code == 0:
                self.logger.info("Apply Patch Successfully")
                applied_patch = True
                break
            else:
                self.logger.error(f"Failed to apply patch with command {git_apply_cmd}: {val.stdout}, {val.stderr}")
        return applied_patch

    async def evaluate(
        self,
        test_spec: SWETestSpec,
        eval_timeout: int = 300,
        test_file_local: Path | None = None,
        test_file_container: Path = Path("/root/run_tests.sh"),
        test_output_file: Path | None = None,
        report_file: Path | None = None,
    ):
        try:
            if not test_file_local:
                fid = str(uuid.uuid4())
                test_file_local = Path(f"/tmp/test_{fid}.py")

            # step 1: get test file and upload to container, and check it
            test_content = test_spec.eval_script
            # for some datasets like r2e, test_content is None as it has existed in the image
            if test_content:
                test_file_local.write_text(test_content)
                self.logger.info(f"Eval script written to {test_file_local}; copying to container...")
                await self.deployment.copy_to_container(src=test_file_local, tgt=test_file_container)
                self.logger.info(f"copy_to_container {test_file_container} successfully")

            # check if test file exists in container
            cmd_str = f"[ -f {test_file_container} ] && echo 'exists' || echo 'not exists'"
            r = await self.deployment.runtime.execute(Command(command=["bash", "-c", cmd_str]))
            if r.stdout.strip() != "exists":
                self.logger.error(f"{test_file_container} not found in container")
                raise RuntimeError(f"{test_file_container} not found in container")

            # step 2: run test file in container
            cmd_str = f"bash {test_file_container} 2>&1"
            execution_t0 = time.perf_counter()
            r = await self.deployment.runtime.execute(Command(command=["bash", "-c", cmd_str], timeout=eval_timeout))
            execution_time = time.perf_counter() - execution_t0
            output, _ = r.stdout, r.exit_code
            # Remove ANSI escape codes and \r characters
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            if test_output_file:
                test_output_file.write_text(output)
                self.logger.info(f"Test output written to {test_output_file}")

            # step 3: get eval report (parse eval logs and judge if it's correct)
            eval_report = test_spec.get_eval_report(eval_output=output)
            eval_report["execution_time"] = execution_time
            self.logger.info(f"Eval report: {eval_report}")
            if report_file:
                report_file.write_text(json.dumps(eval_report, indent=4))
                self.logger.info(f"Report written to {report_file}")
            return eval_report, True
        except Exception as e:
            self.logger.error(f"Failed to evaluate: {e}")
            return None, False

    # MARK: Helper functions #
    async def interrupt_session(self):
        self.logger.info("Interrupting session")
        await self.deployment.runtime.run_in_session(BashInterruptAction(timeout=10))

    async def communicate(
        self,
        input: str,
        timeout: int | float = 60,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        """Executes a command in the running shell. The details of this are handled by
        the SWE-ReX deployment/runtime.

        Args:
            input: input to send to container
            timeout_duration: duration to wait for output
            check: `ignore`: do not extract exit code (more stable), `warn`: extract exit code and log error if
                exit code is non-zero, `raise`: raise error if exit code is non-zero
            error_msg: error message to raise if the command fails

        Returns:
            output: output from container
        """
        self.logger.debug(f"Input:\n{input}")
        rex_check = "silent" if check else "ignore"
        r = await self.deployment.runtime.run_in_session(BashAction(command=input, timeout=timeout, check=rex_check))
        output = r.output
        self.logger.debug(f"Output:\n{output}")
        if check != "ignore" and r.exit_code != 0:
            self.logger.error(f"{error_msg}:\n{output}")
            msg = f"Command {input!r} failed ({r.exit_code=}): {error_msg}"
            if check == "raise":
                await self.close()
                raise RuntimeError(msg)
        return output

    async def read_file(self, path: str | PurePath, encoding: str | None = None, errors: str | None = None) -> str:
        """Read file contents from container

        Args:
            path: Absolute path to file
            encoding: Encoding to use when reading the file. None means default encoding.
                This is the same as the `encoding` argument of `Path.read_text()`
            errors: Error handling to use when reading the file. None means default error handling.
                This is the same as the `errors` argument of `Path.read_text()`

        Returns:
            file_contents: Contents of file as string
        """
        r = await self.deployment.runtime.read_file(ReadFileRequest(path=str(path), encoding=encoding, errors=errors))
        return r.content

    async def write_file(self, path: str | PurePath, content: str) -> None:
        """Write content to file in container"""
        await self.deployment.runtime.write_file(WriteFileRequest(path=str(path), content=content))

    async def set_env_variables(self, env_variables: dict[str, str]) -> None:
        """Set environment variables in the environment."""
        _env_setters = [f"export {k}={shlex.quote(str(v))}" for k, v in env_variables.items()]
        command = " && ".join(_env_setters)
        await self.communicate(command, check="raise")
