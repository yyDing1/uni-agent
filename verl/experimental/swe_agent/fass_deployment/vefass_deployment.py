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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Self

import volcenginesdkcore
import volcenginesdkvefaas
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import BashAction, Command, CreateBashSessionRequest, IsAliveResponse, UploadRequest
from swerex.utils.wait import _wait_until_alive
from volcenginesdkcore.rest import ApiException

from verl.experimental.swe_agent.async_logging import get_logger

from .vefass_remote_runtime import RemoteRuntime, RemoteRuntimeConfig

PUB_VOLCES_IMG_URL_TEMPLATE = {
    "swe-bench": (
        "enterprise-public-cn-beijing.cr.volces.com"
        "/swe-bench/sweb.eval.x86_64.{project_name}_1776_{instance_number}:latest"
    ),
    "swe-bench-verified": (
        "enterprise-public-cn-beijing.cr.volces.com"
        "/swe-bench-verified/sweb.eval.x86_64.{project_name}_1776_{instance_number}:latest"
    ),
    "swe-bench-live": (
        "enterprise-public-cn-beijing.cr.volces.com/swe-bench-live/{project_name}_1776_{instance_number}:latest"
    ),
    "r2e-gym-subset": ("enterprise-public-cn-beijing.cr.volces.com/r2e-gym-subset/{instance_number}:latest"),
}


def get_vefass_image_name(dataset_id: str, instance_id: str) -> str:
    assert dataset_id in PUB_VOLCES_IMG_URL_TEMPLATE, (
        f"only support {list(PUB_VOLCES_IMG_URL_TEMPLATE.keys())}, got {dataset_id}"
    )
    parts = instance_id.split("__")
    assert len(parts) == 2
    project_name = parts[0].lower()
    instance_number = parts[1].lower()

    if dataset_id in ["swe-bench", "swe-bench-verified", "swe-bench-live"]:
        return PUB_VOLCES_IMG_URL_TEMPLATE[dataset_id].format(
            project_name=project_name,
            instance_number=instance_number,
        )
    elif dataset_id == "r2e-gym-subset":
        return PUB_VOLCES_IMG_URL_TEMPLATE[dataset_id].format(instance_number=instance_number)
    else:
        assert dataset_id in PUB_VOLCES_IMG_URL_TEMPLATE, (
            f"only support {list(PUB_VOLCES_IMG_URL_TEMPLATE.keys())}, got {dataset_id}"
        )


class VefaasDeploymentConfig(BaseModel):
    """Configuration for VEFAAS deployment."""

    dataset_id: str
    """Dataset ID for the SWE-Bench instance."""
    instance_id: str
    """Run ID for the current instance."""
    command: str = "python3 -m swerex.server --auth-token {token}"
    """Command to run in the sandbox with authentication token."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout waiting for runtime to start."""
    function_id: str | None = None
    """VEFAAS function ID."""
    function_route: str | None = None
    """VEFAAS function Route."""
    image: str | None = None
    """Docker image to use for the sandbox."""

    type: Literal["vefaas"] = "vefaas"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        return VefaasDeployment.from_config(self, run_id)


class VefaasDeployment(AbstractDeployment):
    def __init__(self, run_id: str, **kwargs: Any):
        load_dotenv()
        self.run_id = run_id
        self._config = VefaasDeploymentConfig(**kwargs)
        self._runtime: RemoteRuntime | None = None
        self.logger = get_logger("rex-deploy", run_id)
        self._hooks = CombinedDeploymentHook()
        self._sandbox_id: str | None = None
        self._stopped: bool = False

        self._thread_pool = ThreadPoolExecutor(max_workers=20)

        access_key = os.getenv("VOLCE_ACCESS_KEY") or os.getenv("VOLCENGINE_ACCESS_KEY")
        secret_key = os.getenv("VOLCE_SECRET_KEY") or os.getenv("VOLCENGINE_SECRET_KEY")
        region = os.getenv("VEFAAS_REGION", "cn-beijing")
        if not all([access_key, secret_key, region]):
            raise ValueError("VOLCE_ACCESS_KEY, VOLCE_SECRET_KEY, and VEFAAS_REGION must be set")
        self._vefaas_client = get_vefaas_client(access_key, secret_key, region)

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: VefaasDeploymentConfig, run_id: str | None = None) -> Self:
        if not run_id:
            run_id = str(uuid.uuid4())

        return cls(run_id=run_id, **config.model_dump())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float = 10.0):
        try:
            return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=0.5)
        except TimeoutError as e:
            self.logger.error("Runtime did not start within timeout.")
            await self.stop()
            raise e

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    async def start(self, max_retries: int = 5):
        self.logger.info(
            f"Starting vefaas deployment,"
            f"function_id = {self._config.function_id}, instance_id = {self._config.instance_id}"
        )
        function_id = self._config.function_id or os.getenv("VEFAAS_FUNCTION_ID")
        if not function_id:
            raise ValueError("VEFAAS_FUNCTION_ID environment variable not set")

        image = self._config.image or get_vefass_image_name(
            dataset_id=self._config.dataset_id, instance_id=self._config.instance_id
        )

        if not image:
            raise ValueError("No image specified and no image list provided")

        token = self._get_token()
        command = self._config.command.format(token=token)

        self.logger.info(f"Creating sandbox with image {image}, command = {command}")
        self._hooks.on_custom_step("Creating vefaas sandbox")
        loop = asyncio.get_running_loop()
        create_sanbox_done = False

        for retry in range(max_retries):
            try:
                self._sandbox_id = await loop.run_in_executor(
                    self._thread_pool,
                    create_sandbox,
                    self._vefaas_client,
                    function_id,
                    image,
                    command,
                    self.logger,
                )
                if self._sandbox_id:
                    create_sanbox_done = True
                    break
            except Exception as e:
                self.logger.error(f"Failed to create sandbox: {e}")
                sleep_time = min(30, 2**retry)
                self.logger.info(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)

        if not create_sanbox_done:
            raise RuntimeError(f"Failed to create sandbox after {max_retries} retries")

        self.logger.info(f"Sandbox {self._sandbox_id} created")
        self._hooks.on_custom_step("Starting runtime")

        function_route = self._config.function_route or os.getenv("VEFAAS_FUNCTION_ROUTE")
        if not function_route:
            raise ValueError("VEFAAS_FUNCTION_ROUTE environment variable not set")

        runtime_config = RemoteRuntimeConfig(
            base_url=function_route,
            extra_params={"faasInstanceName": self._sandbox_id},
            auth_token=token,
            timeout=self._config.timeout,
        )
        self._runtime = RemoteRuntime.from_config(runtime_config, run_id=self.run_id)

        # await self._wait_until_alive(timeout=self._config.startup_timeout)
        await self.runtime.create_session(
            CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=10)
        )
        await self._post_setup()

    async def copy_to_container(self, src: Path, tgt: Path):
        # Make directory if necessary
        await self._runtime.execute(Command(command=["mkdir", "-p", str(tgt.parent)]))

        # Upload file to container
        await self._runtime.upload(UploadRequest(source_path=str(src), target_path=str(tgt)))

    async def _post_setup(self):
        def get_setup_script_path(dataset_id: str) -> str:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            setup_script_path = os.path.join(current_dir, f"setup_scripts/{dataset_id}.sh")
            if not os.path.exists(setup_script_path):
                raise ValueError(f"Setup script not found: {setup_script_path}")
            return setup_script_path

        # upload setup script
        dataset_id = self._config.dataset_id
        setup_script_path = get_setup_script_path(dataset_id)
        self.logger.info(f"Uploading setup script {setup_script_path} to /root/post_setup.sh")
        await self.copy_to_container(src=Path(setup_script_path), tgt=Path("/root/post_setup.sh"))
        self.logger.info("Running post_setup.sh ...")
        await self._runtime.run_in_session(BashAction(command="source /root/post_setup.sh"))
        self.logger.info("post_setup.sh completed")

    async def stop(self):
        # Prevent duplicate stops
        if getattr(self, "_stopped", False):
            return

        if self._runtime:
            try:
                await self._runtime.close()
            except Exception as e:
                self.logger.error(f"Failed to close vefass runtime within timeout: {e}")
            self._runtime = None

        if self._sandbox_id:
            self.logger.info(f"Deleting sandbox {self._sandbox_id}")
            function_id = self._config.function_id or os.getenv("VEFAAS_FUNCTION_ID")
            if not function_id:
                self.logger.error("VEFAAS_FUNCTION_ID not set, cannot delete sandbox")
                return

            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    delete_sandbox,
                    self._vefaas_client,
                    function_id,
                    self._sandbox_id,
                    self.logger,
                )
                self.logger.info(f"Sandbox {self._sandbox_id} deleted")
            except Exception as e:
                self.logger.error(f"Failed to delete sandbox {self._sandbox_id}: {e}")
            finally:
                self._sandbox_id = None

        self._stopped = True

    @property
    def runtime(self) -> RemoteRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def __del__(self):
        if hasattr(self, "_sandbox_id") and self._sandbox_id and not getattr(self, "_stopped", False):
            msg = "Ensuring vefaas deployment is stopped because object is deleted"
            try:
                self.logger.debug(msg)
            except Exception:
                print(msg)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.stop())
                else:
                    loop.run_until_complete(self.stop())
            except Exception:
                pass
        # 关闭线程池，避免资源泄露
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=False)
        # Mark as stopped to prevent duplicate cleanup
        self._stopped = True


def get_vefaas_client(access_key: str, secret_key: str, region: str) -> volcenginesdkvefaas.VEFAASApi:
    configuration = volcenginesdkcore.Configuration()
    configuration.ak = access_key
    configuration.sk = secret_key
    configuration.read_timeout = 40
    configuration.connect_timeout = 40
    configuration.auto_retry = False
    configuration.region = region
    configuration.client_side_validation = True
    configuration.proxy = "http://sys-proxy-rd-relay.byted.org:8118"
    api_client = volcenginesdkcore.ApiClient(configuration)
    return volcenginesdkvefaas.VEFAASApi(api_client)


def create_sandbox(
    client: volcenginesdkvefaas.VEFAASApi,
    function_id: str,
    image: str,
    command: str,
    logger: logging.Logger,
) -> str | None:
    if image.startswith("swebench/"):
        image_name = image.replace("swebench/", "", 1)
        image = f"enterprise-public-cn-beijing.cr.volces.com/swe-bench/{image_name}"

    instance_image_info = volcenginesdkvefaas.InstanceImageInfoForCreateSandboxInput(
        image=image,
        port=8000,  # swerex server port
        command=command,
    )
    start_time = time.time()
    try:
        resp = client.create_sandbox(
            volcenginesdkvefaas.CreateSandboxRequest(
                function_id=function_id,
                instance_image_info=instance_image_info,
                timeout=300,
            )
        )
        end_time = time.time()
        logger.info(f"Sandbox {resp.sandbox_id} created in {end_time - start_time:.2f}s")
        return resp.sandbox_id
    except Exception as e:
        end_time = time.time()
        logger.error(f"Sandbox creation for {image} failed in {end_time - start_time:.2f}s: {e}")
        return None


def delete_sandbox(
    client: volcenginesdkvefaas.VEFAASApi,
    function_id: str,
    sandbox_id: str,
    logger: logging.Logger,
):
    if sandbox_id is None:
        return
    try:
        client.kill_sandbox(
            volcenginesdkvefaas.KillSandboxRequest(
                function_id=function_id,
                sandbox_id=sandbox_id,
            )
        )
    except ApiException as e:
        logger.error(f"Exception when deleting sandbox {sandbox_id}: {e}")
