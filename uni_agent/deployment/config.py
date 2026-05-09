from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class HostDeploymentConfig(BaseModel):
    """Configuration for host-local execution (no container)."""

    type: Literal["host"] = "host"
    """Discriminator for (de)serialization. Do not change."""
    timeout: float = 60.0
    """Default timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout for the initial bash session handshake.

    During parameter-sync weight reloads, fork()/exec() and even the asyncio event loop can be
    starved for tens of seconds.
    """

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .host.deployment import HostDeployment

        return HostDeployment.from_config(self, run_id)


class LocalDeploymentConfig(BaseModel):
    """Configuration for a local Docker/Podman sandbox."""

    image: str = "python:3.12"
    """Container image used for the sandbox."""
    command: str = "python3 -m pip install -q swerex && python3 -m swerex.server --auth-token {token}"
    """Command to run inside the sandbox."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 180.0
    """Timeout waiting for runtime to start."""
    container_runtime: str = "docker"
    """Container runtime executable, typically docker or podman."""
    container_name: str | None = None
    """Optional container name override."""
    host: str | None = None
    """Override the runtime host. Defaults to localhost outside containers and container IP inside containers."""
    published_port: int | None = None
    """Host port mapped to the sandbox runtime port. If unset, a free local port is chosen."""
    runtime_port: int = 8000
    """Port exposed by the swerex server inside the sandbox."""
    network: str | None = None
    """Optional Docker network to attach the sandbox to."""
    shell: str = "/bin/bash"
    """Shell executable used as the container entrypoint."""
    extra_run_args: list[str] = Field(default_factory=list)
    """Extra args appended to the container runtime `run` command."""

    type: Literal["local"] = "local"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .local.deployment import LocalDeployment

        return LocalDeployment.from_config(self, run_id)


class VefaasDeploymentConfig(BaseModel):
    """Configuration for VEFAAS deployment."""

    image: str | None = None
    """Docker image to use for the sandbox."""
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
    proxy: str | None = None
    """Proxy to use for the connection."""

    type: Literal["vefaas"] = "vefaas"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .vefaas.deployment import VefaasDeployment

        return VefaasDeployment.from_config(self, run_id)


DeployConfig: TypeAlias = Annotated[
    VefaasDeploymentConfig | LocalDeploymentConfig | HostDeploymentConfig,
    Field(discriminator="type"),
]
