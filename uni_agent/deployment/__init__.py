from typing import Annotated, TypeAlias

from pydantic import Field

from .host.deployment import HostDeploymentConfig
from .local.deployment import LocalDeploymentConfig
from .vefaas.deployment import VefaasDeploymentConfig

DeployConfig: TypeAlias = Annotated[
    VefaasDeploymentConfig | LocalDeploymentConfig | HostDeploymentConfig,
    Field(discriminator="type"),
]

__all__ = [
    "DeployConfig",
    "HostDeploymentConfig",
    "LocalDeploymentConfig",
    "VefaasDeploymentConfig",
]
