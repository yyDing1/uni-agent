from typing import Annotated, TypeAlias

from pydantic import Field

from .host.deployment import HostDeploymentConfig
from .local.deployment import LocalDeploymentConfig

try:
    from .vefaas.deployment import VefaasDeploymentConfig
except ModuleNotFoundError as exc:
    if exc.name not in {"volcenginesdkcore", "volcenginesdkvefaas"}:
        raise
    VefaasDeploymentConfig = None

if VefaasDeploymentConfig is None:
    DeployConfig: TypeAlias = Annotated[
        LocalDeploymentConfig | HostDeploymentConfig,
        Field(discriminator="type"),
    ]
else:
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
