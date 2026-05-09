from importlib import import_module

from .config import DeployConfig, HostDeploymentConfig, LocalDeploymentConfig, VefaasDeploymentConfig

_LAZY_EXPORTS = {
    "HostDeployment": ".host.deployment",
    "LocalDeployment": ".local.deployment",
    "VefaasDeployment": ".vefaas.deployment",
}

__all__ = [
    "DeployConfig",
    "HostDeploymentConfig",
    "LocalDeploymentConfig",
    "VefaasDeploymentConfig",
    "HostDeployment",
    "LocalDeployment",
    "VefaasDeployment",
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
