import asyncio
from typing import Self

from swerex.deployment.local import LocalDeployment as SwerexLocalDeployment
from swerex.runtime.abstract import CreateBashSessionRequest

from uni_agent.deployment.config import LocalDeploymentConfig


class LocalDeployment(SwerexLocalDeployment):
    """SWE-ReX local deployment with Uni-Agent retry startup."""

    @classmethod
    def from_config(cls, config: LocalDeploymentConfig, run_id: str | None = None) -> Self:
        return cls(**config.model_dump())

    async def start(self, max_retries: int = 5) -> None:
        last_error: Exception | None = None
        for retry in range(max_retries):
            try:
                await super().start()
                await self.runtime.create_session(CreateBashSessionRequest())
                return
            except Exception as exc:
                last_error = exc
                await self.stop()
                if retry < max_retries - 1:
                    sleep_time = min(30, 2**retry)
                    self.logger.info(f"Retrying local deployment startup in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Failed to start local deployment after {max_retries} retries") from last_error
