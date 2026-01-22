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

from recipe.swe_agent.fass_deployment import VefaasDeploymentConfig
from recipe.swe_agent.swe_scaffold import SWEAgentEnv, SWEEnvConfig

# dataset_id = "r2e-gym-subset"
# instance_id = "orange3__269e2a176b"
dataset_id = "swe-bench-verified"
instance_id = "django__django-13809"
run_id = str(uuid.uuid4())

vefaas_config = VefaasDeploymentConfig(
    type="vefaas",
    command="curl -fsSL https://pjw-test-empty.tos-cn-beijing.ivolces.com/bin/tos_swe_rex.sh | bash -s -- {token}",
    timeout=180.0,
    startup_timeout=180.0,
    dataset_id=dataset_id,
    instance_id=instance_id,
)
env_config_dict = {
    "repo": {
        "repo_name": "testbed",
        "base_commit": "bef6f7584280f1cc80e5e2d80b7ad073a93d26ec",
        "reset": True if dataset_id == "swe-bench-verified" else False,
    },
    "tools": [
        {"name": "str_replace_editor"},
        {"name": "execute_bash"},
        {"name": "submit"},
    ],
    "deployment": vefaas_config,
    "action_timeout": 120,
}
env_config = SWEEnvConfig(**env_config_dict)
env = SWEAgentEnv.from_config(env_config)
asyncio.run(env.start())


def run_command(command: str):
    output = asyncio.run(env.communicate(command))
    print("=" * 20)
    print(output)
    print("=" * 20)


breakpoint()

# print(time.time())
# try:
#     print(asyncio.run(env.communicate("sleep 300", timeout=5)))
# except Exception:
#     try:
#         asyncio.run(env.interrupt_session())
#     except Exception:
#         print("Failed to interrupt session")
# print(time.time())
# print(asyncio.run(env.communicate("echo 'hello world'", timeout=5)))
# print(time.time())
# breakpoint()
