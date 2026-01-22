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

import os
import subprocess

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote
class SyncWorker:
    def __init__(self, repo_path, branch="swe_agent_new"):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.repo_path = repo_path
        self.branch = branch

    def start_sync(self):
        if not os.path.isdir(self.repo_path):
            raise RuntimeError(f"Repo path does not exist: {self.repo_path}")

        try:
            result = subprocess.run(
                ["git", "pull", "origin", self.branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print(f"[GitSyncWorker][{self.node_id}] git pull success:\n{result.stdout}")
            else:
                print(f"[GitSyncWorker][{self.node_id}] git pull failed (code={result.returncode}):\n{result.stderr}")
        except Exception as e:
            print(f"Error syncing repo {self.repo_path}: {e}")


def sync_repo(repo_path, branch):
    node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
    print(f"Active nodes: {node_ids}")
    for node_id in node_ids:
        actor = SyncWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ).remote(repo_path, branch)
        ray.get(actor.start_sync.remote())


def main():
    ray.init()
    local_output_path = "/opt/tiger/verl-swe"
    shared_storage = "swe"
    _ = sync_repo(local_output_path, shared_storage)


if __name__ == "__main__":
    main()
