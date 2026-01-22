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
    def __init__(self, src_path, tgt_path):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.src_path = src_path
        self.tgt_path = tgt_path

    def start_sync(self):
        if not os.path.isdir(self.src_path):
            raise RuntimeError(f"Repo path does not exist: {self.src_path}")

        try:
            result = subprocess.run(
                ["cp", "-r", self.src_path, self.tgt_path],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print(f"[SyncWorker][{self.node_id}] cp success:\n{result.stdout}")
            else:
                print(f"[SyncWorker][{self.node_id}] cp failed (code={result.returncode}):\n{result.stderr}")
        except Exception as e:
            print(f"Error syncing repo {self.repo_path}: {e}")


def sync_repo(repo_path, branch):
    node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
    print(f"Active nodes {len(node_ids)}: {node_ids}")
    tasks = []
    for node_id in node_ids:
        actor = SyncWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ).remote(repo_path, branch)
        tasks.append(actor.start_sync.remote())
    ray.get(tasks)


def main():
    ray.init()
    src_path = "/mnt/hdfs/yyding/models/Qwen3-Coder-30B-A3B-Instruct"
    tgt_path = "/opt/tiger/verl-swe"
    _ = sync_repo(src_path, tgt_path)


if __name__ == "__main__":
    main()
