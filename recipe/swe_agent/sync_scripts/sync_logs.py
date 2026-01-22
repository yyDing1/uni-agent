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

import argparse
import os
import shutil
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote
class SyncWorker:
    def __init__(self, src_path, tgt_path):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.node_id = ray.get_runtime_context().get_node_id()

    def start_sync(self, interval=30):
        print(f"LogSyncWorker {self.node_id} started")
        os.makedirs(self.tgt_path, exist_ok=True)

        while True:
            try:
                if os.path.exists(self.src_path):
                    shutil.copytree(
                        self.src_path,
                        self.tgt_path,
                        dirs_exist_ok=True,
                        symlinks=False,
                    )
                    print(f"Successfully synced logs from {self.src_path} to {self.tgt_path}")
                else:
                    print(f"Source path {self.src_path} does not exist, skipping sync")
            except Exception as e:
                print(f"Error syncing logs from {self.src_path} to {self.tgt_path}: {e}")
            time.sleep(interval)


def deploy_log_syncer(local_output_path, shared_hdfs_path):
    node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
    print(f"Active nodes: {node_ids}")
    syncers = []
    for node_id in node_ids:
        actor = SyncWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ).remote(local_output_path, shared_hdfs_path)
        actor.start_sync.remote(interval=10)
        syncers.append(actor)
    return syncers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None)
    parser.add_argument("--dst", type=str, default=None)
    args = parser.parse_args()
    ray.init()
    _ = deploy_log_syncer(args.src, args.dst)
    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        print("Log syncing stopped")


if __name__ == "__main__":
    main()
