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
import json

from datasets import load_dataset


def build_swe_bench_verified():
    def process_swe_bench_verified(example):
        sample = {
            "prompt": [{"role": "user", "content": "<NOT USED>"}],
            "agent_name": "swe_agent",
            "extra_info": {
                "tools_kwargs": {
                    "dataset_id": "swe-bench-verified",
                    "instance_id": example["instance_id"],
                    "metadata": example,
                },
            },
        }
        return sample

    data_source = "princeton-nlp/SWE-bench_Verified"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    dataset = dataset.map(process_swe_bench_verified, remove_columns=dataset.column_names)
    return dataset


def build_swe_bench():
    def process_swe_bench(example):
        sample = {
            "prompt": [{"role": "user", "content": "<NOT USED>"}],
            "agent_name": "swe_agent",
            "extra_info": {
                "tools_kwargs": {
                    "dataset_id": "swe-bench",
                    "instance_id": example["instance_id"],
                    "metadata": example,
                },
            },
        }
        return sample

    data_source = "princeton-nlp/SWE-bench"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    dataset = dataset.map(process_swe_bench, remove_columns=dataset.column_names)
    return dataset


def build_swe_bench_live():
    def process_swe_bench_live(example):
        sample = {
            "prompt": [{"role": "user", "content": "<NOT USED>"}],
            "agent_name": "swe_agent",
            "extra_info": {
                "tools_kwargs": {
                    "dataset_id": "swe-bench-live",
                    "instance_id": example["instance_id"],
                    "metadata": example,
                },
            },
        }
        return sample

    def get_fass_example_list():
        instance_list = "/mnt/hdfs/yyding/data/swe-agent/swe-bench-live-instance-ids.txt"
        with open(instance_list) as f:
            instance_ids = f.readlines()
        instance_ids = [line.strip() for line in instance_ids]
        return instance_ids

    data_source = "SWE-bench-Live/SWE-bench-Live"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="full")
    fass_example_list = get_fass_example_list()
    dataset = dataset.filter(lambda example: example["instance_id"] in fass_example_list)
    dataset = dataset.map(process_swe_bench_live, remove_columns=dataset.column_names)
    return dataset


def build_r2e_gym_subset():
    from r2egym.commit_models.diff_classes import ParsedCommit

    def convert_to_standard_format(example):
        repo_name = example["repo_name"]
        base_commit = example["commit_hash"]
        instance_id = f"{repo_name}__{base_commit[:10]}"
        patch = ParsedCommit(**json.loads(example["parsed_commit_content"])).get_patch()
        problem_statement = example["problem_statement"]
        expected_output_json = example["expected_output_json"]
        return {
            "repo": repo_name,
            "instance_id": instance_id,
            "patch": patch,
            "problem_statement": problem_statement,
            "expected_output_json": expected_output_json,
        }

    def process_r2e_gym_subset(example):
        sample = {
            "prompt": [{"role": "user", "content": "<NOT USED>"}],
            "agent_name": "swe_agent",
            "extra_info": {
                "tools_kwargs": {
                    "dataset_id": "r2e-gym-subset",
                    "instance_id": example["instance_id"],
                    "metadata": example,
                },
            },
        }
        return sample

    data_source = "R2E-Gym/R2E-Gym-Subset"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    dataset = dataset.map(convert_to_standard_format, remove_columns=dataset.column_names)
    dataset = dataset.map(process_r2e_gym_subset, remove_columns=dataset.column_names)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/swe_agent")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # swe_bench_verified_dataset = build_swe_bench_verified()
    # swe_bench_verified_dataset.to_parquet(f"{args.local_dir}/swe_bench_verified.parquet")

    # swe_bench_dataset = build_swe_bench()
    # verified_instance_id = [
    #     example["extra_info"]["tools_kwargs"]["instance_id"] for example in swe_bench_verified_dataset
    # ]
    # swe_bench_dataset = swe_bench_dataset.filter(
    #     lambda example: example["extra_info"]["tools_kwargs"]["instance_id"] not in verified_instance_id
    # )
    # swe_bench_dataset.to_parquet(f"{args.local_dir}/swe_bench_test_wo_verified.parquet")

    # swe_bench_live = build_swe_bench_live()
    # swe_bench_live.to_parquet(f"{args.local_dir}/swe_bench_live_in_vefass.parquet")

    r2e_gym_subset = build_r2e_gym_subset()
    r2e_gym_subset.to_parquet(f"{args.local_dir}/r2e_gym_subset.parquet")
