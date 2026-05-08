# ruff: noqa: E501
import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from dataset_utils import get_r2e_gym_sandbox_image_name, load_local_dataset
from r2egym.commit_models.diff_classes import ParsedCommit

impl = os.getenv("DEPLOYMENT", "vefaas").lower()
if impl == "vefaas":
    PUB_VOLCES_IMG_URL_R2E = "enterprise-public-cn-beijing.cr.volces.com/r2e-gym-subset/{instance_number}:latest"

    def get_image_name(dataset_id: str, instance_id: str) -> str:
        assert dataset_id == "r2e-gym-subset"
        parts = instance_id.split("__")
        assert len(parts) == 2
        instance_number = parts[1].lower()
        return PUB_VOLCES_IMG_URL_R2E.format(instance_number=instance_number)
elif impl == "local":
    LOCAL_R2E_IMAGE_TEMPLATE = os.getenv("LOCAL_R2E_IMAGE_TEMPLATE", "r2e-gym-subset/{instance_number}:latest")

    def get_image_name(dataset_id: str, instance_id: str) -> str:
        assert dataset_id == "r2e-gym-subset"
        return get_r2e_gym_sandbox_image_name(instance_id, image_template=LOCAL_R2E_IMAGE_TEMPLATE)
else:
    raise ValueError(f"Invalid deployment implementation: {impl}")


SYSTEM_PROMPT = """
You are a helpful assistant that can interact with a computer to solve tasks.
""".strip()

USER_PROMPT = """
<uploaded_files>
/testbed
</uploaded_files>
I have uploaded a python code repository in the /testbed directory. You can explore and modify files using the available tools. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository to fix the <issue_description>?
I have already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /testbed directory to ensure the <issue_description> is satisfied.

Follow these steps to resolve the issue:
1. First, explore the codebase to locate and understand the code relevant to the <issue_description>. 
- Use efficient search commands to identify key files and functions.  
- You should err on the side of caution and look at various relevant files and build your understanding of 
    - how the code works
    - what are the expected behaviors and edge cases
    - what are the potential root causes for the given issue

2. Assess whether you can reproduce the issue:
- Create a script at '/testbed/reproduce_issue.py' that demonstrates the error.
- Execute this script to confirm the error behavior.
- You should reproduce the issue before fixing it.
- Your reproduction script should also assert the expected behavior for the fixed code. 

3. Analyze the root cause:
- Identify the underlying problem based on your code exploration and reproduction results.
- Critically analyze different potential approaches to fix the issue. 
- You NEED to explicitly reason about multiple approaches to fix the issue. Next, find the most elegant and effective solution among them considering the tradeoffs (correctness, generality, side effects, etc.).
- You would need to reason about execution paths, edge cases, and other potential issues. You should look at the unit tests to understand the expected behavior of the relevant code.

4. Implement your solution:
- Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.
- You should be thorough and methodical.

5. Verify your solution:
- Rerun your reproduction script to confirm the error is fixed.
- If verification fails, iterate on your solution until successful. If you identify the reproduction script is buggy, adjust it as needed.

6. Run unit tests:
- Find and run the relevant unit tests relevant to the performed fix.
- You should run the unit tests to ensure your solution is correct and does not cause any regressions.
- In cases where the unit tests are do not pass, you should consider whether the unit tests does not reflect the *new* expected behavior of the code. If so, you can test it by writing additional edge test cases.
- Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
    - `python -m pytest -xvs sympy/physics/units/tests/test_dimensions_transcendental.py`
    - `python -m pytest tests/test_domain_py.py::test_pymethod_options`
    - `./tests/runtests.py constraints.tests.CheckConstraintTests -v 2`
- RUN ALL relevant unit tests to ensure your solution is correct and does not cause any regressions.
- DO NOT MODIFY any of the existing unit tests. You can add new edge test cases in a separate file if needed BUT DO NOT MODIFY THE EXISTING TESTS.

7. Test edge cases:
- Identify potential edge cases that might challenge your solution.
- Create additional test cases in a separate file '/testbed/edge_case_tests.py'.
- Execute these tests to verify your solution's robustness.
- You should run multiple rounds of edge cases. When creating edge cases:
    - Consider complex scenarios beyond the original issue description
    - Test for regressions to ensure existing functionality remains intact
    - At each round you should write multiple edge test cases in the same file to be efficient

8. Refine if necessary:
- If edge case testing reveals issues, refine your solution accordingly.
- Ensure your final implementation handles all identified scenarios correctly.
- Document any assumptions or limitations of your solution.

9. Submit your solution:
- Once you have verified your solution, submit your solution using the `submit` tool.

A successful resolution means:
- The specific error/issue described no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled
""".strip()


POST_SETUP_CMD = """
export PIP_CACHE_DIR=~/.cache/pip
export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

ln -s /testbed/.venv /root/.venv
ln -s /testbed/.venv/bin/python /root/.local/bin/python
ln -s /testbed/.venv/bin/python /root/.local/bin/python3
find "/testbed/.venv/bin" -type f -executable -exec ln -sf {} "/root/.local/bin/" \\;

find . -name '*.pyc' -delete
find . -name '__pycache__' -exec rm -rf {} +
find /r2e_tests -name '*.pyc' -delete
find /r2e_tests -name '__pycache__' -exec rm -rf {} +

mv /testbed/run_tests.sh /root/run_tests.sh
mv /testbed/r2e_tests /root/r2e_tests

mv /r2e_tests /root/r2e_tests
ln -s /root/r2e_tests /testbed/r2e_tests
""".strip()


DEFAULT_DATASET_DIR = "~/dataset/r2e-gym-subset-filtered"
DEFAULT_SAVE_DIR = "~/dataset/verl/r2e-gym-subset-filtered"

def get_patch(parsed_commit_content: str) -> str:
    return ParsedCommit(**json.loads(parsed_commit_content)).get_patch()


def load_r2e_gym_subset_filtered(dataset_dir: str, split: str) -> Dataset:
    if impl == "local":
        print(f"Loading local r2e-gym subset filtered dataset from {dataset_dir}...", flush=True)
        return load_local_dataset(Path(dataset_dir), split)

    data_source = "dyyyyyyyy/r2e-gym-subset-filtered"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    return load_dataset(data_source, split=split)


def build_r2e_gym_dataset(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    split: str = "train",
    max_instances: int | None = None,
) -> Dataset:

    def process_r2e_gym_subset(example: dict[str, Any]):
        dataset_id = "r2e-gym-subset"
        repo_name = example["repo_name"]
        base_commit = example["commit_hash"]
        instance_id = f"{repo_name}__{base_commit[:10]}"
        problem_statement = example["problem_statement"]
        image_name = get_image_name(dataset_id, instance_id)
        metadata = {
            "repo": repo_name,
            "instance_id": instance_id,
            "patch": get_patch(example["parsed_commit_content"]),
            "expected_output_json": example["expected_output_json"],
        }
        sample = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(problem_statement=problem_statement)},
            ],
            "agent_name": "swe_agent",
            "extra_info": {
                "tools_kwargs": {
                    "env": {
                        "image": image_name,
                        "post_setup_cmd": POST_SETUP_CMD,
                    },
                    "reward": {
                        "name": "r2e_gym",
                        "metadata": metadata,
                    },
                },
            },
        }
        return sample

    dataset = load_r2e_gym_subset_filtered(dataset_dir, split)
    if max_instances is not None:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
    return dataset.map(process_r2e_gym_subset, remove_columns=dataset.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--local-save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--output-name", default="r2e_gym_subset_filtered.parquet")
    parser.add_argument("--max-instances", type=int, default=None)

    args = parser.parse_args()

    sbv_dataset = build_r2e_gym_dataset(
        dataset_dir=args.dataset_dir,
        split=args.split,
        max_instances=args.max_instances,
    )

    save_dir = Path(args.local_save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / args.output_name
    sbv_dataset.to_parquet(str(output_path))
    print(f"Saved {len(sbv_dataset)} examples to {output_path}")
