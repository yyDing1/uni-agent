# ruff: noqa: E501
import argparse
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from dataset_utils import get_swe_bench_sandbox_image_name, load_instance_ids, load_local_dataset

impl = os.getenv("DEPLOYMENT", "vefaas").lower()
if impl == "vefaas":
    from uni_agent.deployment.vefaas.deployment import get_vefaas_image_name
elif impl != "local":
    raise ValueError(f"Invalid deployment implementation: {impl}")
else:
    get_vefaas_image_name = None


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


DEFAULT_DATASET_DIR = "~/dataset/SWE-bench_Verified"
DEFAULT_SAVE_DIR = "~/dataset/verl/SWE-bench_Verified"


def get_local_image_name(instance_id: str) -> str:
    return get_swe_bench_sandbox_image_name(instance_id)


def get_image_name(instance_id: str) -> str:
    if impl == "local":
        return get_local_image_name(instance_id)

    dataset_id = "swe-bench-verified"
    return get_vefaas_image_name(dataset_id, instance_id)


def load_swe_bench_verified(dataset_dir: str, split: str) -> Dataset:
    if impl == "local":
        print(f"Loading local SWE-bench Verified dataset from {dataset_dir}...", flush=True)
        return load_local_dataset(Path(dataset_dir), split)

    data_source = "princeton-nlp/SWE-bench_Verified"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    return load_dataset(data_source, split=split)


def build_swe_bench_dataset(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    split: str = "test",
    max_instances: int | None = None,
    instance_ids: set[str] | None = None,
) -> Dataset:
    def process_swe_bench_verified(example: dict[str, Any]):
        instance_id = example["instance_id"]
        reset_cmds = [
            "cd /testbed",
            "git restore .",
            "git reset --hard",
            f"git checkout {example['base_commit']}",
            "git clean -fdq",
        ]
        reset_script = " && ".join(reset_cmds)
        sample = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(problem_statement=example["problem_statement"])},
            ],
            "agent_name": "swe_agent",
            "extra_info": {
                "tools_kwargs": {
                    "env": {
                        "image": get_image_name(instance_id),
                        "post_setup_cmd": reset_script,
                    },
                    "reward": {
                        "name": "swe_bench",
                        "metadata": dict(example),
                    },
                },
            },
        }
        return sample

    dataset = load_swe_bench_verified(dataset_dir, split)
    if instance_ids is not None:
        dataset = dataset.filter(lambda example: example["instance_id"] in instance_ids)
    if max_instances is not None:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
    return dataset.map(process_swe_bench_verified, remove_columns=dataset.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--local-save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--output-name", default="swe_bench_verified.parquet")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--instance-ids-file", default=None)

    args = parser.parse_args()

    instance_ids = load_instance_ids(args.instance_ids_file)
    sbv_dataset = build_swe_bench_dataset(
        dataset_dir=args.dataset_dir,
        split=args.split,
        max_instances=args.max_instances,
        instance_ids=instance_ids,
    )

    save_dir = Path(args.local_save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / args.output_name
    sbv_dataset.to_parquet(str(output_path))
    print(f"Saved {len(sbv_dataset)} examples to {output_path}")
