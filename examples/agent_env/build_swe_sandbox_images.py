#!/usr/bin/env python3
# ruff: noqa: E501
"""Build local SWE-bench sandbox images from a raw Hugging Face dataset directory."""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import docker
from swebench.harness.docker_build import build_instance_images
from swebench.harness.prepare_images import filter_dataset_to_build
from swebench.harness.utils import load_swebench_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_preprocess"))
from dataset_utils import get_swe_bench_sandbox_image_name, load_instance_ids, load_local_dataset_as_dicts


DEFAULT_DATASET_DIR = "~/dataset/SWE-bench_Verified"
DEFAULT_SWE_REX_VERSION = "1.4.0"
DEFAULT_SANDBOX_PYTHON = "/usr/bin/python3"

REQUIRED_SWEBENCH_KEYS = {
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "patch",
    "test_patch",
}


def load_raw_instances_from_dir(dataset_dir: str, split: str) -> list[dict[str, Any]]:
    path = Path(dataset_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(
            "Only raw Hugging Face dataset directories are supported. "
            "Pass the directory downloaded from Hugging Face, not a converted parquet file."
        )
    return load_local_dataset_as_dicts(str(path), split)


def select_instances(
    rows: list[dict[str, Any]],
    *,
    max_instances: int | None,
    instance_ids: set[str] | None,
) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for row in rows:
        if "extra_info" in row and "instance_id" not in row:
            raise ValueError(
                "Converted Uni-Agent datasets are not supported for building sandbox images. "
                "Pass the raw Hugging Face SWE-bench dataset directory instead."
            )

        instance_id = row.get("instance_id")
        if instance_id is None:
            raise ValueError("Expected raw SWE-bench row with top-level instance_id.")
        if instance_ids is not None and instance_id not in instance_ids:
            continue
        if instance_id in seen:
            continue
        selected.append(row)
        seen.add(instance_id)
        if max_instances is not None and len(selected) >= max_instances:
            break
    return selected


def missing_required_keys(instances: list[dict[str, Any]]) -> dict[str, list[str]]:
    missing = {}
    for instance in instances:
        missing_keys = sorted(key for key in REQUIRED_SWEBENCH_KEYS if key not in instance)
        if missing_keys:
            missing[instance.get("instance_id", "<unknown>")] = missing_keys
    return missing


def write_jsonl(instances: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for instance in instances:
            f.write(json.dumps(instance, ensure_ascii=True) + "\n")


def list_existing_images() -> set[str]:
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return set(result.stdout.splitlines())


def image_has_swe_rex(image: str, *, version: str, python_bin: str) -> bool:
    check_code = (
        "import importlib.metadata as metadata, sys; "
        "installed = metadata.version('swe-rex'); "
        "raise SystemExit(0 if installed == sys.argv[1] else 1)"
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            python_bin,
            image,
            "-c",
            check_code,
            version,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def install_swe_rex_into_image(
    image: str,
    *,
    version: str,
    python_bin: str,
    docker_build_network: str | None,
    build_http_proxy: str | None,
    build_no_proxy: str | None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="swe_rex_layer_") as tmpdir:
        dockerfile = Path(tmpdir) / "Dockerfile"
        dockerfile.write_text(
            "\n".join(
                [
                    f"FROM {image}",
                    "ARG http_proxy",
                    "ARG https_proxy",
                    "ARG HTTP_PROXY",
                    "ARG HTTPS_PROXY",
                    "ARG no_proxy",
                    "ARG NO_PROXY",
                    "ENV http_proxy=${http_proxy}",
                    "ENV https_proxy=${https_proxy}",
                    "ENV HTTP_PROXY=${HTTP_PROXY}",
                    "ENV HTTPS_PROXY=${HTTPS_PROXY}",
                    "ENV no_proxy=${no_proxy}",
                    "ENV NO_PROXY=${NO_PROXY}",
                    "ENV PIP_PROGRESS_BAR=off",
                    "RUN set -eux; \\",
                    f"    attempt=1; until {python_bin} -m pip install --no-cache-dir -q swe-rex=={version}; do \\",
                    '        if [ "$attempt" -ge 5 ]; then exit 1; fi; \\',
                    '        sleep $((attempt * 10)); \\',
                    '        attempt=$((attempt + 1)); \\',
                    "    done; \\",
                    f"    {python_bin} -c \"import swerex\"",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        command = ["docker", "build", "-t", image]
        if docker_build_network:
            command.extend(["--network", docker_build_network])
        if build_http_proxy:
            command.extend(
                [
                    "--build-arg",
                    f"http_proxy={build_http_proxy}",
                    "--build-arg",
                    f"https_proxy={build_http_proxy}",
                    "--build-arg",
                    f"HTTP_PROXY={build_http_proxy}",
                    "--build-arg",
                    f"HTTPS_PROXY={build_http_proxy}",
                ]
            )
        if build_no_proxy:
            command.extend(
                [
                    "--build-arg",
                    f"no_proxy={build_no_proxy}",
                    "--build-arg",
                    f"NO_PROXY={build_no_proxy}",
                ]
            )
        command.append(tmpdir)

        print(f"Installing swe-rex=={version} into image {image}")
        subprocess.run(command, check=True)


def ensure_swe_rex_in_images(
    images: list[str],
    *,
    version: str,
    python_bin: str,
    docker_build_network: str | None,
    build_http_proxy: str | None,
    build_no_proxy: str | None,
) -> int:
    failed = []
    for image in images:
        if image_has_swe_rex(image, version=version, python_bin=python_bin):
            print(f"Image already has swe-rex=={version}: {image}")
            continue
        try:
            install_swe_rex_into_image(
                image,
                version=version,
                python_bin=python_bin,
                docker_build_network=docker_build_network,
                build_http_proxy=build_http_proxy,
                build_no_proxy=build_no_proxy,
            )
        except subprocess.CalledProcessError as exc:
            failed.append(image)
            print(f"Failed to install swe-rex into {image}: {exc}", file=sys.stderr)

    if failed:
        print(f"Failed to install swe-rex into {len(failed)} image(s):", file=sys.stderr)
        for image in failed:
            print(f"  {image}", file=sys.stderr)
        return 1
    return 0


def instance_image_name(
    instance: dict[str, Any],
    *,
    namespace: str | None,
    tag: str | None,
    env_image_tag: str | None,
) -> str:
    del env_image_tag
    return get_swe_bench_sandbox_image_name(instance["instance_id"], namespace=namespace, tag=tag or "latest")


def build_prepare_images_command(
    *,
    dataset_name: str,
    split: str,
    instance_ids: list[str],
    max_workers: int,
    force_rebuild: bool,
    open_file_limit: int,
    namespace: str | None,
    tag: str | None,
    env_image_tag: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "swebench.harness.prepare_images",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--max_workers",
        str(max_workers),
        "--force_rebuild",
        str(force_rebuild),
        "--open_file_limit",
        str(open_file_limit),
    ]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    if namespace is not None:
        command.extend(["--namespace", namespace])
    if tag is not None:
        command.extend(["--tag", tag])
    if env_image_tag is not None:
        command.extend(["--env_image_tag", env_image_tag])
    return command


def configure_build_overrides(
    *,
    docker_build_network: str | None,
    build_http_proxy: str | None,
    build_no_proxy: str | None,
) -> None:
    del build_no_proxy
    from swebench.harness.test_spec.test_spec import TestSpec

    git_mirror_base_url = os.environ.get("SWE_AGENT_GIT_MIRROR_BASE_URL")
    mirror_url_replacements = {}
    if git_mirror_base_url:
        git_mirror_base_url = git_mirror_base_url.rstrip("/")
        mirror_url_replacements = {
            "https://github.com/astropy/astropy": f"{git_mirror_base_url}/astropy__astropy.git",
            "https://github.com/django/django": f"{git_mirror_base_url}/django__django.git",
        }

    stale_proxy_cleanup_lines = [
        "git config --global --unset-all http.proxy || true",
        "git config --global --unset-all https.proxy || true",
    ]
    pip_constraint_lines = [
        "cat > /tmp/swebench-pip-constraints.txt <<'EOF'",
        "setuptools==44.1.1",
        "EOF",
        "export PIP_CONSTRAINT=/tmp/swebench-pip-constraints.txt",
        "export PIP_BUILD_CONSTRAINT=/tmp/swebench-pip-constraints.txt",
    ]

    original_install_repo_script = TestSpec.install_repo_script.fget

    def apply_mirror_replacements(script: str) -> str:
        for source_url, mirror_url in mirror_url_replacements.items():
            script = script.replace(source_url, mirror_url)
        return script

    def apply_legacy_astropy_install(script: str) -> str:
        if "git://127.0.0.1/astropy__astropy.git" not in script and "github.com/astropy/astropy" not in script:
            return script
        legacy_command = "SETUPTOOLS_USE_DISTUTILS=stdlib python setup.py develop --verbose"
        return script.replace("python -m pip install -e .[test] --verbose", legacy_command)

    def install_repo_script_without_stale_proxy(self):
        script = original_install_repo_script(self)
        lines = script.splitlines()
        if len(lines) >= 2:
            lines = lines[:2] + stale_proxy_cleanup_lines + pip_constraint_lines + lines[2:]
        else:
            lines.extend(stale_proxy_cleanup_lines + pip_constraint_lines)
        return apply_legacy_astropy_install(apply_mirror_replacements("\n".join(lines) + "\n"))

    TestSpec.install_repo_script = property(install_repo_script_without_stale_proxy)

    if docker_build_network:
        original_build = docker.APIClient.build

        def build_with_overrides(self, *args, **kwargs):
            kwargs.setdefault("network_mode", docker_build_network)
            return original_build(self, *args, **kwargs)

        docker.APIClient.build = build_with_overrides

    if build_http_proxy:
        def with_retry(command: str) -> str:
            retry_prefixes = (
                "conda create ",
                "conda install ",
                "python -m pip install ",
                "python3 -m pip install ",
                "pip install ",
            )
            if command.startswith(retry_prefixes):
                return f"retry {command}"
            return command

        def setup_env_script_with_proxy(self):
            proxy_lines = [
                f"export http_proxy={build_http_proxy}",
                f"export https_proxy={build_http_proxy}",
                f"export HTTP_PROXY={build_http_proxy}",
                f"export HTTPS_PROXY={build_http_proxy}",
                'retry() { local attempt=1; until "$@"; do if [ "$attempt" -ge 5 ]; then return 1; fi; sleep $((attempt * 10)); attempt=$((attempt + 1)); done; }',
                "conda config --set remote_connect_timeout_secs 60",
                "conda config --set remote_read_timeout_secs 180",
                "conda config --set remote_max_retries 5",
            ]
            return (
                "\n".join(
                    ["#!/bin/bash", "set -euxo pipefail"]
                    + proxy_lines
                    + [with_retry(command) for command in self.env_script_list]
                )
                + "\n"
            )

        def install_repo_script_with_proxy(self):
            proxy_lines = [
                f"export http_proxy={build_http_proxy}",
                f"export https_proxy={build_http_proxy}",
                f"export HTTP_PROXY={build_http_proxy}",
                f"export HTTPS_PROXY={build_http_proxy}",
                f"git config --global --replace-all http.proxy {build_http_proxy}",
                f"git config --global --replace-all https.proxy {build_http_proxy}",
                "git config --global --replace-all http.version HTTP/1.1",
                "git config --global --replace-all http.postBuffer 524288000",
                "git config --global --replace-all http.lowSpeedLimit 0",
                "git config --global --replace-all http.lowSpeedTime 999999",
            ]
            repo_script_list = [
                command.replace(" --no-use-pep517", "")
                for command in self.repo_script_list
            ]
            script = "\n".join(
                ["#!/bin/bash", "set -euxo pipefail"]
                + stale_proxy_cleanup_lines
                + pip_constraint_lines
                + proxy_lines
                + repo_script_list
            )
            return apply_legacy_astropy_install(apply_mirror_replacements(script)) + "\n"

        TestSpec.setup_env_script = property(setup_env_script_with_proxy)
        TestSpec.install_repo_script = property(install_repo_script_with_proxy)


def run_prepare_images(
    *,
    dataset_name: str,
    split: str,
    instance_ids: list[str],
    max_workers: int,
    force_rebuild: bool,
    open_file_limit: int,
    namespace: str | None,
    tag: str | None,
    env_image_tag: str | None,
    docker_build_network: str | None,
    build_http_proxy: str | None,
    build_no_proxy: str | None,
) -> int:
    configure_build_overrides(
        docker_build_network=docker_build_network,
        build_http_proxy=build_http_proxy,
        build_no_proxy=build_no_proxy,
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()
    dataset = load_swebench_dataset(dataset_name, split)
    dataset = filter_dataset_to_build(
        dataset,
        instance_ids,
        client,
        force_rebuild,
        namespace,
        tag,
        env_image_tag,
    )

    if len(dataset) == 0:
        print("All images exist. Nothing left to build.")
        return 0

    successful, failed = build_instance_images(
        client=client,
        dataset=dataset,
        force_rebuild=force_rebuild,
        max_workers=max_workers,
        namespace=namespace,
        tag=tag,
        env_image_tag=env_image_tag,
    )
    print(f"Successfully built {len(successful)} images")
    print(f"Failed to build {len(failed)} images")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build SWE-bench local sandbox images from a raw Hugging Face dataset directory."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Raw SWE-bench dataset directory downloaded from Hugging Face.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to use when loading a datasets directory.")
    parser.add_argument("--max-instances", type=int, default=None, help="Build only the first N selected instances.")
    parser.add_argument("--instance-id", action="append", default=None, help="Specific instance id to build. Can be passed multiple times.")
    parser.add_argument("--instance-ids-file", default=None, help="Text file with one instance id per line.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--open-file-limit", type=int, default=8192)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--tag", default="latest", help="Instance image tag passed to SWE-bench harness.")
    parser.add_argument("--env-image-tag", default="latest", help="Environment image tag passed to SWE-bench harness.")
    parser.add_argument("--docker-build-network", default=os.environ.get("SWE_AGENT_DOCKER_BUILD_NETWORK"), help="Optional Docker build network mode, e.g. host.")
    parser.add_argument("--build-http-proxy", default=os.environ.get("SWE_AGENT_BUILD_HTTP_PROXY"), help="Optional git HTTP proxy used inside instance image setup scripts.")
    parser.add_argument("--build-no-proxy", default=os.environ.get("SWE_AGENT_BUILD_NO_PROXY", "localhost,127.0.0.1"), help="Reserved for compatibility; currently not injected into Docker builds.")
    parser.add_argument("--swe-rex-version", default=DEFAULT_SWE_REX_VERSION, help="Version of swe-rex to preinstall into each sandbox image.")
    parser.add_argument("--skip-swe-rex-install", action="store_true", help="Do not verify or install swe-rex in the final sandbox images.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected instances and command without building.")
    args = parser.parse_args()

    requested_ids = set(args.instance_id or [])
    file_ids = load_instance_ids(args.instance_ids_file)
    if file_ids is not None:
        requested_ids.update(file_ids)
    requested_ids = requested_ids or None

    rows = load_raw_instances_from_dir(args.dataset_dir, args.split)
    instances = select_instances(rows, max_instances=args.max_instances, instance_ids=requested_ids)
    if not instances:
        raise ValueError("No instances selected from dataset.")

    instance_ids = [instance["instance_id"] for instance in instances]
    print(f"Selected {len(instance_ids)} instance(s):")
    for instance_id in instance_ids:
        print(f"  {instance_id}")
    expected_images = [
        instance_image_name(
            instance,
            namespace=args.namespace,
            tag=args.tag,
            env_image_tag=args.env_image_tag,
        )
        for instance in instances
    ]

    if missing_keys := missing_required_keys(instances):
        raise ValueError(
            "Raw SWE-bench dataset rows are missing fields required by the SWE-bench harness. "
            f"Missing keys by instance: {missing_keys}"
        )

    if args.dry_run:
        command = build_prepare_images_command(
            dataset_name="<temporary-jsonl-from-raw-hf-dataset>",
            split=args.split,
            instance_ids=instance_ids,
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            open_file_limit=args.open_file_limit,
            namespace=args.namespace,
            tag=args.tag,
            env_image_tag=args.env_image_tag,
        )
        print("Dry run command:")
        print(" ".join(command))
        if args.skip_swe_rex_install:
            print("Dry run: skip swe-rex install/check in final sandbox images.")
        else:
            print(f"Dry run: would verify/install swe-rex=={args.swe_rex_version} in final sandbox images:")
            for image in expected_images:
                print(f"  {image}")
        return 0

    existing_images = list_existing_images()
    missing_images = [image for image in expected_images if image not in existing_images]
    print(f"Existing local images for selected instances: {len(instance_ids) - len(missing_images)}/{len(instance_ids)}")
    if missing_images:
        print("Images to build:")
        for image in missing_images:
            print(f"  {image}")
    elif not args.force_rebuild:
        print("All selected images already exist. Nothing left to build.")
        if args.skip_swe_rex_install:
            return 0
        return ensure_swe_rex_in_images(
            expected_images,
            version=args.swe_rex_version,
            python_bin=DEFAULT_SANDBOX_PYTHON,
            docker_build_network=args.docker_build_network,
            build_http_proxy=args.build_http_proxy,
            build_no_proxy=args.build_no_proxy,
        )

    with tempfile.TemporaryDirectory(prefix="swebench_images_") as tmpdir:
        dataset_jsonl = Path(tmpdir) / "instances.jsonl"
        write_jsonl(instances, dataset_jsonl)
        command = build_prepare_images_command(
            dataset_name=str(dataset_jsonl),
            split=args.split,
            instance_ids=instance_ids,
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            open_file_limit=args.open_file_limit,
            namespace=args.namespace,
            tag=args.tag,
            env_image_tag=args.env_image_tag,
        )
        print("Running:")
        print(" ".join(command))
        exit_code = run_prepare_images(
            dataset_name=str(dataset_jsonl),
            split=args.split,
            instance_ids=instance_ids,
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            open_file_limit=args.open_file_limit,
            namespace=args.namespace,
            tag=args.tag,
            env_image_tag=args.env_image_tag,
            docker_build_network=args.docker_build_network,
            build_http_proxy=args.build_http_proxy,
            build_no_proxy=args.build_no_proxy,
        )
        if exit_code != 0 or args.skip_swe_rex_install:
            return exit_code
        return ensure_swe_rex_in_images(
            expected_images,
            version=args.swe_rex_version,
            python_bin=DEFAULT_SANDBOX_PYTHON,
            docker_build_network=args.docker_build_network,
            build_http_proxy=args.build_http_proxy,
            build_no_proxy=args.build_no_proxy,
        )


if __name__ == "__main__":
    raise SystemExit(main())
