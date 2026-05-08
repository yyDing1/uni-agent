#!/usr/bin/env python3
# ruff: noqa: E501
"""Build local R2E-Gym sandbox images from a raw Hugging Face dataset directory."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import docker
import r2egym
from r2egym.commit_models.diff_classes import ParsedCommit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_preprocess"))
from dataset_utils import get_r2e_gym_sandbox_image_name, load_instance_ids, load_local_dataset_as_dicts


DEFAULT_DATASET_DIR = "~/dataset/r2e-gym-subset-filtered"
DEFAULT_IMAGE_TEMPLATE = os.getenv("LOCAL_R2E_IMAGE_TEMPLATE", "r2e-gym-subset/{instance_number}:latest")
DEFAULT_SWE_REX_VERSION = "1.4.0"
DEFAULT_SANDBOX_PYTHON = "/testbed/.venv/bin/python"

REQUIRED_R2E_KEYS = {
    "repo_name",
    "commit_hash",
    "parsed_commit_content",
    "problem_statement",
    "expected_output_json",
}

R2EGYM_PACKAGE_DIR = Path(r2egym.__file__).resolve().parent
R2EGYM_BASE_DOCKERFILES_DIR = R2EGYM_PACKAGE_DIR / "repo_analysis" / "base_dockerfiles"
R2EGYM_INSTALL_UTILS_DIR = R2EGYM_PACKAGE_DIR / "install_utils"

R2E_TEST_COMMANDS = {
    "aiohttp": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "coveragepy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "datalad": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "numpy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "orange3": "QT_QPA_PLATFORM=minimal PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' xvfb-run --auto-servernum .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "pandas": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "pillow": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "pyramid": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "scrapy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "sympy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "tornado": ".venv/bin/python -W ignore r2e_tests/tornado_unittest_runner.py",
}


def load_raw_rows_from_dir(dataset_dir: str, split: str) -> list[dict[str, Any]]:
    path = Path(dataset_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(
            "Only raw Hugging Face dataset directories are supported. "
            "Pass the directory downloaded from Hugging Face, not a converted parquet file."
        )
    return load_local_dataset_as_dicts(str(path), split)


def missing_required_keys(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    missing = {}
    for row in rows:
        missing_keys = sorted(key for key in REQUIRED_R2E_KEYS if key not in row)
        if missing_keys:
            instance_id = make_instance_id(row) if "repo_name" in row and "commit_hash" in row else "<unknown>"
            missing[instance_id] = missing_keys
    return missing


def make_instance_id(row: dict[str, Any]) -> str:
    return f"{row['repo_name']}__{row['commit_hash'][:10]}"


def parse_commit(row: dict[str, Any]) -> ParsedCommit:
    parsed_commit = ParsedCommit(**json.loads(row["parsed_commit_content"]))
    if row["commit_hash"] != parsed_commit.new_commit_hash:
        raise ValueError(
            f"Row commit_hash does not match parsed_commit_content.new_commit_hash for {make_instance_id(row)}: "
            f"{row['commit_hash']} != {parsed_commit.new_commit_hash}"
        )
    return parsed_commit


def select_rows(
    rows: list[dict[str, Any]],
    *,
    max_instances: int | None,
    instance_ids: set[str] | None,
) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for row in rows:
        if "extra_info" in row and "parsed_commit_content" not in row:
            raise ValueError(
                "Converted Uni-Agent datasets are not supported for building R2E-Gym sandbox images. "
                "Pass the raw Hugging Face R2E-Gym dataset directory instead."
            )

        instance_id = make_instance_id(row)
        if instance_ids is not None and instance_id not in instance_ids:
            continue
        if instance_id in seen:
            continue
        selected.append(row)
        seen.add(instance_id)
        if max_instances is not None and len(selected) >= max_instances:
            break
    return selected


def r2e_image_name(row: dict[str, Any], *, image_template: str) -> str:
    return get_r2e_gym_sandbox_image_name(make_instance_id(row), image_template=image_template)


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
    with tempfile.TemporaryDirectory(prefix="r2e_swe_rex_layer_") as tmpdir:
        dockerfile = Path(tmpdir) / "Dockerfile"
        dockerfile.write_text(
            "\n".join(
                [
                    f"FROM {image}",
                    "ENV PIP_PROGRESS_BAR=off",
                    "RUN set -eux; \\",
                    f"    install_cmd='{python_bin} -m pip install --no-cache-dir -q swe-rex=={version}'; \\",
                    f"    if command -v uv >/dev/null 2>&1; then install_cmd='uv pip install --python {python_bin} swe-rex=={version}'; fi; \\",
                    '    attempt=1; until sh -c "$install_cmd"; do \\',
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=4, ensure_ascii=True), encoding="utf-8")


def write_r2e_tests(build_dir: Path, repo_name: str, parsed_commit: ParsedCommit) -> None:
    r2e_tests_dir = build_dir / "r2e_tests"
    r2e_tests_dir.mkdir(parents=True, exist_ok=True)
    (r2e_tests_dir / "__init__.py").write_text("", encoding="utf-8")

    test_file_diffs = [file_diff for file_diff in parsed_commit.file_diffs if file_diff.is_test_file]
    if not test_file_diffs:
        raise ValueError(f"No test file diffs found for {repo_name}__{parsed_commit.new_commit_hash[:10]}")

    old_file_names = []
    for index, file_diff in enumerate(test_file_diffs, start=1):
        test_file_name = f"test_{index}.py"
        test_file_content = file_diff.new_file_content
        old_file_names.append(file_diff.path)
        (r2e_tests_dir / test_file_name).write_text(test_file_content, encoding="utf-8")

    if repo_name == "pillow" and any("unittest" in file_diff.new_file_content for file_diff in test_file_diffs):
        shutil.copy(R2EGYM_INSTALL_UTILS_DIR / "unittest_custom_runner.py", r2e_tests_dir / "unittest_custom_runner.py")
    elif repo_name == "tornado":
        shutil.copy(R2EGYM_INSTALL_UTILS_DIR / "tornado_unittest_runner.py", r2e_tests_dir / "tornado_unittest_runner.py")
    elif repo_name == "datalad":
        conftest = (R2EGYM_INSTALL_UTILS_DIR / "datalads_conftest.py").read_text(encoding="utf-8")
        (r2e_tests_dir / "conftest.py").write_text(conftest, encoding="utf-8")
        old_file_names.append("a/b/c/conftest.py")

    if repo_name in {"datalad", "numpy"}:
        for path in r2e_tests_dir.glob("test_*.py"):
            index = int(path.stem.split("_", 1)[1]) - 1
            old_parts = old_file_names[index].split("/")
            old_file_name_dot = ".".join(old_parts[:-1]) + "."
            old_file_name_dot_dot = ".".join(old_parts[:-2]) + "."
            old_file_name_dot_dot_dot = ".".join(old_parts[:-3]) + "."
            content = path.read_text(encoding="utf-8")
            content = content.replace("def setup(self)", "def setup_method(self)")
            content = content.replace("def teardown(self)", "def teardown_method(self)")
            content = content.replace("from ...", f"from {old_file_name_dot_dot_dot}")
            content = content.replace("from ..", f"from {old_file_name_dot_dot}")
            content = content.replace("from .", f"from {old_file_name_dot}")
            path.write_text(content, encoding="utf-8")


def get_test_command(repo_name: str, r2e_tests_dir: Path) -> str:
    if repo_name == "pillow" and (r2e_tests_dir / "unittest_custom_runner.py").exists():
        return ".venv/bin/python -W ignore r2e_tests/unittest_custom_runner.py"
    if repo_name not in R2E_TEST_COMMANDS:
        raise ValueError(f"Unsupported R2E repo: {repo_name}")
    return R2E_TEST_COMMANDS[repo_name]


def create_placeholder_tool_scripts(build_dir: Path) -> None:
    (build_dir / "list_files.py").write_text(
        "from pathlib import Path\nfor path in sorted(Path('/testbed').rglob('*')):\n    if path.is_file():\n        print(path)\n",
        encoding="utf-8",
    )
    (build_dir / "read_file.py").write_text(
        "import sys\nfrom pathlib import Path\nprint(Path(sys.argv[1]).read_text())\n",
        encoding="utf-8",
    )


def write_post_copy_fixes(build_dir: Path) -> None:
    (build_dir / "post_copy_fixes.sh").write_text(
        """#!/usr/bin/env bash
set -e

R2E_TESTS=""
for candidate in /r2e_tests /testbed/r2e_tests /sympy/r2e_tests; do
    if [ -d "$candidate" ]; then
        R2E_TESTS="$candidate"
        break
    fi
done

if [ -z "$R2E_TESTS" ]; then
    exit 0
fi

for helper in /testbed/Tests/helper.py /sympy/Tests/helper.py; do
    if [ -f "$helper" ]; then
        cp "$helper" "$R2E_TESTS/helper.py"
    fi
done

for base in /testbed/tests /testbed/pyramid/tests /sympy/tests /sympy/pyramid/tests; do
    if [ -d "$base" ]; then
        for folder in fixtures test_config test_scripts; do
            if [ -d "$base/$folder" ] && [ ! -e "$R2E_TESTS/$folder" ]; then
                cp -r "$base/$folder" "$R2E_TESTS/$folder"
            fi
        done
    fi
done

python3 - "$R2E_TESTS" <<'PY'
import sys
from pathlib import Path

r2e_tests = Path(sys.argv[1])
for path in r2e_tests.glob("*.py"):
    content = path.read_text()
    if "sys.modules" in content and not content.startswith("import pyramid.tests\\n"):
        path.write_text("import pyramid.tests\\n" + content)
PY
""",
        encoding="utf-8",
    )


def write_orange3_install_script(build_dir: Path) -> None:
    (build_dir / "install.sh").write_text(
        """#!/usr/bin/env bash
set -e

pip_install() {
    .venv/bin/python -m pip install --retries 8 --timeout 60 "$@"
}

check_orange() {
    echo "Verifying Orange installation..."
    if .venv/bin/python -c "import Orange; print(Orange.__file__)" > /dev/null; then
        echo "Orange installation successful"
        ln -sfn Orange/tests/datasets datasets
        return 0
    fi
    echo "Orange verification failed"
    return 1
}

run_setup_py() {
    .venv/bin/python - "$@" <<'PY'
import runpy
import sys

try:
    build_classes = []
    from distutils.command.build import build as distutils_build

    build_classes.append(distutils_build)
    try:
        from numpy.distutils.command.build import build as numpy_build

        build_classes.append(numpy_build)
    except Exception:
        pass

    defaults = {
        "warn_error": False,
        "cpu_baseline": None,
        "cpu_dispatch": None,
        "disable_optimization": False,
        "fcompiler": None,
        "parallel": None,
    }
    for build_class in build_classes:
        for name, value in defaults.items():
            if not hasattr(build_class, name):
                setattr(build_class, name, value)
except Exception:
    pass

sys.argv = ["setup.py", *sys.argv[1:]]
runpy.run_path("setup.py", run_name="__main__")
PY
}

install_with_python() {
    local python_version="$1"
    echo "Attempting Orange installation with Python ${python_version}..."
    rm -rf build Orange3.egg-info
    if [ -x .venv/bin/python ]; then
        current_version=$(.venv/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
        if [ "${current_version}" != "${python_version}" ]; then
            rm -rf .venv
        fi
    fi

    if [ ! -x .venv/bin/python ]; then
        uv venv --python "${python_version}" --python-preference only-managed
    fi

    if [ ! -x .venv/bin/python ]; then
        echo "Failed to create Orange virtual environment with Python ${python_version}"
        return 1
    fi

    uv pip install --python .venv/bin/python pip
    source .venv/bin/activate
    local constraints=/tmp/orange3_constraints.txt
    cat > "${constraints}" <<'EOF'
setuptools<60
numpy<1.24
cython<0.30
EOF

    if [ -f .orange3_base_deps_ready ]; then
        echo "Using cached Orange base dependency layer"
    else
        pip_install -c "${constraints}" --upgrade "setuptools<60" wheel
        pip_install -c "${constraints}" "numpy<1.24" "cython<0.30" pytest
        pip_install -c "${constraints}" "PyQt5>=5.12,!=5.15.1"
        pip_install -c "${constraints}" "PyQtWebEngine>=5.12"
        pip_install -c "${constraints}" -r requirements-core.txt
        pip_install -c "${constraints}" -r requirements-gui.txt
        if [ -f requirements-opt.txt ]; then
            echo "Skipping optional Orange requirements from requirements-opt.txt"
        fi
        if [ -f requirements-sql.txt ]; then
            pip_install -c "${constraints}" -r requirements-sql.txt || echo "Optional Orange SQL requirements failed; continuing"
        fi
    fi

    .venv/bin/python - <<'PY'
import numpy
from numpy.distutils.core import setup
print(f"Using numpy {numpy.__version__} with numpy.distutils")
PY
    run_setup_py build_ext --inplace
    run_setup_py develop
    check_orange
}

main() {
    echo "Starting Orange installation attempts..."
    if install_with_python 3.8; then
        echo "Successfully installed Orange using Python 3.8"
        return 0
    fi

    echo "Python 3.8 installation failed, trying Python 3.10..."
    if install_with_python 3.10; then
        echo "Successfully installed Orange using Python 3.10"
        return 0
    fi

    echo "All Orange installation attempts failed"
    return 1
}

main
""",
        encoding="utf-8",
    )


def write_orange3_prepare_env_script(build_dir: Path) -> None:
    (build_dir / "prepare_orange3_env.sh").write_text(
        """#!/usr/bin/env bash
set -e

pip_install() {
    .venv/bin/python -m pip install --retries 8 --timeout 60 "$@"
}

rm -rf .venv
uv venv --python 3.8 --python-preference only-managed

if [ ! -x .venv/bin/python ]; then
    echo "Failed to create Orange base virtual environment"
    exit 1
fi

uv pip install --python .venv/bin/python pip
source .venv/bin/activate
constraints=/tmp/orange3_constraints.txt
cat > "${constraints}" <<'EOF'
setuptools<60
numpy<1.24
cython<0.30
EOF

pip_install -c "${constraints}" --upgrade "setuptools<60" wheel
pip_install -c "${constraints}" "numpy<1.24" "cython<0.30" pytest
pip_install -c "${constraints}" "PyQt5>=5.12,!=5.15.1"
pip_install -c "${constraints}" "PyQtWebEngine>=5.12"
pip_install -c "${constraints}" \\
    anyqt baycomp bottleneck chardet commonmark contourpy cryptography cycler \\
    dictdiffer docutils et-xmlfile fonttools importlib-metadata importlib-resources \\
    joblib keyring keyrings.alt kiwisolver matplotlib networkx openpyxl openTSNE \\
    orange-canvas-core orange-widget-base packaging pandas pillow platformdirs \\
    pyparsing pyqtgraph python-dateutil python-louvain pytz PyYAML qasync requests \\
    requests-cache scikit-learn scipy serverfiles six threadpoolctl tzdata \\
    url-normalize urllib3 xlrd xlsxwriter zipp
pip_install tree_sitter_languages

touch .orange3_base_deps_ready
""",
        encoding="utf-8",
    )


def dockerfile_content_for_repo(repo_name: str, dockerfile_path: Path) -> str:
    content = dockerfile_path.read_text(encoding="utf-8")
    content = content.replace("\nARG OLD_COMMIT\n\n", "\n\n")
    content = content.replace("\nRUN git checkout $OLD_COMMIT", "\nARG OLD_COMMIT\nRUN git checkout $OLD_COMMIT")
    if repo_name == "orange3":
        content = content.replace(
            "\nCOPY install.sh /testbed/install.sh\n\nWORKDIR /testbed\n",
            "\nCOPY install.sh /testbed/install.sh\nCOPY prepare_orange3_env.sh /testbed/prepare_orange3_env.sh\n\nWORKDIR /testbed\n\nRUN bash prepare_orange3_env.sh\n",
        )
        content = content.replace(
            "\nRUN uv pip install tree_sitter_languages\n",
            "\nRUN .venv/bin/python -c \"import tree_sitter_languages\"\n",
        )
    if repo_name == "aiohttp":
        content = content.replace(
            "COPY Makefile /testbed/Makefile",
            "RUN sed -i 's/python -m pip install/pip install/g; s/pip install/uv pip install/g' Makefile",
        )
    return content.rstrip() + "\n\nCOPY post_copy_fixes.sh /post_copy_fixes.sh\nRUN bash /post_copy_fixes.sh\n"


def prepare_build_context(row: dict[str, Any], build_dir: Path) -> ParsedCommit:
    repo_name = row["repo_name"]
    parsed_commit = parse_commit(row)
    dockerfile_path = R2EGYM_BASE_DOCKERFILES_DIR / f"Dockerfile.{repo_name}"
    install_path = R2EGYM_INSTALL_UTILS_DIR / f"{repo_name}_install.sh"
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"R2E-Gym Dockerfile not found for repo {repo_name}: {dockerfile_path}")
    if not install_path.exists() and repo_name != "sympy":
        raise FileNotFoundError(f"R2E-Gym install script not found for repo {repo_name}: {install_path}")

    (build_dir / "Dockerfile").write_text(dockerfile_content_for_repo(repo_name, dockerfile_path), encoding="utf-8")
    write_post_copy_fixes(build_dir)
    if repo_name == "orange3":
        write_orange3_install_script(build_dir)
        write_orange3_prepare_env_script(build_dir)
    elif install_path.exists():
        shutil.copy(install_path, build_dir / "install.sh")

    write_r2e_tests(build_dir, repo_name, parsed_commit)
    r2e_tests_dir = build_dir / "r2e_tests"
    (build_dir / "run_tests.sh").write_text(get_test_command(repo_name, r2e_tests_dir), encoding="utf-8")
    write_json(build_dir / "expected_test_output.json", json.loads(row["expected_output_json"]))
    write_json(build_dir / "parsed_commit.json", json.loads(row["parsed_commit_content"]))
    write_json(build_dir / "modified_files.json", parsed_commit.file_name_list)
    write_json(
        build_dir / "modified_entities.json",
        [entity.json_summary_dict() for entity in parsed_commit.edited_entities()],
    )
    write_json(
        build_dir / "syn_issue.json",
        {
            "model_output": "",
            "syn_issue": row["problem_statement"],
            "prompt": row["problem_statement"],
        },
    )
    write_json(
        build_dir / "execution_result.json",
        {
            "repo_name": repo_name,
            "new_commit_hash": parsed_commit.new_commit_hash,
        },
    )

    if repo_name == "aiohttp":
        shutil.copy(R2EGYM_INSTALL_UTILS_DIR / "process_aiohttp_updateasyncio.py", build_dir / "process_aiohttp_updateasyncio.py")
    elif repo_name == "sympy":
        create_placeholder_tool_scripts(build_dir)

    return parsed_commit


def docker_build(
    *,
    image: str,
    build_dir: Path,
    old_commit: str,
    docker_build_network: str | None,
    build_http_proxy: str | None,
    build_no_proxy: str | None,
) -> None:
    command = [
        "docker",
        "build",
        "--memory",
        "1000000000",
        "-t",
        image,
        ".",
        "--build-arg",
        f"OLD_COMMIT={old_commit}",
    ]
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

    print("Running:")
    print(" ".join(command))
    subprocess.run(command, cwd=build_dir, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build R2E-Gym local sandbox images from a raw Hugging Face dataset directory.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Raw R2E-Gym dataset directory downloaded from Hugging Face.")
    parser.add_argument("--split", default="train", help="Dataset split to use when loading a datasets directory.")
    parser.add_argument("--max-instances", type=int, default=None, help="Build only the first N selected instances.")
    parser.add_argument("--instance-id", action="append", default=None, help="Specific instance id to build, e.g. orange3__2d9617bd0c. Can be passed multiple times.")
    parser.add_argument("--instance-ids-file", default=None, help="Text file with one instance id per line.")
    parser.add_argument("--image-template", default=DEFAULT_IMAGE_TEMPLATE, help="Local image template. Supports {repo_name}, {instance_number}, and {instance_id}.")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--docker-build-network", default=os.environ.get("R2E_GYM_DOCKER_BUILD_NETWORK"), help="Optional Docker build network mode, e.g. host.")
    parser.add_argument("--build-http-proxy", default=os.environ.get("R2E_GYM_BUILD_HTTP_PROXY"), help="Optional HTTP proxy passed as Docker build args.")
    parser.add_argument("--build-no-proxy", default=os.environ.get("R2E_GYM_BUILD_NO_PROXY", "localhost,127.0.0.1"), help="Optional no_proxy passed as Docker build args.")
    parser.add_argument("--swe-rex-version", default=DEFAULT_SWE_REX_VERSION, help="Version of swe-rex to preinstall into each sandbox image.")
    parser.add_argument("--skip-swe-rex-install", action="store_true", help="Do not verify or install swe-rex in the final sandbox images.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected instances without building.")
    args = parser.parse_args()

    requested_ids = set(args.instance_id or [])
    file_ids = load_instance_ids(args.instance_ids_file)
    if file_ids is not None:
        requested_ids.update(file_ids)
    requested_ids = requested_ids or None

    rows = load_raw_rows_from_dir(args.dataset_dir, args.split)
    if missing_keys := missing_required_keys(rows):
        raise ValueError(f"Raw R2E-Gym dataset rows are missing required fields: {missing_keys}")

    selected_rows = select_rows(rows, max_instances=args.max_instances, instance_ids=requested_ids)
    if not selected_rows:
        raise ValueError("No instances selected from dataset.")

    expected_images = [r2e_image_name(row, image_template=args.image_template) for row in selected_rows]
    print(f"Selected {len(selected_rows)} instance(s):")
    for row, image in zip(selected_rows, expected_images, strict=True):
        parsed_commit = parse_commit(row)
        print(f"  {make_instance_id(row)} -> {image} (OLD_COMMIT={parsed_commit.old_commit_hash})")

    if args.dry_run:
        if args.skip_swe_rex_install:
            print("Dry run: skip swe-rex install/check in final sandbox images.")
        else:
            print(f"Dry run: would verify/install swe-rex=={args.swe_rex_version} in final sandbox images.")
        return 0

    existing_images = list_existing_images()
    rows_to_build = []
    images_to_build = []
    for row, image in zip(selected_rows, expected_images, strict=True):
        if image in existing_images and not args.force_rebuild:
            print(f"Image already exists, skipping build: {image}")
            continue
        rows_to_build.append(row)
        images_to_build.append(image)

    failed = []
    for row, image in zip(rows_to_build, images_to_build, strict=True):
        instance_id = make_instance_id(row)
        try:
            with tempfile.TemporaryDirectory(prefix=f"r2e_gym_{instance_id}_") as tmpdir:
                build_dir = Path(tmpdir)
                parsed_commit = prepare_build_context(row, build_dir)
                docker_build(
                    image=image,
                    build_dir=build_dir,
                    old_commit=parsed_commit.old_commit_hash,
                    docker_build_network=args.docker_build_network,
                    build_http_proxy=args.build_http_proxy,
                    build_no_proxy=args.build_no_proxy,
                )
        except Exception as exc:
            failed.append(image)
            print(f"Failed to build {image}: {exc}", file=sys.stderr)

    if failed:
        print(f"Failed to build {len(failed)} image(s):", file=sys.stderr)
        for image in failed:
            print(f"  {image}", file=sys.stderr)
        return 1

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


if __name__ == "__main__":
    raise SystemExit(main())
