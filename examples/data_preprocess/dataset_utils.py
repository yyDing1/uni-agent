"""Shared dataset loading utilities for data preprocessing scripts."""

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def get_swe_bench_sandbox_image_name(
    instance_id: str,
    *,
    namespace: str | None = None,
    tag: str = "latest",
) -> str:
    image_name = f"sweb.eval.x86_64.{instance_id}:{tag}"
    if namespace:
        image_name = f"{namespace}/{image_name}".replace("__", "_1776_")
    return image_name


def get_r2e_gym_sandbox_image_name(
    instance_id: str,
    *,
    image_template: str = "r2e-gym-subset/{instance_number}:latest",
) -> str:
    parts = instance_id.split("__")
    if len(parts) != 2:
        raise ValueError(f"Expected R2E instance id in '<repo>__<commit>' format, got: {instance_id}")
    repo_name = parts[0].lower()
    instance_number = parts[1].lower()
    return image_template.format(
        repo_name=repo_name,
        instance_number=instance_number,
        instance_id=instance_id.lower(),
    )


def load_local_dataset(data_dir: Path, split: str) -> Dataset:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {data_dir}")

    if (data_dir / "dataset_info.json").exists() or (data_dir / "dataset_dict.json").exists():
        dataset = load_from_disk(str(data_dir))
        if isinstance(dataset, DatasetDict):
            return dataset[split]
        return dataset

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if parquet_files:
        return load_dataset("parquet", data_files=[str(p) for p in parquet_files], split="train")

    jsonl_files = sorted(data_dir.rglob("*.jsonl"))
    if jsonl_files:
        return load_dataset("json", data_files=[str(p) for p in jsonl_files], split="train")

    json_files = sorted(
        p for p in data_dir.rglob("*.json") if p.name not in {"dataset_info.json", "dataset_dict.json"}
    )
    if json_files:
        return load_dataset("json", data_files=[str(p) for p in json_files], split="train")

    arrow_files = sorted(data_dir.rglob("*.arrow"))
    if arrow_files:
        return Dataset.from_file(str(arrow_files[0]))

    raise FileNotFoundError(
        f"No supported dataset files found in {data_dir}. "
        "Expected a datasets.save_to_disk directory, parquet, jsonl, json, or arrow files."
    )


def load_local_dataset_as_dicts(path: str, split: str) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    if data_path.is_dir():
        return [dict(row) for row in load_local_dataset(data_path, split)]

    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    elif suffix in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=str(data_path), split="train")
    elif suffix == ".arrow":
        dataset = Dataset.from_file(str(data_path))
    else:
        raise ValueError(f"Unsupported dataset file extension: {data_path}")
    return [dict(row) for row in dataset]


def load_instance_ids(path: str | None) -> set[str] | None:
    if path is None:
        return None
    ids_path = Path(path).expanduser()
    ids = set()
    for line in ids_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids
