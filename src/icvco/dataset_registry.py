# Copyright 2026 OPPO. All rights reserved.
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

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from datasets import Dataset, DatasetDict, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_REGISTRY_PATH = REPO_ROOT / "configs" / "datasets" / "registry.json"


def _expand_path_string(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _infer_hf_dataset_builder_name(sample_path: str) -> str:
    suffix = Path(sample_path).suffix.lower()
    if not suffix:
        raise ValueError(f"Could not infer file format from {sample_path!r}.")

    # Hugging Face datasets uses the "json" builder for both .json and .jsonl files.
    if suffix in {".json", ".jsonl"}:
        return "json"

    return suffix.lstrip(".")


def _registered_dataset_column_names(dataset_config: Mapping[str, Any]) -> set[str]:
    column_names: set[str] = set()
    for key, value in dataset_config.items():
        if key.endswith("_col") and isinstance(value, str):
            column_names.add(value)
    return column_names


def _registered_image_column_names(dataset_config: Mapping[str, Any]) -> set[str]:
    image_columns: set[str] = set()
    for key, value in dataset_config.items():
        if key.endswith("_col") and "image" in key and isinstance(value, str):
            image_columns.add(value)
    return image_columns


def _looks_like_uri(path: str) -> bool:
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)


def _resolve_local_jsonl_image_path(value: Any, base_dir: Path) -> Any:
    if isinstance(value, str):
        expanded = _expand_path_string(value)
        if os.path.isabs(expanded) or _looks_like_uri(expanded):
            return expanded
        return str((base_dir / expanded).resolve())

    if isinstance(value, Mapping) and isinstance(value.get("path"), str):
        normalized = dict(value)
        normalized["path"] = _resolve_local_jsonl_image_path(value["path"], base_dir)
        return normalized

    return value


def _load_registered_jsonl_split(
    split_path: str,
    required_fields: set[str],
    optional_fields: set[str],
    image_fields: set[str],
) -> Dataset:
    records: list[dict[str, Any]] = []
    split_base_dir = Path(split_path).resolve().parent

    with Path(split_path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record in {split_path}:{line_number}") from exc

            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL record in {split_path}:{line_number} must be an object.")

            record = {field_name: row.get(field_name) for field_name in required_fields}
            for field_name in optional_fields:
                if field_name in row:
                    record[field_name] = row[field_name]
            for field_name in image_fields:
                if field_name in record:
                    record[field_name] = _resolve_local_jsonl_image_path(record[field_name], split_base_dir)
            records.append(record)

    return Dataset.from_list(records)


def _load_registered_jsonl_dataset(
    data_files: Mapping[str, str],
    dataset_config: Mapping[str, Any],
) -> DatasetDict:
    required_fields = _registered_dataset_column_names(dataset_config)
    image_fields = _registered_image_column_names(dataset_config)
    optional_fields = {"synthetic_image", "synthetic_response"}

    split_datasets = {
        split_name: _load_registered_jsonl_split(split_path, required_fields, optional_fields, image_fields)
        for split_name, split_path in data_files.items()
    }
    return DatasetDict(split_datasets)


def _normalize_dataset_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    dataset_config = dict(raw_config)

    if "path" in dataset_config and isinstance(dataset_config["path"], str):
        dataset_config["path"] = _expand_path_string(dataset_config["path"])

    data_files = dataset_config.get("data_files")
    if isinstance(data_files, Mapping):
        dataset_config["data_files"] = {
            split_name: str((REPO_ROOT / _expand_path_string(path)).resolve())
            if isinstance(path, str) and not os.path.isabs(_expand_path_string(path)) and not _looks_like_uri(path)
            else _expand_path_string(path) if isinstance(path, str) else path
            for split_name, path in data_files.items()
        }

    return dataset_config


@lru_cache(maxsize=1)
def load_dataset_registry() -> dict[str, dict[str, Any]]:
    if not DATASET_REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Dataset registry not found: {DATASET_REGISTRY_PATH}")

    payload = json.loads(DATASET_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset registry must be a JSON object: {DATASET_REGISTRY_PATH}")

    registry: dict[str, dict[str, Any]] = {}
    for dataset_name, raw_config in payload.items():
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"Dataset config for '{dataset_name}' must be an object.")
        registry[dataset_name] = _normalize_dataset_config(raw_config)
    return registry


def is_registered_dataset(dataset_name: str) -> bool:
    return dataset_name in load_dataset_registry()


def get_registered_dataset_config(dataset_name: str) -> dict[str, Any] | None:
    dataset_config = load_dataset_registry().get(dataset_name)
    if dataset_config is None:
        return None
    return dict(dataset_config)


def require_registered_dataset_config(dataset_name: str) -> dict[str, Any]:
    dataset_config = get_registered_dataset_config(dataset_name)
    if dataset_config is None:
        raise ValueError(
            f"Unknown dataset alias: {dataset_name}. "
            f"Add it to {DATASET_REGISTRY_PATH.relative_to(REPO_ROOT)} or pass a direct datasets identifier."
        )
    return dataset_config


def ensure_dataset_config_fields(
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    required_fields: list[str],
) -> None:
    missing_fields = [field_name for field_name in required_fields if field_name not in dataset_config]
    if missing_fields:
        missing_str = ", ".join(missing_fields)
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required config fields: {missing_str}. "
            f"Update {DATASET_REGISTRY_PATH.relative_to(REPO_ROOT)}."
        )


def _is_valid_text_value(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_valid_image_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        image_path = value.get("path")
        image_bytes = value.get("bytes")
        return (isinstance(image_path, str) and bool(image_path.strip())) or image_bytes is not None
    return True


def _valid_preference_batch(
    examples: Mapping[str, list[Any]],
    *,
    text_columns: list[str],
    image_columns: list[str],
) -> list[bool]:
    reference_column = text_columns[0] if text_columns else image_columns[0]
    batch_size = len(examples[reference_column])
    keep: list[bool] = []

    for index in range(batch_size):
        valid_text = all(_is_valid_text_value(examples[column][index]) for column in text_columns)
        valid_images = all(_is_valid_image_value(examples[column][index]) for column in image_columns)
        keep.append(valid_text and valid_images)

    return keep


def filter_invalid_preference_rows(
    dataset: DatasetDict,
    *,
    text_columns: list[str],
    image_columns: list[str],
    num_proc: int | None = None,
    batch_size: int = 1000,
) -> tuple[DatasetDict, dict[str, dict[str, int]]]:
    """Drop rows that cannot be converted into preference/SFT chat examples."""
    filtered = DatasetDict()
    stats: dict[str, dict[str, int]] = {}

    for split_name, split_dataset in dataset.items():
        before_count = len(split_dataset)
        filtered_split = split_dataset.filter(
            _valid_preference_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            fn_kwargs={
                "text_columns": text_columns,
                "image_columns": image_columns,
            },
        )
        after_count = len(filtered_split)
        if after_count == 0 and before_count > 0:
            raise ValueError(
                f"All rows in split '{split_name}' were filtered as invalid. "
                f"Required text columns: {text_columns}; image columns: {image_columns}."
            )

        filtered[split_name] = filtered_split
        stats[split_name] = {
            "before": before_count,
            "after": after_count,
            "dropped": before_count - after_count,
        }

    return filtered, stats


def load_registered_dataset(dataset_name: str):
    dataset_config = require_registered_dataset_config(dataset_name)

    if "data_files" in dataset_config:
        data_files = dataset_config["data_files"]
        if not isinstance(data_files, Mapping) or not data_files:
            raise ValueError(f"Dataset '{dataset_name}' has invalid data_files config.")

        sample_path = next(iter(data_files.values()))
        file_format = _infer_hf_dataset_builder_name(sample_path)

        if Path(sample_path).suffix.lower() == ".jsonl":
            dataset = _load_registered_jsonl_dataset(data_files, dataset_config)
        else:
            dataset = load_dataset(file_format, data_files=data_files)
    elif "path" in dataset_config:
        dataset = load_dataset(dataset_config["path"])
    else:
        raise ValueError(f"Dataset '{dataset_name}' must define either 'data_files' or 'path'.")

    return dataset, dataset_config


def cast_image_columns_decode_false(dataset, column_names: list[str]):
    from datasets import Image as DatasetImage
    from datasets.features.features import List as DatasetList

    for split_name in dataset.keys():
        split_dataset = dataset[split_name]
        for column_name in column_names:
            if column_name not in split_dataset.column_names:
                continue
            feature = split_dataset.features.get(column_name)
            if isinstance(feature, DatasetImage):
                dataset[split_name] = dataset[split_name].cast_column(column_name, DatasetImage(decode=False))
            elif isinstance(feature, DatasetList) and isinstance(feature.feature, DatasetImage):
                dataset[split_name] = dataset[split_name].cast_column(
                    column_name,
                    DatasetList(DatasetImage(decode=False)),
                )
    return dataset
