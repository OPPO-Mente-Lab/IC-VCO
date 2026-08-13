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

from argparse import ArgumentParser
import json
import os
import shutil

from peft import PeftModel
import torch
from transformers import AutoModelForImageTextToText

ADAPTER_DIRNAME = "adapter_output"
ADAPTER_MARKER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def resolve_adapter_dir(path: str) -> str:
    root_adapter = os.path.join(path, "adapter_config.json")
    nested_adapter = os.path.join(path, ADAPTER_DIRNAME, "adapter_config.json")

    if os.path.isfile(root_adapter):
        return path
    if os.path.isfile(nested_adapter):
        return os.path.join(path, ADAPTER_DIRNAME)
    raise FileNotFoundError(
        f"No adapter checkpoint found under {path}. Expected adapter_config.json "
        f"either at the root or under adapter_output/."
    )


def resolve_save_path(path: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    if base.startswith("checkpoint-") or base == "adapter_output":
        return os.path.dirname(os.path.normpath(path))
    return path


def load_base_model_path(adapter_dir: str) -> str:
    with open(os.path.join(adapter_dir, "adapter_config.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)["base_model_name_or_path"]


def preserve_root_adapter(adapter_dir: str, save_path: str) -> None:
    if os.path.abspath(adapter_dir) != os.path.abspath(save_path):
        return

    nested_adapter_dir = os.path.join(save_path, ADAPTER_DIRNAME)
    os.makedirs(nested_adapter_dir, exist_ok=True)
    for filename in ADAPTER_MARKER_FILES + ("README.md",):
        source = os.path.join(adapter_dir, filename)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(nested_adapter_dir, filename))


def remove_root_adapter_markers(save_path: str) -> None:
    for filename in ADAPTER_MARKER_FILES:
        marker_path = os.path.join(save_path, filename)
        if os.path.isfile(marker_path):
            os.remove(marker_path)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    adapter_dir = resolve_adapter_dir(args.path)
    save_path = resolve_save_path(args.path)
    base_model_path = load_base_model_path(adapter_dir)
    preserve_root_adapter(adapter_dir, save_path)

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()

    model.config.torch_dtype = torch.bfloat16
    model = model.to(torch.bfloat16)

    print("Verifying image_newline existence...")
    state_dict = model.state_dict()
    found = False
    for key, value in state_dict.items():
        if "image_newline" in key:
            print(f"Found weight: {key}, Shape: {value.shape}")
            found = True

    if not found:
        print("WARNING: image_newline NOT found in current state_dict!")
    else:
        print("image_newline is present in memory.")

    model.save_pretrained(save_path)
    remove_root_adapter_markers(save_path)
    print(f"Saved merged model to {save_path}")


if __name__ == "__main__":
    main()
