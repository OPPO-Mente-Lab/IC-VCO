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

import json
import os
import runpy
import socket
import sys


def _prefer_current_vlmevalkit_root() -> None:
    root = os.environ.get("VLMEVALKIT_ROOT") or os.getcwd()
    if not root:
        return
    root = os.path.abspath(root)
    if not os.path.isdir(os.path.join(root, "vlmeval")):
        return

    normalized = []
    for path in sys.path:
        if path == "":
            path_abs = os.path.abspath(os.getcwd())
        else:
            path_abs = os.path.abspath(path)
        if path_abs != root:
            normalized.append(path)
    sys.path[:] = [root, *normalized]


_prefer_current_vlmevalkit_root()


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no"}


def _arg_value(flag: str) -> str | None:
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _load_config_datasets() -> list[str]:
    config_path = _arg_value("--config")
    if not config_path or not os.path.isfile(config_path):
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return []
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        return []
    return [str(name) for name in data_cfg.keys()]


def _set_torchrun_cuda_device() -> None:
    local_rank_raw = os.environ.get("LOCAL_RANK")
    if local_rank_raw is None:
        return

    try:
        local_rank = int(local_rank_raw)
    except ValueError:
        return

    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    torch.cuda.set_device(local_rank)
    if _truthy_env("ICVCO_EVAL_BOOTSTRAP_DIAG"):
        print(
            "[icvco-bootstrap] "
            f"set_cuda_device={local_rank} "
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '-')}",
            file=sys.stderr,
            flush=True,
        )


def _print_bootstrap_diag() -> None:
    if not _truthy_env("ICVCO_EVAL_BOOTSTRAP_DIAG"):
        return

    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    world_size = os.environ.get("WORLD_SIZE", "?")
    hostname = socket.gethostname()
    config_path = _arg_value("--config") or "-"
    datasets = _load_config_datasets()

    try:
        from vlmeval.vlm.base import BaseModel

        preproc_file = BaseModel.preproc_content.__code__.co_filename
        preproc_line = BaseModel.preproc_content.__code__.co_firstlineno
    except Exception as err:
        preproc_file = f"<import failed: {err}>"
        preproc_line = -1

    print(
        "[icvco-bootstrap] "
        f"host={hostname} rank={rank}/{world_size} local_rank={local_rank} "
        f"config={config_path} datasets={','.join(datasets) or '-'} "
        f"lmu_data={os.environ.get('LMUData', '-')}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[icvco-bootstrap] "
        f"host={hostname} rank={rank} "
        f"BaseModel.preproc_content={preproc_file}:{preproc_line}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    _set_torchrun_cuda_device()
    _print_bootstrap_diag()
    runpy.run_path("run.py", run_name="__main__")
