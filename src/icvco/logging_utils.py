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

from contextlib import contextmanager
import inspect

from transformers import TrainerCallback
from datetime import datetime
from pathlib import Path
import dataclasses
import enum
import json
import os
from typing import Dict, Any, Optional

import torch.distributed as dist

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


@contextmanager
def global_main_process_first(training_args, desc: str):
    """Run a block with global rank-0 entering first across all nodes when supported."""
    main_process_first = getattr(training_args, "main_process_first", None)
    if main_process_first is None:
        yield
        return

    kwargs = {"desc": desc}
    try:
        signature = inspect.signature(main_process_first)
        if "local" in signature.parameters:
            kwargs["local"] = False
    except (TypeError, ValueError):
        pass

    with main_process_first(**kwargs):
        yield


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {key: _snapshot_value(val) for key, val in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _snapshot_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_snapshot_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _snapshot_value(val) for key, val in vars(value).items()}
    return str(value)


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)

    text = str(value)
    if (
        text == ""
        or text.strip() != text
        or "\n" in text
        or any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", ">", "%", "@", "`"])
        or text.lower() in {"null", "true", "false", "yes", "no", "on", "off"}
    ):
        return json.dumps(text, ensure_ascii=False)
    return text


def _yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent

    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines

    return [f"{prefix}{_yaml_scalar(value)}"]


def save_config_snapshot(output_dir: str, payload: Dict[str, Any]) -> str:
    snapshot_path = Path(output_dir) / "config_snapshot.yaml"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_payload = _snapshot_value(payload)
    snapshot_path.write_text("\n".join(_yaml_lines(normalized_payload)) + "\n", encoding="utf-8")
    return str(snapshot_path)

class TrainerLogger(TrainerCallback):
    """
    通用的 JSONL 日志记录器，自动记录所有可用指标
    仅在主进程 (Rank 0) 记录训练指标
    """
    def __init__(
        self, 
        log_dir: str,
        log_filename: str = "trainer_log.jsonl",
        include_timestamp: bool = True,
        log_on_step: bool = True,
        log_on_epoch_end: bool = True,
    ):
        self.log_dir = log_dir
        self.log_filename = log_filename
        self.log_file = os.path.join(log_dir, log_filename)
        
        self.include_timestamp = include_timestamp
        self.log_on_step = log_on_step
        self.log_on_epoch_end = log_on_epoch_end
        
        # 注意：不要在 __init__ 中进行文件写操作 (open/write)
        # 因为所有 rank 都会执行 __init__，且此时无法通过 state 判断 rank
        
    def _log_metadata(self, data: Dict[str, Any]):
        """记录元数据 (内部辅助函数)"""
        # 这里不需要再次判断 rank，因为调用它的上层函数会判断
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def _extract_metrics(self, logs: Dict[str, Any], state) -> Dict[str, Any]:
        """自动提取所有可用的指标"""
        log_entry = {
            "step": state.global_step,
            "epoch": round(state.epoch, 6) if state.epoch is not None else None,
        }
        
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.now().isoformat()
        
        exclude_keys = {'total_flos', 'epoch'}
        
        for key, value in logs.items():
            if key not in exclude_keys:
                if isinstance(value, (int, float, str, bool, type(None))):
                    log_entry[key] = value
                elif hasattr(value, 'item'):  # PyTorch tensor
                    log_entry[key] = value.item()
                else:
                    log_entry[key] = str(value)
        
        return log_entry

    def on_train_begin(self, args, state, control, **kwargs):
        """
        [新增] 训练开始时调用
        用于安全地初始化日志文件，确保只有 Rank 0 执行
        """
        if not state.is_world_process_zero:
            return

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 清空或创建文件 (仅 Rank 0)
        open(self.log_file, 'w').close()

        # 记录开始时间
        self._log_metadata({
            "event": "training_start",
            "timestamp": datetime.now().isoformat(),
            "model_name": args.output_dir, # 可选：记录一些参数信息
        })

    def on_log(self, args, state, control, logs: Optional[Dict] = None, **kwargs):
        """每次日志记录时调用"""
        # [修改] 增加 Rank 0 检查
        if not state.is_world_process_zero:
            return
            
        if logs is None or not self.log_on_step:
            return
        
        log_entry = self._extract_metrics(logs, state)
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个 epoch 结束时记录"""
        # [修改] 增加 Rank 0 检查
        if not state.is_world_process_zero:
            return

        if self.log_on_epoch_end:
            self._log_metadata({
                "event": "epoch_end",
                "epoch": state.epoch,
                "step": state.global_step,
                "timestamp": datetime.now().isoformat(),
            })
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时记录"""
        # [修改] 增加 Rank 0 检查
        if not state.is_world_process_zero:
            return

        self._log_metadata({
            "event": "training_end",
            "total_steps": state.global_step,
            "timestamp": datetime.now().isoformat(),
        })
