# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

# Modified by OPPO in 2026.
# Based on Hugging Face TRL's trl/scripts/dpo.py:
# https://github.com/huggingface/trl/blob/v0.26.0/trl/scripts/dpo.py
# Changes include visual contrastive preference data processing, VCO trainer
# integration, checkpoint handling, logging, and model-saving behavior.


import os
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import torch
from datasets import load_dataset, Sequence
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from peft import  PeftModel, PeftConfig
from PIL import Image as PILImage
import random

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from peft import PeftModel

from ..config import (
    MODULE_KEYWORDS,
)
from ..dataset_registry import (
    cast_image_columns_decode_false,
)
from ..logging_utils import TrainerLogger, global_main_process_first, rank0_print, save_config_snapshot
from ..trainers.vco_trainer import (
    VCOConfig,
    VCOTrainer,
    DataCollatorForVisualContrastivePreference,
    make_vco_data
)

PEFT_ADAPTER_DIRNAME = "adapter_output"


def _standard_preference_image_path(value, image_base_dir):
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if os.path.isabs(expanded):
            return expanded
        return str((Path(image_base_dir) / expanded).resolve())
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        normalized = dict(value)
        normalized["path"] = _standard_preference_image_path(value["path"], image_base_dir)
        return normalized
    return value


def _standard_preference_spans(value):
    if value is None:
        return []
    return [[int(span[0]), int(span[1])] for span in value]


def make_standard_preference_data(
    examples,
    prompt_col,
    chosen_col,
    rejected_col,
    images_file_names_col,
    image_base_dir,
    use_single_branch_token_mask,
    chosen_response_spans_col=None,
    rejected_response_spans_col=None,
):
    vco_examples = {
        "prompt": [],
        "resp_1": [],
        "resp_2": [],
        "image_1_src": [],
        "image_2_src": [],
        "swap_pair": [],
    }
    if use_single_branch_token_mask:
        vco_examples["resp_1_target_spans"] = []
        vco_examples["resp_2_target_spans"] = []

    for index in range(len(examples[prompt_col])):
        image_names = examples[images_file_names_col][index]
        if isinstance(image_names, str):
            image_names = [image_names]
        if len(image_names) != 2:
            raise ValueError("Standard preference rows must include exactly two image paths.")

        vco_examples["prompt"].append(examples[prompt_col][index])
        vco_examples["resp_1"].append(examples[chosen_col][index])
        vco_examples["resp_2"].append(examples[rejected_col][index])
        vco_examples["image_1_src"].append(_standard_preference_image_path(image_names[0], image_base_dir))
        vco_examples["image_2_src"].append(_standard_preference_image_path(image_names[1], image_base_dir))
        vco_examples["swap_pair"].append(random.random() > 0.5)

        if use_single_branch_token_mask:
            vco_examples["resp_1_target_spans"].append(
                _standard_preference_spans(examples[chosen_response_spans_col][index])
            )
            vco_examples["resp_2_target_spans"].append(
                _standard_preference_spans(examples[rejected_response_spans_col][index])
            )

    return vco_examples


@dataclass
class DataArguments(ScriptArguments):
    max_samples: int = field(
        default=None,
        metadata={"help": "Maximum number of samples to use for training."},
    )


@dataclass
class ModelArguments(ModelConfig):
    model_family_id: str = field(
        default=None,
        metadata={"help": "Model family ID."},
    )
    freeze_vit: bool = field(
        default=False, metadata={"help": "Whether to freeze ViT parameters in full finetuning."}
    )
    freeze_mlp: bool = field(
        default=False, metadata={"help": "Whether to freeze MLP parameters in full finetuning."}
    )


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = TrlParser((DataArguments, VCOConfig, ModelArguments))
    data_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    logger = TrainerLogger(training_args.logging_dir)

    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config


    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )


    if not model_args.use_peft:
        # freeze certain params
        vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
        if model_args.freeze_vit: 
            rank0_print(f"Vision encoder is freezed... including:")
            for module in vision_encoder_keys:
                rank0_print(f"\t{module}")
                eval(f"model.{module}").requires_grad_(False)
        
        vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
        if model_args.freeze_mlp: 
            rank0_print(f"Vision projector is freezed... including:")
            for module in vision_projector_keys:
                rank0_print(f"\t{module}")
                eval(f"model.{module}").requires_grad_(False)

        # print trainable parameters
        rank0_print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                rank0_print(f"\t{name}")



    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    # fix the potential inconsistent chat template
    processor.tokenizer.chat_template = processor.chat_template
    pad_token = training_args.pad_token or processor.tokenizer.pad_token or processor.tokenizer.eos_token
    pad_token_id = processor.tokenizer.convert_tokens_to_ids(pad_token)
    if pad_token_id is None:
        raise ValueError(
            f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
            f"`processing` ({processor.__class__.__name__}). Ensure that the `pad_token` exists "
            "in the vocabulary before using it as a padding token."
        )
    if "max_grid_side" in MODULE_KEYWORDS[model_args.model_family_id]:
        max_side = MODULE_KEYWORDS[model_args.model_family_id]['max_grid_side']
        
        # 1. 获取原始 grid (从 image_processor 获取)
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "image_grid_pinpoints"):
            original_grids = processor.image_processor.image_grid_pinpoints
            
            # 2. 动态过滤：只要有一条边超过 max_side 就扔掉
            new_grids = [g for g in original_grids if g[0] <= max_side and g[1] <= max_side]
            
            # 3. 【第一步】修改 Processor 的配置
            processor.image_processor.image_grid_pinpoints = new_grids
            rank0_print(f"Auto-filtered PROCESSOR grid points. Max side restricted to {max_side}. Count: {len(new_grids)}")
            
            # 4. 【关键修正】同步修改 Model Config
            # 这是解决 "split_with_sizes" 报错的核心
            if hasattr(model, "config") and hasattr(model.config, "image_grid_pinpoints"):
                model.config.image_grid_pinpoints = new_grids
                rank0_print(f"Sync: Updated MODEL.config image_grid_pinpoints.")
            
            # 5. 【防御性编程】如果是 LoRA/PEFT 模型，base_model 的 config 也要改
            if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
                if hasattr(model.base_model.config, "image_grid_pinpoints"):
                    model.base_model.config.image_grid_pinpoints = new_grids
                    rank0_print(f"Sync: Updated BASE_MODEL.config image_grid_pinpoints.")
                    
        else:
            rank0_print("Warning: Could not find 'image_grid_pinpoints' in processor. Skipping resize.")
    # if script_args.ignore_bias_buffers:
    #     # torch distributed hack
    #     model._ddp_params_and_buffers_to_ignore = [
    #         name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    #     ]

    ################
    # Dataset
    ################

    dataset_map_num_proc = training_args.dataset_num_proc if training_args.dataset_num_proc and training_args.dataset_num_proc > 1 else None
    dataset_map_batch_size = 32
    dataset_writer_batch_size = 32

    with global_main_process_first(training_args, desc="preference dataset load and cache preparation"):
        dataset = load_dataset(data_args.dataset_name, "preference")

    if data_args.max_samples:
        max_train_samples = min(data_args.max_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(max_train_samples))

    train_columns = set(dataset["train"].column_names)
    required_columns = {"prompt", "chosen", "rejected", "images"}
    missing_columns = sorted(required_columns - train_columns)
    if missing_columns:
        raise ValueError(
            f"Dataset '{data_args.dataset_name}' preference config is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    dataset = cast_image_columns_decode_false(dataset, ["images"])
    with global_main_process_first(training_args, desc="preference dataset formatting"):
        for split_name in list(dataset.keys()):
            dataset[split_name] = dataset[split_name].map(
                make_standard_preference_data,
                batched=True,
                batch_size=dataset_map_batch_size,
                writer_batch_size=dataset_writer_batch_size,
                num_proc=dataset_map_num_proc,
                remove_columns=dataset[split_name].column_names,
                fn_kwargs={
                    "prompt_col": "prompt",
                    "chosen_col": "chosen",
                    "rejected_col": "rejected",
                    "images_file_names_col": "images",
                    "image_base_dir": "",
                    "use_single_branch_token_mask": training_args.use_single_branch_token_mask,
                    "chosen_response_spans_col": "chosen_response_spans",
                    "rejected_response_spans_col": "rejected_response_spans",
                },
            )
    

    ################
    # Training
    ################
    trainer = VCOTrainer(
        model,
        ref_model=ref_model,
        processing_class=processor,
        args=training_args,
        data_collator=DataCollatorForVisualContrastivePreference(pad_token_id=pad_token_id),
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        callbacks=[logger],
    )

    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            resume_from_checkpoint = last_checkpoint
            rank0_print(f"Resuming VCO training from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


    if model_args.use_peft:
        # Keep the adapter artifact out of the merged model root. The merge
        # helper resolves adapter_output/ and writes the merged model at root,
        # so eval/loaders see a plain full-model checkpoint.
        adapter_output_dir = os.path.join(training_args.output_dir, PEFT_ADAPTER_DIRNAME)
        trainer.save_model(adapter_output_dir)
        rank0_print("Adapter saved to", adapter_output_dir)
        # rank0_print("Attempting to merge weights...")
        # if hasattr(trainer.model, "module"):
        #     peft_model = trainer.model.module
        # else:
        #     peft_model = trainer.model
        # if isinstance(peft_model, PeftModel):
        #     merged_model = peft_model.merge_and_unload()
        #     merged_model.config.dtype = torch.bfloat16
        #     merged_model = merged_model.to(torch.bfloat16)
        #     merged_model.save_pretrained(training_args.output_dir, safe_serialization=True)
        #     rank0_print("Merged model saved to", training_args.output_dir)
        # else:
        #     rank0_print("Model is not a PeftModel, skipping merge.")
    else:
        trainer.save_model(training_args.output_dir)
    
    if trainer.is_world_process_zero():
        # save the original tokenizer (as the chat template and processor config may not be corretly saved by trainer)
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        processor.save_pretrained(training_args.output_dir)
        save_config_snapshot(
            training_args.output_dir,
            {
                "entrypoint": "icvco.cli.train_vco",
                "generated_at": datetime.now().isoformat(),
                "command": sys.argv,
                "resume_from_checkpoint": resume_from_checkpoint,
                "method": "icvco",
                "train_samples": len(dataset["train"]),
                "eval_samples": len(dataset["validation"]) if "validation" in dataset else 0,
                "data_args": data_args,
                "training_args": training_args,
                "model_args": model_args,
            },
        )

    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
