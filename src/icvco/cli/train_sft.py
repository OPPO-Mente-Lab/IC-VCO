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
# Based on Hugging Face TRL's trl/scripts/sft.py:
# https://github.com/huggingface/trl/blob/v0.26.0/trl/scripts/sft.py
# Changes include IC-VCO multimodal data processing, training configuration,
# checkpoint handling, logging, and model-saving behavior.


import os
import sys
from datetime import datetime
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
import random
import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from PIL import Image

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from peft import PeftModel

from ..config import MODULE_KEYWORDS
from ..dataset_registry import (
    cast_image_columns_decode_false,
)
from ..logging_utils import TrainerLogger, global_main_process_first, rank0_print, save_config_snapshot

PEFT_ADAPTER_DIRNAME = "adapter_output"


@dataclass
class DataArguments(ScriptArguments):
    max_samples: int = field(
        default=None,
        metadata={"help": "Maximum number of samples to use for training."},
    )

@dataclass
class TrainingArguments(SFTConfig):
    use_single_image: bool = field(
        default=True, metadata={"help": "Whether to include single-image samples in SFT training."}
    )

    use_multi_image: bool = field(
        default=False, metadata={"help": "Whether to use multi-image for training."}
    )

    use_symmetrical_optimization: bool = field(
        default=True,
        metadata={"help": "Whether to train on both chosen and reversed rejected pairings when available."},
    )

    use_rejected_samples: bool = field(
        default=False,
        metadata={"help": "When use_symmetrical_optimization=false, keep the rejected branch instead of the chosen branch."},
    )

    # use_mixed_image: bool = field(
    #     default=False, metadata={"help": "Whether to both edited and synthetic images for training."}
    # )

    max_length: int = field(
        default=None,
        metadata={"help": "Maximum sequence length for the model."},
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


def read_image(img):
    if isinstance(img, str):
        with Image.open(img) as opened_img:
            if opened_img.mode != 'RGB':
                return opened_img.convert('RGB')
            return opened_img.copy()
    elif isinstance(img, dict):
        image_path = img.get("path")
        image_bytes = img.get("bytes")
        if image_path:
            with Image.open(image_path) as opened_img:
                if opened_img.mode != 'RGB':
                    return opened_img.convert('RGB')
                return opened_img.copy()
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as opened_img:
                if opened_img.mode != 'RGB':
                    return opened_img.convert('RGB')
                return opened_img.copy()
        raise ValueError("Unsupported image dict: expected `path` or `bytes`.")
    else:
        assert isinstance(img, Image.Image)
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img.copy()


def get_image_size(img):
    if isinstance(img, str):
        with Image.open(img) as opened_img:
            return opened_img.size
    if isinstance(img, dict):
        image_path = img.get("path")
        image_bytes = img.get("bytes")
        if image_path:
            with Image.open(image_path) as opened_img:
                return opened_img.size
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as opened_img:
                return opened_img.size
        raise ValueError("Unsupported image dict: expected `path` or `bytes`.")
    assert isinstance(img, Image.Image)
    return img.size


def resolve_sft_images(features):
    is_batched = isinstance(features.get("prompt"), list)
    if not is_batched:
        resolved = dict(features)
        resolved["images"] = [read_image(image) for image in features["images"]]
        return resolved

    resolved = dict(features)
    resolved["images"] = [[read_image(image) for image in sample_images] for sample_images in features["images"]]
    return resolved


def _standard_sft_image_path(value, image_base_dir):
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if os.path.isabs(expanded):
            return expanded
        return str((Path(image_base_dir) / expanded).resolve())
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        normalized = dict(value)
        normalized["path"] = _standard_sft_image_path(value["path"], image_base_dir)
        return normalized
    return value


def make_standard_sft_data(examples, prompt_col, completion_col, images_file_names_col, image_base_dir):
    sft_examples = {
        "prompt": [],
        "completion": [],
        "images": [],
    }
    for index in range(len(examples[prompt_col])):
        prompt_text = examples[prompt_col][index]
        completion_text = examples[completion_col][index]
        image_names = examples[images_file_names_col][index]
        if isinstance(image_names, str):
            image_names = [image_names]
        if not image_names:
            raise ValueError("Standard SFT rows must include at least one image path.")

        sft_examples["prompt"].append(
            [
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in image_names]
                    + [{"type": "text", "text": prompt_text}],
                }
            ]
        )
        sft_examples["completion"].append(
            [{"role": "assistant", "content": [{"type": "text", "text": completion_text}]}]
        )
        sft_examples["images"].append(
            [_standard_sft_image_path(image_name, image_base_dir) for image_name in image_names]
        )
    return sft_examples



def make_multi_image_prompts(prompt, conversational=True):
    instruction = " Answer based on the {INDEX} image."
    prompt1 = prompt + instruction.format(INDEX="first")
    prompt2 = prompt + instruction.format(INDEX="second")
    if conversational:
        prompt1 = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt1}]}]
        prompt2 = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt2}]}]
    return prompt1, prompt2

def make_sft_data(
    examples,
    prompt_col,
    image1_col,
    image2_col,
    resp1_col,
    resp2_col,
    use_single_image,
    use_multi_image,
    use_symmetrical_optimization,
    use_rejected_samples,
):
    sft_examples = {
        "prompt": [],
        "completion": [],
        "images": []
    }
    for i in range(len(examples[prompt_col])):
        prompt = examples[prompt_col][i]
        image1 = examples[image1_col][i]
        image2 = examples[image2_col][i]
        image1_size = get_image_size(image1)
        image2_size = get_image_size(image2)
        if max(image1_size[0] * image1_size[1], image2_size[0] * image2_size[1]) > 1024 * 1024:
            continue
        has_synthetic = 'synthetic_image' in examples.keys()
        image3 = examples['synthetic_image'][i] if has_synthetic else None
        conv_prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        conv_resp1 = [{"role": "assistant", "content": [{"type": "text", "text": examples[resp1_col][i]}]}]
        conv_resp2 = [{"role": "assistant", "content": [{"type": "text", "text": examples[resp2_col][i]}]}]
        conv_resp3 = [{"role": "assistant", "content": [{"type": "text", "text": examples['synthetic_response'][i]}]}] if has_synthetic else None

        if use_single_image:
            preferred_single_image = image2 if use_rejected_samples else image1
            preferred_single_resp = conv_resp2 if use_rejected_samples else conv_resp1
            sft_examples['prompt'].append(conv_prompt)
            sft_examples['completion'].append(preferred_single_resp)
            sft_examples['images'].append([preferred_single_image])

            if use_symmetrical_optimization:
                sft_examples['prompt'].append(conv_prompt)
                sft_examples['completion'].append(conv_resp2)
                sft_examples['images'].append([image2])

        # if training_args.use_mixed_image:
        if use_single_image and has_synthetic:
            if use_symmetrical_optimization or use_rejected_samples:
                sft_examples['prompt'].append(conv_prompt)
                sft_examples['completion'].append(conv_resp3)
                sft_examples['images'].append([image3])

        # if training_args.use_multi_image:
        #     if random.random() < 0.5:
        #         image1, image2 = image2, image1
        #         conv_resp1, conv_resp2 = conv_resp2, conv_resp1
        #     conv_prompt1, conv_prompt2 = make_multi_image_prompts(prompt, conversational=True)
        #     sft_examples['prompt'].append(conv_prompt1)
        #     sft_examples['completion'].append(conv_resp1)
        #     sft_examples['images'].append([image1, image2])
        #     sft_examples['prompt'].append(conv_prompt2)
        #     sft_examples['completion'].append(conv_resp2)
        #     sft_examples['images'].append([image1, image2])

        if use_multi_image:
            conv_prompt1, conv_prompt2 = make_multi_image_prompts(prompt, conversational=True)
            sft_examples['images'].append([image1, image2])
            if use_rejected_samples and not use_symmetrical_optimization:
                sft_examples['prompt'].append(conv_prompt2)
                sft_examples['completion'].append(conv_resp2)
            else:
                sft_examples['prompt'].append(conv_prompt1)
                sft_examples['completion'].append(conv_resp1)

            if use_symmetrical_optimization:
                sft_examples['images'].append([image1, image2])
                sft_examples['prompt'].append(conv_prompt2)
                sft_examples['completion'].append(conv_resp2)

            if has_synthetic:
                conv_prompt1, conv_prompt3 = make_multi_image_prompts(prompt, conversational=True)
                sft_examples['images'].append([image1, image3])
                if use_rejected_samples and not use_symmetrical_optimization:
                    sft_examples['prompt'].append(conv_prompt3)
                    sft_examples['completion'].append(conv_resp3)
                else:
                    sft_examples['prompt'].append(conv_prompt1)
                    sft_examples['completion'].append(conv_resp1)

                if use_symmetrical_optimization:
                    sft_examples['images'].append([image1, image3])
                    sft_examples['prompt'].append(conv_prompt3)
                    sft_examples['completion'].append(conv_resp3)
    return sft_examples




def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = TrlParser((DataArguments, TrainingArguments, ModelArguments))
    data_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.max_length = None

    if not training_args.use_single_image and not training_args.use_multi_image:
        raise ValueError("At least one of `use_single_image` or `use_multi_image` must be enabled for SFT.")


    logger = TrainerLogger(training_args.logging_dir)

    ################
    # Model
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
    
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

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

    ################
    # Dataset
    ################
    dataset_map_num_proc = training_args.dataset_num_proc if training_args.dataset_num_proc and training_args.dataset_num_proc > 1 else None
    dataset_map_batch_size = 32
    dataset_writer_batch_size = 32

    with global_main_process_first(training_args, desc="SFT dataset load and cache preparation"):
        dataset = load_dataset(data_args.dataset_name, "sft")

    if data_args.max_samples:
        max_train_samples = min(data_args.max_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(max_train_samples))

    dataset = cast_image_columns_decode_false(dataset, ["images"])
    for split_name in dataset.keys():
        if "images" in dataset[split_name].column_names:
            dataset[split_name].set_transform(resolve_sft_images)
    
    
    ################
    # Training
    ################
    peft_config = get_peft_config(model_args)
    if model_args.use_peft and peft_config is None:
        raise ValueError("use_peft=True but get_peft_config returned None; check LoRA arguments.")

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        callbacks=[logger],
    )

    trainable_params = [(name, param) for name, param in trainer.model.named_parameters() if param.requires_grad]
    trainable_param_count = sum(param.numel() for _, param in trainable_params)
    total_param_count = sum(param.numel() for _, param in trainer.model.named_parameters())
    trainable_pct = 100.0 * trainable_param_count / total_param_count if total_param_count else 0.0
    rank0_print(
        "Trainable parameters after trainer setup: "
        f"{trainable_param_count} / {total_param_count} ({trainable_pct:.6f}%)."
    )
    if model_args.use_peft:
        rank0_print(f"PEFT model active: {isinstance(trainer.model, PeftModel)}")
    for name, _ in trainable_params[:20]:
        rank0_print(f"\t{name}")
    if len(trainable_params) > 20:
        rank0_print(f"\t... {len(trainable_params) - 20} more trainable tensors")
    if trainable_param_count == 0:
        raise RuntimeError(
            "No trainable parameters found after SFTTrainer setup. "
            "This would detach the training loss from autograd; check PEFT/LoRA setup and runtime package pins."
        )

    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            resume_from_checkpoint = last_checkpoint
            rank0_print(f"Resuming SFT from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if model_args.use_peft:
        # Keep the adapter artifact out of the merged model root. Transformers
        # auto-detects adapter_config.json and would otherwise reload the path
        # as a PEFT adapter checkpoint instead of a plain merged model.
        adapter_output_dir = os.path.join(training_args.output_dir, PEFT_ADAPTER_DIRNAME)
        trainer.save_model(adapter_output_dir)
        rank0_print("Adapter saved to", adapter_output_dir)
        rank0_print("Attempting to merge weights...")
        if hasattr(trainer.model, "module"):
            peft_model = trainer.model.module
        else:
            peft_model = trainer.model
        if isinstance(peft_model, PeftModel):
            merged_model = peft_model.merge_and_unload()
            merged_model.config.dtype = torch.bfloat16
            merged_model = merged_model.to(torch.bfloat16)
            merged_model.save_pretrained(training_args.output_dir, safe_serialization=True)
            rank0_print("Merged model saved to", training_args.output_dir)
        else:
            rank0_print("Model is not a PeftModel, skipping merge.")
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
                "entrypoint": "icvco.cli.train_sft",
                "generated_at": datetime.now().isoformat(),
                "command": sys.argv,
                "resume_from_checkpoint": resume_from_checkpoint,
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
