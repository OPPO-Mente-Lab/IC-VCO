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
# Based on Hugging Face TRL's trl/trainer/dpo_trainer.py:
# https://github.com/huggingface/trl/blob/v0.26.0/trl/trainer/dpo_trainer.py
# Changes include multimodal data collation and preprocessing, visual-contrastive
# preference losses, token masking, and single- and multi-image training paths.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from functools import partial
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
import random
from trl.trainer.dpo_trainer import *

from .batch_builders import build_single_image_completion_token_mask
from .token_masks import prepare_token_mask as build_response_token_mask


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

def make_multi_image_prompts(prompt, processor):
    instruction = " Answer based on the {INDEX} image."
    prompt1_text = prompt + instruction.format(INDEX="first")
    prompt2_text = prompt + instruction.format(INDEX="second")
    
    conv_prompt1 = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt1_text}]}]
    conv_prompt2 = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt2_text}]}]
    
    # 2. 转换为字符串 (apply_chat_template)
    # 注意: 这里假设 processor 支持处理 list of dicts 并识别 "type": "image"
    prompt1 = processor.apply_chat_template(conv_prompt1, tokenize=False, add_generation_prompt=True)
    prompt2 = processor.apply_chat_template(conv_prompt2, tokenize=False, add_generation_prompt=True)
    
    return prompt1, prompt2


def make_single_image_prompt(prompt, processor):
    conv_prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(conv_prompt, tokenize=False, add_generation_prompt=True)


def make_vco_data(
    examples,
    prompt_col,
    image1_col,
    image2_col,
    resp1_col,
    resp2_col,
    use_single_branch_token_mask,
    resp1_target_span_col: Optional=None,
    resp2_target_span_col: Optional=None,  
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
        vco_examples["resp_1_target_spans"], vco_examples["resp_2_target_spans"] = [], []
        
    for i in range(len(examples[prompt_col])):
        prompt = examples[prompt_col][i]
        resp_1 = examples[resp1_col][i]
        resp_2 = examples[resp2_col][i]
        if use_single_branch_token_mask:
            resp_1_spans = examples[resp1_target_span_col][i] or []
            resp_2_spans = examples[resp2_target_span_col][i] or []
        else:
            resp_1_spans, resp_2_spans = None, None

        vco_examples["resp_1"].append(resp_1)
        vco_examples["resp_2"].append(resp_2)
        if use_single_branch_token_mask:
            vco_examples["resp_1_target_spans"].append(resp_1_spans)
            vco_examples["resp_2_target_spans"].append(resp_2_spans)
        
        vco_examples["image_1_src"].append(examples[image1_col][i])
        vco_examples["image_2_src"].append(examples[image2_col][i])
        # Keep the stochastic pair swap in the lazy transform path so the map
        # cache only stores lightweight metadata instead of decoded image data.
        vco_examples["swap_pair"].append(random.random() > 0.5)

        vco_examples["prompt"].append(prompt)
            
    return vco_examples

@dataclass
class VCOConfig(DPOConfig):
    use_single_image: bool = field(
        default=True, metadata={"help": "Whether to include single-image contexts in training."}
    )
    use_multi_image: bool = field(
        default=True, metadata={"help": "Whether to include multi-image IC-VCO contexts in training."}
    )
    use_single_branch_token_mask: bool = field(
        default=True,
        metadata={"help": "Whether to apply response token masks on the single-image IC-VCO branch."},
    )
    single_weight: float = field(
        default=1.75, metadata={"help": "Weight for the IC-VCO single-image term (paper lambda)."}
    )
    multi_weight: float = field(
        default=0.75,
        metadata={
            "help": (
                "Weight for the IC-VCO multi-image term."
            )
        },
    )
    add_anchor_loss: bool = field(
        default=True, metadata={"help": "Whether to include the IC-VCO anchor regularizer."}
    )
    anchor_delta: float = field(
        default=0.0, metadata={"help": "Delta for anchor loss."}
    )
    anchor_weight: float = field(
        default=1, metadata={"help": "Weight for anchor loss (paper eta in IC-VCO Eq. 13)."}
    )
    single_anchor_weight: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional branch-specific weight for the IC-VCO single-image anchor term. "
                "If unset, follows single_weight."
            )
        },
    )
    multi_anchor_weight: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional branch-specific weight for the IC-VCO multi-image anchor term. "
                "If unset, follows multi_weight."
            )
        },
    )
    add_vcdist: bool = field(
        default=True, metadata={"help": "Whether to include the IC-VCO visual contrastive distillation term."}
    )
    vcdist_weight: float = field(
        default=0.4, metadata={"help": "Weight for the IC-VCO visual contrastive distillation term (paper gamma in Eq. 13)."}
    )
    vcdist_threshold: float = field(
        default=0.5, metadata={"help": "Teacher correctness threshold for VCDist filtering."}
    )
    vcdist_filter: bool = field(
        default=True, metadata={"help": "Whether to filter VCDist updates by teacher correctness and confidence."}
    )
    vcdist_stopgrad: bool = field(
        default=True, metadata={"help": "Whether to stop gradients through the VCDist teacher branch."}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Must be set to False so the custom dataloader can work."}
    )   
    max_prompt_length: int = field(
        default=None, metadata={"help": "Maximum length of the prompt."}
    )
    max_completion_length: int = field(
        default=None, metadata={"help": "Maximum length of the completion."}
    )
    max_length: int = field(
        default=None, metadata={"help": "Maximum length of the sequence. If None, will use the maximum length of the model."}
    )

    def __post_init__(self):
        if not self.use_single_image:
            self.single_weight = 0.0

        if not self.use_multi_image:
            self.multi_weight = 0.0

        if self.add_vcdist and not (self.use_single_image and self.use_multi_image):
            self.add_vcdist = False
            self.vcdist_weight = 0.0

        if self.single_anchor_weight is None:
            self.single_anchor_weight = self.single_weight
        if self.multi_anchor_weight is None:
            self.multi_anchor_weight = self.multi_weight

        if not self.use_single_image and not self.use_multi_image:
            raise ValueError("At least one of `use_single_image` or `use_multi_image` must be enabled for VCO training.")

        super().__post_init__()

@dataclass
class DataCollatorForVisualContrastivePreference(DataCollatorForPreference):
    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        output = {}

        def to_typed_tensor(value, dtype):
            if isinstance(value, torch.Tensor):
                return value.to(dtype=dtype)
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value).to(dtype=dtype)
            if isinstance(value, list) and value:
                first_elem = value[0]
                if isinstance(first_elem, torch.Tensor):
                    return torch.stack([elem.to(dtype=dtype) for elem in value], dim=0)
                if isinstance(first_elem, np.ndarray):
                    return torch.from_numpy(np.stack(value)).to(dtype=dtype)
            return torch.tensor(value, dtype=dtype)

        # =================================================================
        # 1. 辅助函数：处理文本序列 (Input IDs + Attention Mask)
        # =================================================================
        def collate_text(key_name: str, padding_side: str = "right", ):
            """
            检查 batch 中是否存在该 key，如果存在则进行 Tensor 转换、生成 Mask 并 Padding。
            """
            if key_name not in examples[0]:
                return
            
            # 转换为 Tensor
            input_ids_list = [torch.tensor(example[key_name], dtype=torch.long) for example in examples]
            # 生成 Attention Mask (1 for real tokens, 0 for padding)
            attention_mask_list = [torch.ones_like(input_ids) for input_ids in input_ids_list] 
            
            # 执行 Padding
            output[key_name] = pad(input_ids_list, padding_value=self.pad_token_id, padding_side=padding_side)
            
            # 自动生成对应的 mask key (例如 image_1_prompt_input_ids -> image_1_prompt_attention_mask)
            attn_mask_key = key_name.replace("input_ids", "attention_mask")
            output[attn_mask_key] = pad(attention_mask_list, padding_value=0, padding_side=padding_side)

            token_mask_key = key_name.replace("input_ids", "token_mask")
            if token_mask_key in examples[0]:
                token_mask_ids = [torch.tensor(example[token_mask_key], dtype=torch.long) for example in examples]
                output[token_mask_key] = pad(token_mask_ids, padding_value=0, padding_side=padding_side)


            # 处理对应的 Token Type IDs (如果有)
            token_type_key = key_name.replace("prompt_input_ids", "token_type_ids").replace("input_ids", "token_type_ids")
            # 注意：上面的 replace 逻辑是为了覆盖 'image_1_prompt_input_ids' -> 'image_1_token_type_ids' 
            # 以及可能存在的 'resp_1_input_ids' -> 'resp_1_token_type_ids'
            
            # 修正：通常 key 命名比较特定，这里尝试直接查找
            token_type_key = key_name.replace("input_ids", "token_type_ids")
            if token_type_key in examples[0]:
                tt_ids = [torch.tensor(example[token_type_key], dtype=torch.long) for example in examples]
                output[token_type_key] = pad(tt_ids, padding_value=0, padding_side=padding_side)


        # =================================================================
        # 2. 处理 Prompts (通常使用 Left Padding)
        # =================================================================
        # 单图模式 Keys
        collate_text("image_1_prompt_input_ids", padding_side="left")
        collate_text("image_2_prompt_input_ids", padding_side="left")
        # 多图模式 Keys (基于你上一轮的 multi_image 函数)
        collate_text("multi_image_1_prompt_input_ids", padding_side="left")
        collate_text("multi_image_2_prompt_input_ids", padding_side="left")

        # =================================================================
        # 3. 处理 Responses (通常使用 Right Padding)
        # =================================================================
        collate_text("resp_1_input_ids", padding_side="right")
        collate_text("resp_2_input_ids", padding_side="right")


        # =================================================================
        # 4. 处理视觉特征 (Pixel Values & Masks) - 适配 (Num_Images, Num_Tiles, ...)
        # =================================================================
        
        vision_keys = [
            "image_1_pixel_values", 
            "image_2_pixel_values", 
            "multi_image_pixel_values"
        ]
        
        mask_keys = [
            "image_1_pixel_attention_mask", 
            "image_2_pixel_attention_mask", 
            "multi_image_pixel_attention_mask"
        ]
        
        # --- 核心修复：更鲁棒的模式检测 ---
        def check_is_anyres(examples, vision_keys):
            # 判据 1: 如果存在 image_sizes 字段，几乎肯定是 AnyRes (OneVision/NeXT)
            size_keys = ["image_1_image_sizes", "image_2_image_sizes", "multi_image_image_sizes"]
            for ex in examples:
                for sk in size_keys:
                    if sk in ex and ex[sk] is not None:
                        return True
            return False

        is_anyres_mode = check_is_anyres(examples, vision_keys)
        
        # 增加 Debug 信息 (只在 Rank 7 报错时有用，防止再次静默失败)
        # if not is_anyres_mode:
        #     print(f"[DEBUG] Collator detected Fixed-Res Mode. Sample Key Shape: {len(examples[0].get('image_1_pixel_values', []))}")

        # =================================================================
        # 分支 A: AnyRes 模式 (LLaVA-OneVision) -> 需要 Padding Tiles
        # =================================================================            
        if is_anyres_mode:
            def normalize_anyres_pixel_tensor(value):
                tensor = to_typed_tensor(value, torch.float32)
                if tensor.dim() == 6 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.dim() == 4:
                    tensor = tensor.unsqueeze(0)
                return tensor

            def normalize_anyres_mask_tensor(value):
                tensor = to_typed_tensor(value, torch.long)
                if tensor.dim() == 5 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                return tensor

            # 1. 寻找 Global Max Tiles (在 dim=1 上)
            global_max_tiles = 0
            for example in examples:
                for key in vision_keys:
                    if key in example and example[key] is not None:
                        val = normalize_anyres_pixel_tensor(example[key])
                        if val.dim() < 2:
                            raise ValueError(f"Unexpected AnyRes pixel tensor shape for {key}: {tuple(val.shape)}")
                        curr_tiles = val.shape[1]
                        if curr_tiles > global_max_tiles:
                            global_max_tiles = curr_tiles

            # 2. 定义 Padding 函数 (Dim=1)
            def pad_vision_tensor(tensor, target_n, is_mask=False):
                # Input: (num_images, num_tiles, ...)
                curr_n = tensor.shape[1]
                if curr_n == target_n: return tensor
                if curr_n > target_n:
                    raise ValueError(
                        f"AnyRes tile padding target {target_n} is smaller than tensor tiles {curr_n}; "
                        f"tensor shape={tuple(tensor.shape)}"
                    )
                
                diff = target_n - curr_n
                num_imgs = tensor.shape[0]
                rest_shape = tensor.shape[2:] 
                
                padding_shape = (num_imgs, diff, *rest_shape)
                padding = torch.zeros(padding_shape, dtype=tensor.dtype)
                return torch.cat([tensor, padding], dim=1)



            # # 3. 处理 Pixel Values (Pad + Stack)
            # for key in vision_keys:
            #     if any(key in ex for ex in examples):
            #         batch_tensors = []
            #         for example in examples:
            #             # 确保转 Tensor
            #             val = torch.tensor(example[key], dtype=torch.float32)
            #             padded_val = pad_vision_tensor(val, global_max_tiles, is_mask=False)
            #             batch_tensors.append(padded_val)
            #         output[key] = torch.stack(batch_tensors, dim=0)

            # 3. 处理 Pixel Values (Pad + Stack)
            for key in vision_keys:
                if any(key in ex for ex in examples):
                    batch_tensors = []
                    for example in examples:
                        val = normalize_anyres_pixel_tensor(example[key])
                        padded_val = pad_vision_tensor(val, global_max_tiles, is_mask=False)
                        
                        # [FIX 3] 必须恢复维度！(1, Max_T, 3, H, W) -> (Max_T, 3, H, W)
                        # 如果不 Squeeze，Collator 输出就是 (Batch, 1, Tiles...), 导致下游 5D 检查失败
                        if padded_val.shape[0] == 1:
                            padded_val = padded_val.squeeze(0)
                            
                        batch_tensors.append(padded_val)
                    output[key] = torch.stack(batch_tensors, dim=0)

        # 4. 处理 Attention Mask (Pad + Stack)
            for key in mask_keys:
                if any(key in ex for ex in examples):
                    batch_masks = []
                    for example in examples:
                        val = normalize_anyres_mask_tensor(example[key])
                        padded_val = pad_vision_tensor(val, global_max_tiles, is_mask=True)
                        
                        # [FIX 5] 同样需要恢复维度
                        if padded_val.shape[0] == 1:
                            padded_val = padded_val.squeeze(0)

                        batch_masks.append(padded_val)
                    output[key] = torch.stack(batch_masks, dim=0)

        # =================================================================
        # 分支 B: 固定分辨率模式 (LLaVA-Interleave) -> 直接 Stack
        # =================================================================
        else:
            # 直接 Stack，无需计算 Tiles，因为 H,W 是固定的
            # Input Shape: (Num_Images, C, H, W)
            # Output Shape: (Batch, Num_Images, C, H, W)
            
            # 处理 Pixel Values
            for key in vision_keys:
                if any(key in ex for ex in examples):
                    # 列表推导式 + torch.stack
                    # 注意：这里假设同一 key 下所有样本的 num_images 是一致的
                    # 如果 num_images 不一致（变长图文混排），则需要对 dim=0 进行 pad，但通常偏好数据是对齐的
                    tensors = [to_typed_tensor(ex[key], torch.float32) for ex in examples]
                    output[key] = torch.stack(tensors, dim=0)

            # 处理 Masks (如果有)
            for key in mask_keys:
                if any(key in ex for ex in examples):
                    masks = [to_typed_tensor(ex[key], torch.long) for ex in examples]
                    output[key] = torch.stack(masks, dim=0)

        # =================================================================
        # 5. 处理 Image Sizes (通用)
        # =================================================================
        # image_sizes 对于 AnyRes 是必须的，对于 Fixed Res 有时也需要用来做 Aspect Ratio 处理
        # 统一处理方式：Stack
        size_keys = [
            "image_1_image_sizes", "image_2_image_sizes", "multi_image_image_sizes"
        ]
        for k in size_keys:
            if k in examples[0]:
                val_list = [torch.tensor(ex[k], dtype=torch.long) for ex in examples]
                output[k] = torch.stack(val_list, dim=0)


        
        return output



class VCOTrainer(DPOTrainer):
    CORE_LOG_METRIC_SUFFIXES = frozenset(
        {
            "loss_combined",
            "single_loss",
            "multi_loss",
            "vcdist_loss",
            "loss_single_weighted",
            "loss_multi_weighted",
            "loss_anchor_weighted",
            "loss_vcdist_weighted",
            "single_rewards/accuracies",
            "multi_rewards/accuracies",
            "single_rewards/margins",
            "multi_rewards/margins",
            "vcdist_active_rate",
            "vcdist_teacher_acc",
        }
    )

    @classmethod
    def filter_core_log_metrics(cls, metrics: dict[str, float], train_eval: str) -> dict[str, float]:
        allowed_names = {f"{train_eval}_{suffix}" for suffix in cls.CORE_LOG_METRIC_SUFFIXES}
        return {key: value for key, value in metrics.items() if key in allowed_names}

    @staticmethod
    def is_custom_vco_dataset(dataset) -> bool:
        if not isinstance(dataset, Dataset):
            return False
        column_names = set(dataset.column_names)
        has_prompt_and_responses = {"prompt", "resp_1", "resp_2"}.issubset(column_names)
        has_image_refs = "image_1_src" in column_names or "image_1" in column_names
        return has_prompt_and_responses and has_image_refs

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        dataset_name: str,
    ):
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc nor writer_batch_size
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10

        with PartialState().main_process_first():
            if not self.is_custom_vco_dataset(dataset):
                # Extract prompt if needed
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
                dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

                # Apply the chat template if needed
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
                dataset = dataset.map(
                    maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
                )

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            # 使用 set_transform 替代 map，实现即时转换（lazy evaluation）
            # 避免预处理缓存，节省磁盘空间
            transform_fn = partial(
                self.process_row,
                processing_class=processing_class,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                add_special_tokens=False,
                use_multi_image=args.use_multi_image,
            )
            dataset.set_transform(transform_fn)
        return dataset

    @staticmethod
    def resolve_sample_features(
        features: dict[str, Any],
        processing_class,
        use_multi_image: bool = False,
    ) -> dict[str, Any]:
        resolved = dict(features)

        image_1_src = resolved.get("image_1_src", resolved.get("image_1"))
        image_2_src = resolved.get("image_2_src", resolved.get("image_2"))
        image_1 = read_image(image_1_src)
        image_2 = read_image(image_2_src)
        prompt_text = resolved["prompt"]

        resp_1 = resolved["resp_1"]
        resp_2 = resolved["resp_2"]
        resp_1_spans = resolved.get("resp_1_target_spans")
        resp_2_spans = resolved.get("resp_2_target_spans")

        if resolved.get("swap_pair", False):
            image_1, image_2 = image_2, image_1
            resp_1, resp_2 = resp_2, resp_1
            resp_1_spans, resp_2_spans = resp_2_spans, resp_1_spans

        resolved["image_1"] = image_1
        resolved["image_2"] = image_2
        resolved["prompt"] = make_single_image_prompt(prompt_text, processing_class)
        resolved["resp_1"] = resp_1
        resolved["resp_2"] = resp_2
        if use_multi_image:
            prompt_multi_1, prompt_multi_2 = make_multi_image_prompts(prompt_text, processing_class)
            resolved["prompt_multi_image_1"] = prompt_multi_1
            resolved["prompt_multi_image_2"] = prompt_multi_2
        if "resp_1_target_spans" in resolved and "resp_2_target_spans" in resolved:
            resolved["resp_1_target_spans"] = resp_1_spans
            resolved["resp_2_target_spans"] = resp_2_spans
        return resolved

    @staticmethod
    def process_row_single_image(
        features,
        processing_class,
        max_prompt_length = None,
        max_completion_length = None,
        add_special_tokens = True,
    ) -> dict[str, list[int]]:
        """
        Same as `tokenize_row` but for vision models. Please refer to `tokenize_row` for more information.
        """
        processor, tokenizer = processing_class, processing_class.tokenizer  # the processing class is a processor
        image_1_processed_features = processor(images=features['image_1'], text=features['prompt'], add_special_tokens=False)
        image_1_prompt_input_ids = image_1_processed_features["input_ids"][0]
        image_1_pixel_values = image_1_processed_features["pixel_values"][0]

        image_2_processed_features = processor(images=features['image_2'], text=features['prompt'], add_special_tokens=False)
        image_2_prompt_input_ids = image_2_processed_features["input_ids"][0]
        image_2_pixel_values = image_2_processed_features["pixel_values"][0]

        has_response_spans = 'resp_1_target_spans' in features and 'resp_2_target_spans' in features
        resp_1_features = tokenizer(features['resp_1'], add_special_tokens=False, return_offsets_mapping=has_response_spans)
        resp_2_features = tokenizer(features['resp_2'], add_special_tokens=False, return_offsets_mapping=has_response_spans)
        resp_1_input_ids = resp_1_features['input_ids']
        resp_2_input_ids = resp_2_features['input_ids']

        if has_response_spans:
            resp_1_token_mask = build_response_token_mask(
                resp_1_input_ids,
                resp_1_features["offset_mapping"],
                features["resp_1_target_spans"],
            )
            resp_2_token_mask = build_response_token_mask(
                resp_2_input_ids,
                resp_2_features["offset_mapping"],
                features["resp_2_target_spans"],
            )
        else:
            resp_1_token_mask, resp_2_token_mask = None, None


        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                image_1_prompt_input_ids = [tokenizer.bos_token_id] + image_1_prompt_input_ids
                image_2_prompt_input_ids = [tokenizer.bos_token_id] + image_2_prompt_input_ids
            if tokenizer.eos_token_id is not None:
                image_1_prompt_input_ids = image_1_prompt_input_ids + [tokenizer.eos_token_id]
                image_2_prompt_input_ids = image_2_prompt_input_ids + [tokenizer.eos_token_id]

        resp_1_input_ids = resp_1_input_ids + [tokenizer.eos_token_id]
        resp_2_input_ids = resp_2_input_ids + [tokenizer.eos_token_id]
        if has_response_spans:
            resp_1_token_mask += [0]
            resp_2_token_mask += [0]


        # Truncate prompt and completion sequences
        # if max_prompt_length is not None:
        #     image_1_prompt_input_ids = image_1_prompt_input_ids[-max_prompt_length:]
        #     image_2_prompt_input_ids = image_2_prompt_input_ids[-max_prompt_length:]
                
        if max_completion_length is not None:
            resp_1_input_ids = resp_1_input_ids[:max_completion_length]
            resp_2_input_ids = resp_2_input_ids[:max_completion_length]
            if has_response_spans:
                resp_1_token_mask = resp_1_token_mask[:max_completion_length]
                resp_2_token_mask = resp_2_token_mask[:max_completion_length]

        output = {
            "image_1_prompt_input_ids": image_1_prompt_input_ids,
            "image_2_prompt_input_ids": image_2_prompt_input_ids,
            "image_1_pixel_values": image_1_pixel_values,
            "image_2_pixel_values": image_2_pixel_values,
            "resp_1_input_ids": resp_1_input_ids,
            "resp_2_input_ids": resp_2_input_ids,
        }
        if has_response_spans:
            output["resp_1_token_mask"] = resp_1_token_mask
            output["resp_2_token_mask"] = resp_2_token_mask

        # if "pixel_attention_mask" in image_1_processed_features:
        #     output["image_1_pixel_attention_mask"] = image_1_processed_features["pixel_attention_mask"][0]
        #     output["image_2_pixel_attention_mask"] = image_2_processed_features["pixel_attention_mask"][0]




        if "image_sizes" in image_1_processed_features: # AnyRes Mode
            output["image_1_image_sizes"] = image_1_processed_features["image_sizes"][0]
            output["image_2_image_sizes"] = image_2_processed_features["image_sizes"][0]

            # =======================================================
            # [FIX] 强制提取或生成 Pixel Attention Mask
            # =======================================================
            def ensure_pixel_mask(processed_features, key_prefix):
                # 1. 尝试直接从 processor 输出中获取
                if "pixel_attention_mask" in processed_features:
                    return processed_features["pixel_attention_mask"][0]
                
                # 2. 如果没有 (Processor 没返回)，则手动生成全 1 Mask
                # Pixel Values Shape: (Tiles, C, H, W)
                # Target Mask Shape:  (Tiles, H, W)
                pv = processed_features["pixel_values"][0]
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv)
                
                # 生成全 1 mask (表示所有像素都有效)
                # shape 取 pv 的 (Tiles, H, W) -> 也就是 (dim 0, dim 2, dim 3)
                mask_shape = (pv.shape[0], pv.shape[2], pv.shape[3])
                return torch.ones(mask_shape, dtype=torch.long)

            # 注入到 output 字典中
            output["image_1_pixel_attention_mask"] = ensure_pixel_mask(image_1_processed_features, "image_1")
            output["image_2_pixel_attention_mask"] = ensure_pixel_mask(image_2_processed_features, "image_2")

        if "token_type_ids" in image_1_processed_features:
            output["image_1_token_type_ids"] = image_1_processed_features["token_type_ids"][0]
            output["image_2_token_type_ids"] = image_2_processed_features["token_type_ids"][0]

        if "image_grid_thw" in image_1_processed_features:
            output["image_1_image_grid_thw"] = image_1_processed_features["image_grid_thw"].squeeze(0)
            output["image_2_image_grid_thw"] = image_2_processed_features["image_grid_thw"].squeeze(0)

        return output
    
    @staticmethod
    def process_row_multi_image(
        features,
        processing_class,
        max_prompt_length = None,
        add_special_tokens = True,
    ) -> dict[str, list[int]]:
        processor, tokenizer = processing_class, processing_class.tokenizer

        # 1. 准备图像数据
        # 假设 features['image_X'] 是列表格式，这里将两组图片合并作为上下文
        # 如果是单张图片对象，可能需要调整为 [features['image_1'], features['image_2']]
        combined_images = [features["image_1"], features["image_2"]]
        num_images_in_context = 2

        # 2. 调用 Processor 处理两个不同的 Prompt
        processed_features_1 = processor(
            images=combined_images, 
            text=features["prompt_multi_image_1"], 
            add_special_tokens=False
        )
        processed_features_2 = processor(
            images=combined_images, 
            text=features["prompt_multi_image_2"], 
            add_special_tokens=False
        )


        # 3. 提取基本特征 (Input IDs 和 Pixel Values)
        multi_image_1_prompt_input_ids = processed_features_1["input_ids"][0]
        multi_image_pixel_values = processed_features_1["pixel_values"]
        
        multi_image_2_prompt_input_ids = processed_features_2["input_ids"][0]


        # # Truncate prompt and completion sequences
        # if max_prompt_length is not None:
        #     multi_image_1_prompt_input_ids = multi_image_1_prompt_input_ids[-max_prompt_length:]
        #     multi_image_2_prompt_input_ids = multi_image_2_prompt_input_ids[-max_prompt_length:]

        # 5. 构建输出字典
        output = {
            "multi_image_1_prompt_input_ids": multi_image_1_prompt_input_ids,
            "multi_image_2_prompt_input_ids": multi_image_2_prompt_input_ids,
            "multi_image_pixel_values": multi_image_pixel_values
        }

        # 6. 处理可选的视觉特征 (Masks, Sizes, Token Types)
        # 许多新模型(如Qwen-VL, Idefics2)需要这些额外参数
        
        # 处理 Prompt 1 的额外特征
        # if "pixel_attention_mask" in processed_features_1:
        #     output["multi_image_pixel_attention_mask"] = processed_features_1["pixel_attention_mask"][0]
        if "image_sizes" in processed_features_1: # AnyRes Mode
            output["multi_image_image_sizes"] = processed_features_1["image_sizes"][0]

            # =================================================================
            # [FIX] 强制提取或生成 Pixel Attention Mask (多图版)
            # =================================================================
            # 逻辑：如果有 mask 就拿，没有就根据 pixel_values 形状造一个全 1 的
    

            def ensure_multi_image_pixel_mask(proc_feats, pv):
                # 1. 尝试直接获取
                if "pixel_attention_mask" in proc_feats:
                    return proc_feats["pixel_attention_mask"][0]
                
                # 2. 手动生成全 1 Mask
                
                # Case A: pv 是 Tensor -> 直接用
                if isinstance(pv, torch.Tensor):
                    pv_tensor = pv
                    
                # Case B: pv 是 List -> 需要转换
                elif isinstance(pv, list):
                    if len(pv) > 0:
                        # 检查 list 里的元素类型
                        first_elem = pv[0]
                        if isinstance(first_elem, torch.Tensor):
                            pv_tensor = torch.stack(pv)
                        elif isinstance(first_elem, np.ndarray):
                            # [FIX] 先用 numpy stack，再转 tensor，速度快且无 warning
                            pv_tensor = torch.from_numpy(np.stack(pv))
                        else:
                            # Fallback: 可能是 list of list，或者其他 weird format
                            pv_tensor = torch.tensor(pv)
                    else:
                        # 空 list
                        pv_tensor = torch.tensor([])
                
                # Case C: pv 是 ndarray
                elif isinstance(pv, np.ndarray):
                    pv_tensor = torch.from_numpy(pv)
                
                else:
                    # Fallback
                    pv_tensor = torch.tensor(pv)

                mask_shape = list(pv_tensor.shape)
                c_dim_index = -3 # C 在倒数第3维
                del mask_shape[c_dim_index] # 删除 C 维度
                
                return torch.ones(mask_shape, dtype=torch.long)

            # 生成 mask
            pixel_mask = ensure_multi_image_pixel_mask(processed_features_1, multi_image_pixel_values)
            output["multi_image_pixel_attention_mask"] = pixel_mask
            # =================================================================

            # 4. 添加特殊 Token (BOS / EOS)
            # 逻辑与 process_row_single_image 保持完全一致
            if add_special_tokens:
                if tokenizer.bos_token_id is not None:
                    multi_image_1_prompt_input_ids = [tokenizer.bos_token_id] + multi_image_1_prompt_input_ids
                    multi_image_2_prompt_input_ids = [tokenizer.bos_token_id] + multi_image_2_prompt_input_ids
                
                if tokenizer.eos_token_id is not None:
                    multi_image_1_prompt_input_ids = multi_image_1_prompt_input_ids + [tokenizer.eos_token_id]
                    multi_image_2_prompt_input_ids = multi_image_2_prompt_input_ids + [tokenizer.eos_token_id]


        if "token_type_ids" in processed_features_1:
            output["multi_image_token_type_ids"] = processed_features_1["token_type_ids"][0]

        if "image_grid_thw" in processed_features_1:
            output["multi_image_grid_thw"] = processed_features_1["image_grid_thw"]


        # 处理 Prompt 2 的额外特征
        # if "pixel_attention_mask" in processed_features_2:
        #     output["multi_image_2_pixel_attention_mask"] = processed_features_2["pixel_attention_mask"][0]
        # if "image_sizes" in processed_features_2:
        #     output["multi_image_2_image_sizes"] = processed_features_2["image_sizes"][0]
        if "token_type_ids" in processed_features_2:
            output["multi_image_2_token_type_ids"] = processed_features_2["token_type_ids"][0]

        return output


    @staticmethod
    def process_row(
        features,
        processing_class,
        max_prompt_length = None,
        max_completion_length = None,
        add_special_tokens: bool = True,
        use_multi_image: bool = False,

    ) -> dict[str, list[int]]:
        # 检测是否为 batch 格式（set_transform 在 DataLoader 批量获取时传入 batch）
        # 通过检查 'prompt' 字段是否为列表来判断
        is_batched = isinstance(features.get('prompt'), list)
        
        if not is_batched:
            # 单样本处理（原逻辑）
            prepared_features = VCOTrainer.resolve_sample_features(
                features,
                processing_class=processing_class,
                use_multi_image=use_multi_image,
            )
            output = VCOTrainer.process_row_single_image(
                features=prepared_features, 
                processing_class=processing_class,
                max_prompt_length=max_prompt_length, 
                max_completion_length=max_completion_length, 
                add_special_tokens=add_special_tokens
            )
            if use_multi_image:
                output.update(VCOTrainer.process_row_multi_image(
                    features=prepared_features, 
                    processing_class=processing_class,
                    max_prompt_length=max_prompt_length, 
                    add_special_tokens=add_special_tokens
                ))
            return output
        
        # batch 处理：逐样本处理后合并
        batch_size = len(features['prompt'])
        all_outputs = []
        
        for i in range(batch_size):
            # 提取第 i 个样本的 features
            single_features = {k: v[i] for k, v in features.items()}
            prepared_features = VCOTrainer.resolve_sample_features(
                single_features,
                processing_class=processing_class,
                use_multi_image=use_multi_image,
            )
            
            # 单样本处理
            single_output = VCOTrainer.process_row_single_image(
                features=prepared_features, 
                processing_class=processing_class,
                max_prompt_length=max_prompt_length, 
                max_completion_length=max_completion_length, 
                add_special_tokens=add_special_tokens
            )
            if use_multi_image:
                single_output.update(VCOTrainer.process_row_multi_image(
                    features=prepared_features, 
                    processing_class=processing_class,
                    max_prompt_length=max_prompt_length, 
                    add_special_tokens=add_special_tokens
                ))
            all_outputs.append(single_output)
        
        # 合并为 batch 格式：{key: [sample1_value, sample2_value, ...]}
        if not all_outputs:
            return {}
        batch_output = {k: [out[k] for out in all_outputs] for k in all_outputs[0].keys()}
        return batch_output

    
    
    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        batch: dict[str, list | torch.LongTensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Refactored: Modular execution for Single and Multi-Image tasks.
        """
        metrics = {}
        single_logps, multi_logps = None, None
        single_components, multi_components = None, None
        vcdist_loss = None

        if self.args.use_single_image:
            _, single_logps, single_components = self.run_single_image_preference(model, batch, metrics, train_eval)

        if self.args.use_multi_image:
            _, multi_logps, multi_components = self.run_multi_image_preference(model, batch, metrics, train_eval)

        if self.args.add_vcdist and single_logps is not None and multi_logps is not None:
            vcdist_loss, vcdist_metrics = self.soft_label_preference_loss(
                student_chosen_logps=single_logps["chosen"],
                student_rejected_logps=single_logps["rejected"],
                student_ref_chosen_logps=single_logps["ref_chosen"],
                student_ref_rejected_logps=single_logps["ref_rejected"],
                
                teacher_chosen_logps=multi_logps["chosen"],
                teacher_rejected_logps=multi_logps["rejected"],
                teacher_ref_chosen_logps=multi_logps["ref_chosen"],
                teacher_ref_rejected_logps=multi_logps["ref_rejected"],
                beta=self.args.beta,
                validation_threshold=self.args.vcdist_threshold,
            )

            def safe_ratio(numerator_key, denominator_key):
                denominator = global_stats[denominator_key].item()
                if denominator > 0:
                    return (global_stats[numerator_key] / global_stats[denominator_key]).item()
                return 0.0

            keys = list(vcdist_metrics.keys())
            # 2. Stack & All Reduce
            # 确保 vcdist_metrics 里所有 value 都是 Tensor 且在同一个 device
            local_stats = torch.stack([vcdist_metrics[k].float() for k in keys])

            if dist.is_initialized():
                dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

            # 3. 解包 & 计算 Mean
            global_stats = {k: v for k, v in zip(keys, local_stats)}

            valid_count = global_stats["vcdist_valid_count"].item()
            active_count = global_stats["vcdist_active_count"].item()
            global_count = global_stats["vcdist_global_count"].item() # 这就是全局的 Total Batch Size
            gate_pass_count = global_stats["vcdist_gate_pass_count"].item()
            trigger_count = global_stats["vcdist_trigger_count"].item()
            correctness_blocked_count = max(global_count - gate_pass_count, 0.0)
            confidence_blocked_count = max(gate_pass_count - trigger_count, 0.0)

            metrics[f"{train_eval}_vcdist_threshold"] = float(self.args.vcdist_threshold)
            metrics[f"{train_eval}_vcdist_loss"] = vcdist_loss.item()

            # 计算 Valid Metrics
            if valid_count > 0:
                metrics[f"{train_eval}_vcdist_valid_kl"] = (global_stats["vcdist_valid_kl_sum"] / valid_count).item()
            else:
                metrics[f"{train_eval}_vcdist_valid_kl"] = 0.0

            if active_count > 0:
                metrics[f"{train_eval}_vcdist_active_kl"] = (global_stats["vcdist_active_kl_sum"] / active_count).item()
            else:
                metrics[f"{train_eval}_vcdist_active_kl"] = 0.0
            
            # [新增] 计算 Global Metrics
            if global_count > 0:
                metrics[f"{train_eval}_vcdist_global_kl"] = (global_stats["vcdist_global_kl_sum"] / global_count).item()
            else:
                metrics[f"{train_eval}_vcdist_global_kl"] = 0.0

            # 记录样本数供参考
            metrics[f"{train_eval}_vcdist_valid_count"] = valid_count
            metrics[f"{train_eval}_vcdist_active_count"] = active_count
            metrics[f"{train_eval}_vcdist_global_count"] = global_count
            metrics[f"{train_eval}_vcdist_gate_pass_count"] = gate_pass_count
            metrics[f"{train_eval}_vcdist_trigger_count"] = trigger_count
            metrics[f"{train_eval}_vcdist_correctness_blocked_count"] = correctness_blocked_count
            metrics[f"{train_eval}_vcdist_confidence_blocked_count"] = confidence_blocked_count

            metrics[f"{train_eval}_vcdist_teacher_prob"] = safe_ratio("vcdist_teacher_prob_sum", "vcdist_global_count")
            metrics[f"{train_eval}_vcdist_student_prob"] = safe_ratio("vcdist_student_prob_sum", "vcdist_global_count")
            metrics[f"{train_eval}_vcdist_teacher_confidence"] = safe_ratio("vcdist_teacher_confidence_sum", "vcdist_global_count")
            metrics[f"{train_eval}_vcdist_teacher_acc"] = safe_ratio("vcdist_teacher_acc_count", "vcdist_global_count")
            metrics[f"{train_eval}_vcdist_confident_wrong_rate"] = safe_ratio("vcdist_confident_wrong_count", "vcdist_global_count")
            metrics[f"{train_eval}_vcdist_ambiguous_rate"] = safe_ratio("vcdist_ambiguous_count", "vcdist_global_count")
            metrics[f"{train_eval}_vcdist_active_rate"] = active_count / global_count if global_count > 0 else 0.0
            metrics[f"{train_eval}_vcdist_gate_pass_rate"] = gate_pass_count / global_count if global_count > 0 else 0.0
            metrics[f"{train_eval}_vcdist_trigger_rate"] = trigger_count / global_count if global_count > 0 else 0.0
            metrics[f"{train_eval}_vcdist_correctness_blocked_rate"] = correctness_blocked_count / global_count if global_count > 0 else 0.0
            metrics[f"{train_eval}_vcdist_confidence_blocked_rate"] = confidence_blocked_count / global_count if global_count > 0 else 0.0

            for pos_name in ["pos1", "pos2"]:
                if f"vcdist_{pos_name}_global_count" not in global_stats:
                    continue
                pos_count = global_stats[f"vcdist_{pos_name}_global_count"].item()
                metrics[f"{train_eval}_vcdist_{pos_name}_teacher_prob"] = safe_ratio(
                    f"vcdist_{pos_name}_teacher_prob_sum",
                    f"vcdist_{pos_name}_global_count",
                )
                metrics[f"{train_eval}_vcdist_{pos_name}_teacher_acc"] = safe_ratio(
                    f"vcdist_{pos_name}_teacher_acc_count",
                    f"vcdist_{pos_name}_global_count",
                )
                metrics[f"{train_eval}_vcdist_{pos_name}_gate_pass_rate"] = safe_ratio(
                    f"vcdist_{pos_name}_gate_pass_count",
                    f"vcdist_{pos_name}_global_count",
                )
                metrics[f"{train_eval}_vcdist_{pos_name}_trigger_rate"] = safe_ratio(
                    f"vcdist_{pos_name}_trigger_count",
                    f"vcdist_{pos_name}_global_count",
                )
                metrics[f"{train_eval}_vcdist_{pos_name}_global_count"] = pos_count

        metrics[f"{train_eval}_multi_weight"] = float(self.args.multi_weight)
        metrics[f"{train_eval}_single_weight"] = float(self.args.single_weight)
        metrics[f"{train_eval}_anchor_weight"] = float(self.args.anchor_weight)
        single_anchor_weight = (
            self.args.anchor_weight
            if self.args.single_anchor_weight is None
            else self.args.single_anchor_weight
        )
        multi_anchor_weight = (
            self.args.anchor_weight
            if self.args.multi_anchor_weight is None
            else self.args.multi_anchor_weight
        )
        metrics[f"{train_eval}_single_anchor_weight"] = float(single_anchor_weight)
        metrics[f"{train_eval}_multi_anchor_weight"] = float(multi_anchor_weight)
        metrics[f"{train_eval}_vcdist_weight"] = float(self.args.vcdist_weight)

        multi_total = None
        single_total = None

        if multi_components is not None:
            weighted_multi_loss = self.args.multi_weight * multi_components["base_loss"]
            multi_total = weighted_multi_loss + multi_components["aux_loss"]
            metrics[f"{train_eval}_loss_multi_main"] = multi_components["base_loss"].item()
            metrics[f"{train_eval}_loss_multi_weighted"] = weighted_multi_loss.item()
            metrics[f"{train_eval}_loss_multi_aux"] = multi_components["aux_loss"].item()

        if single_components is not None:
            weighted_single_loss = self.args.single_weight * single_components["base_loss"]
            single_total = weighted_single_loss + single_components["aux_loss"]
            metrics[f"{train_eval}_loss_single_main"] = single_components["base_loss"].item()
            metrics[f"{train_eval}_loss_single_weighted"] = weighted_single_loss.item()
            metrics[f"{train_eval}_loss_single_aux"] = single_components["aux_loss"].item()

        if self.args.add_anchor_loss:
            anchor_total = None
            weighted_anchor_total = None
            if single_components is not None:
                weighted_single_anchor = single_anchor_weight * single_components["anchor_loss"]
                single_total = weighted_single_anchor if single_total is None else single_total + weighted_single_anchor
                anchor_total = single_components["anchor_loss"]
                weighted_anchor_total = weighted_single_anchor
                metrics[f"{train_eval}_loss_single_anchor_weighted"] = weighted_single_anchor.item()
            if multi_components is not None:
                weighted_multi_anchor = multi_anchor_weight * multi_components["anchor_loss"]
                multi_total = weighted_multi_anchor if multi_total is None else multi_total + weighted_multi_anchor
                anchor_total = multi_components["anchor_loss"] if anchor_total is None else anchor_total + multi_components["anchor_loss"]
                weighted_anchor_total = (
                    weighted_multi_anchor
                    if weighted_anchor_total is None
                    else weighted_anchor_total + weighted_multi_anchor
                )
                metrics[f"{train_eval}_loss_multi_anchor_weighted"] = weighted_multi_anchor.item()
            if anchor_total is not None:
                metrics[f"{train_eval}_loss_anchor_total"] = anchor_total.item()
                metrics[f"{train_eval}_loss_anchor_weighted"] = weighted_anchor_total.item()

        if vcdist_loss is not None:
            weighted_vcdist_loss = self.args.vcdist_weight * vcdist_loss
            single_total = weighted_vcdist_loss if single_total is None else single_total + weighted_vcdist_loss
            metrics[f"{train_eval}_loss_vcdist_weighted"] = weighted_vcdist_loss.item()

        branch_totals = [loss for loss in (single_total, multi_total) if loss is not None]
        if branch_totals:
            final_loss = torch.stack(branch_totals).mean()
        else:
            final_loss = torch.tensor(0.0, device=model.device, requires_grad=True)

        metrics[f"{train_eval}_loss_combined"] = final_loss.item()
        
        return final_loss, self.filter_core_log_metrics(metrics, train_eval)

    def forward_with_pre_concatenated_batch(
        self, model: nn.Module, concatenated_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        A modified version of `concatenated_forward` that skips `concatenated_inputs`.
        It expects `concatenated_batch` to be already constructed with shape (2*Batch, ...).
        Now supports 'completion_token_mask' for token-level preference loss.
        """
        # 1. 准备模型参数
        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # 透传视觉参数
        for k in ["pixel_values", "pixel_attention_mask", "image_sizes", "image_grid_thw"]:
            if k in concatenated_batch:
                model_kwargs[k] = concatenated_batch[k]
        
        if "token_type_ids" in concatenated_batch:
            model_kwargs["token_type_ids"] = concatenated_batch["token_type_ids"]

        # 2. 提取并拼接 Input IDs (Prompt + Completion)
        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        # 拼接 Prompt 和 Completion (Left + Right)
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        
        # 3. 构建 Loss Mask (基础: 只计算 Completion 部分的 Loss)
        # Prompt 部分全为 0，Completion 部分为 1
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
            dim=1,
        )

        # 如果 batch 中提供了 completion 级 token mask，我们需要将其对齐到 full sequence
        token_level_mask = None
        if "completion_token_mask" in concatenated_batch:
            comp_token_mask = concatenated_batch["completion_token_mask"]
            # 同样拼接：Prompt 部分不计算 loss (全0) + Completion 部分使用自定义 mask
            token_level_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), comp_token_mask),
                dim=1
            )

        if self.max_length is not None and self.max_length < attention_mask.size(1):
            input_ids = input_ids[:, -self.max_length :]
            attention_mask = attention_mask[:, -self.max_length :]
            loss_mask = loss_mask[:, -self.max_length :]
            
            if token_level_mask is not None:
                token_level_mask = token_level_mask[:, -self.max_length :]
            
            if "token_type_ids" in model_kwargs:
                model_kwargs["token_type_ids"] = model_kwargs["token_type_ids"][:, -self.max_length :]

        # 5. 模型前向传播
        model_kwargs["attention_mask"] = attention_mask
        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        # 6. 计算 Log Probability
        # Shift logits and labels
        labels = input_ids.clone()
        
        # Shift: logits[t] predicts labels[t+1]
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        loss_mask = loss_mask[:, 1:]
        
        # 如果有 token_level_mask，也要同步 Shift 并合并到 loss_mask
        if token_level_mask is not None:
            token_level_mask = token_level_mask[:, 1:]
            loss_mask = loss_mask * token_level_mask

        # -------------------------------------------------
        # 防止 gather 越界：先把不计算 loss 的位置设为 0 (或其他有效 index)
        labels[~loss_mask.bool()] = 0 
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        # 再次 Mask：把无效位置的 logps 抹零
        # 此时 loss_mask 已经结合了 attention_mask 和 completion_token_mask
        per_token_logps[~loss_mask.bool()] = 0
        # -------------------------------------------------

        # Sum per sequence
        all_logps = per_token_logps.sum(-1)

        # 7. 构建输出
        batch_size = input_ids.shape[0] // 2
        return {
            "chosen_logps": all_logps[:batch_size],
            "rejected_logps": all_logps[batch_size:],
            "logits": logits, 
        }
    
    

    def run_single_image_preference(
        self, 
        model, 
        batch, 
        metrics, 
        train_eval
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
        """
        Executes Image 1 and Image 2 tasks in parallel using `forward_with_pre_concatenated_batch`.
        Supports:
        1. Unified DataCollator (Auto-Padding).
        2. Token-Level Masking (completion_token_mask).
        """
        # 基础校验
        if "image_1_prompt_input_ids" not in batch:
            return None, None, None

        pad_token_id = self.processing_class.tokenizer.pad_token_id or 0
        device = model.device

        # =================================================================
        # 1. 准备基础文本数据 (Lists)
        # =================================================================
        prompts = [batch["image_1_prompt_input_ids"], batch["image_2_prompt_input_ids"]]
        prompt_masks = [batch["image_1_prompt_attention_mask"], batch["image_2_prompt_attention_mask"]]
        chosens = [batch["resp_1_input_ids"], batch["resp_2_input_ids"]]
        chosen_masks = [batch["resp_1_attention_mask"], batch["resp_2_attention_mask"]]
        rejecteds = [batch["resp_2_input_ids"], batch["resp_1_input_ids"]]
        rejected_masks = [batch["resp_2_attention_mask"], batch["resp_1_attention_mask"]]

        # Token Type IDs
        token_type_ids_list = []
        if "image_1_prompt_token_type_ids" in batch:
            tt1 = batch["image_1_prompt_token_type_ids"]
            tt2 = batch["image_2_prompt_token_type_ids"]
            token_type_ids_list = [tt1, tt2]

        # =================================================================
        # 2. 处理 Completion Token Mask
        # =================================================================
        final_token_mask = None
        if self.args.use_single_branch_token_mask:
            final_token_mask = build_single_image_completion_token_mask(
                batch,
                pad_and_cat=self.pad_and_cat,
                device=device,
            )

        # =================================================================
        # 3. 处理视觉特征 (Pixel Values)
        # =================================================================
        p1 = batch["image_1_pixel_values"]
        p2 = batch["image_2_pixel_values"]

        # 1. 展平 Pixel Values
        def flatten_vision_input(tensor):
            if tensor.dim() == 6: return tensor.flatten(0, 2)
            elif tensor.dim() == 5: return tensor.flatten(0, 1)
            return tensor
        
        p1 = flatten_vision_input(p1)
        p2 = flatten_vision_input(p2)

        combined_pv = torch.cat([p1, p2], dim=0)
        final_pixel_values = torch.cat([combined_pv, combined_pv], dim=0)

        if 'image_1_pixel_attention_mask' in batch.keys():
            # 2. 准备 Mask
            m1 = batch["image_1_pixel_attention_mask"]
            m2 = batch["image_2_pixel_attention_mask"]

            def flatten_mask_input(tensor):
                if tensor.dim() == 5: return tensor.flatten(0, 2)
                elif tensor.dim() == 4: return tensor.flatten(0, 1)
                return tensor

            m1 = flatten_mask_input(m1)
            m2 = flatten_mask_input(m2)

            combined_pm = torch.cat([m1, m2], dim=0)
            final_pixel_mask = torch.cat([combined_pm, combined_pm], dim=0)

            # 3. 执行 Unpadding (过滤)
            # 计算有效性: 只要 HxW 平面上有一个点是 1，该 Tile 就有效
            valid_tile_mask = final_pixel_mask.sum(dim=(1, 2)) > 0
            
            # 维度对齐检查
            if final_pixel_values.shape[0] != valid_tile_mask.shape[0]:
                raise RuntimeError(f"Shape Mismatch! PV={final_pixel_values.shape[0]}, Mask={valid_tile_mask.shape[0]}")

            # 应用过滤
            # [Visual Guide]
            # Before: [Tile1(Valid), Tile2(Valid), Tile3(Pad), Tile4(Pad)] -> Shape 4
            # Mask:   [True,         True,         False,      False      ]
            # After:  [Tile1,        Tile2]                                 -> Shape 2
            
            final_pixel_values = final_pixel_values[valid_tile_mask]




        # E. 处理 Image Sizes
        final_image_sizes = None
        if "image_1_image_sizes" in batch:
            s1 = batch["image_1_image_sizes"]
            s2 = batch["image_2_image_sizes"]
            combined_sizes = torch.cat([s1, s2], dim=0)
            if combined_sizes.shape[1] == 1: combined_sizes = combined_sizes.squeeze(1)
            final_image_sizes = torch.cat([combined_sizes, combined_sizes], dim=0)

        # =================================================================
        # 4. 构造 Concatenated Batch (4*B)
        # =================================================================
        # 拼接顺序: [Chosen, Rejected]
        all_prompts = prompts + prompts
        all_prompt_masks = prompt_masks + prompt_masks
        
        all_resps = chosens + rejecteds
        all_resp_masks = chosen_masks + rejected_masks

        concatenated_batch = {
            "prompt_input_ids": self.pad_and_cat(all_prompts, padding_side="left", padding_value=pad_token_id, device=device),
            "prompt_attention_mask": self.pad_and_cat(all_prompt_masks, padding_side="left", padding_value=0, device=device),
            
            "completion_input_ids": self.pad_and_cat(all_resps, padding_side="right", padding_value=pad_token_id, device=device),
            "completion_attention_mask": self.pad_and_cat(all_resp_masks, padding_side="right", padding_value=0, device=device),
            
            "pixel_values": final_pixel_values,
        }

        # 注入可选参数
        if final_token_mask is not None:
            concatenated_batch["completion_token_mask"] = final_token_mask
        # if final_pixel_mask is not None:
        #     concatenated_batch["pixel_attention_mask"] = final_pixel_mask
        if final_image_sizes is not None:
            concatenated_batch["image_sizes"] = final_image_sizes
        
        if token_type_ids_list:
            # [TT1, TT2, TT1, TT2]
            all_tt = token_type_ids_list + token_type_ids_list
            concatenated_batch["token_type_ids"] = self.pad_and_cat(all_tt, padding_side="left", padding_value=pad_token_id, device=device)


        # =================================================================
        # 5. 前向传播 (调用新写的 forward)
        # =================================================================
        # 注意：这里不需要再传 model_kwargs，因为 forward_with_pre_concatenated_batch 会自己处理
        policy_output = self.forward_with_pre_concatenated_batch(model, concatenated_batch)

        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_output = self.forward_with_pre_concatenated_batch(model, concatenated_batch)
            else:
                ref_output = self.forward_with_pre_concatenated_batch(self.ref_model, concatenated_batch)

        # =================================================================
        # 6. Loss 计算
        # =================================================================
        all_losses, all_chosen_rewards, all_rejected_rewards = self.dpo_loss(
            policy_output["chosen_logps"], policy_output["rejected_logps"],
            ref_output["chosen_logps"], ref_output["rejected_logps"]
        )


        self.log_metrics(
            metrics, train_eval, "single", all_losses, all_chosen_rewards, all_rejected_rewards
        )

        base_loss = all_losses.mean()
        aux_loss = base_loss.new_zeros(())
        anchor_loss = base_loss.new_zeros(())
        total_loss = base_loss

        if self.args.add_anchor_loss:
            anchor_losses = self.anchor_loss(
                policy_output["chosen_logps"], ref_output["chosen_logps"], self.args.anchor_delta
            )
            anchor_loss = anchor_losses.mean()
            metrics[f"{train_eval}_single_anchor_loss"] = anchor_loss.item()
            total_loss += self.args.anchor_weight * anchor_loss

        logps_pack = {
            "chosen": policy_output["chosen_logps"],
            "rejected": policy_output["rejected_logps"],
            "ref_chosen": ref_output["chosen_logps"],
            "ref_rejected": ref_output["rejected_logps"]
        }

        loss_components = {
            "base_loss": base_loss,
            "aux_loss": aux_loss,
            "anchor_loss": anchor_loss,
        }

        return total_loss, logps_pack, loss_components



    def run_multi_image_preference(
        self, 
        model, 
        batch, 
        metrics, 
        train_eval
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
        """
        Executes Multi 1 and Multi 2 tasks in PARALLEL.
        FIXED: 
        1. Correctly flattens 6D pixel_values (OneVision).
        2. Manually concatenates batch to prevent TRL from dropping image_sizes.
        3. Passes visual masks and metadata through the pre-concatenated forward path.
        """
        if "multi_image_1_prompt_input_ids" not in batch:
            return None, None, None

        assert "multi_image_2_prompt_input_ids" in batch, "Multi Image Prompt 2 missing"
        
        pad_token_id = self.processing_class.tokenizer.pad_token_id or 0

        # =================================================================
        # 1. 准备基础数据 (List)
        # =================================================================
        chosen_prompts = [batch["multi_image_1_prompt_input_ids"], batch["multi_image_2_prompt_input_ids"]]
        chosen_prompt_masks = [batch["multi_image_1_prompt_attention_mask"], batch["multi_image_2_prompt_attention_mask"]]
        chosen_resps = [batch["resp_1_input_ids"], batch["resp_2_input_ids"]]
        chosen_resp_masks = [batch["resp_1_attention_mask"], batch["resp_2_attention_mask"]]
        rejected_prompts = [batch["multi_image_1_prompt_input_ids"], batch["multi_image_2_prompt_input_ids"]]
        rejected_prompt_masks = [batch["multi_image_1_prompt_attention_mask"], batch["multi_image_2_prompt_attention_mask"]]
        rejected_resps = [batch["resp_2_input_ids"], batch["resp_1_input_ids"]]
        rejected_resp_masks = [batch["resp_2_attention_mask"], batch["resp_1_attention_mask"]]

        # =================================================================
        # 2. 准备 & 修正视觉特征 (Pixel Values)
        # =================================================================
        pixel_values_list = [batch["multi_image_pixel_values"], batch["multi_image_pixel_values"]]
        raw_pv = torch.cat(pixel_values_list, dim=0) # Shape: (2B, Num_Images, ...)
        
        # 同时支持 6D (OneVision) 和 5D (Standard) 的 Flatten
        if raw_pv.dim() == 6:
            # (2B, N, Patches, C, H, W) -> (2B*N, Patches, C, H, W)
            final_pv = raw_pv.flatten(0, 2)
        elif raw_pv.dim() == 5:
            # (2B, N, C, H, W) -> (2B*N, C, H, W)
            final_pv = raw_pv.flatten(0, 1)
        else:
            final_pv = raw_pv


        # =================================================================
        # 3. 准备视觉元数据 (Sizes, Grids)
        # =================================================================
        # Image Sizes
        final_sizes = None
        if "multi_image_image_sizes" in batch:
            sizes_list = [batch["multi_image_image_sizes"], batch["multi_image_image_sizes"]]
            raw_sizes = torch.cat(sizes_list, dim=0) # (2B, N, 2)
            
            # 如果 pv 被展平了，sizes 也要展平
            if raw_pv.dim() >= 5 and raw_sizes.dim() == 3:
                 final_sizes = raw_sizes.flatten(0, 1)
            else:
                 final_sizes = raw_sizes

        # Image Grids (Qwen)
        # final_grids = None
        # if "multi_image_grid_thw" in batch:
        #     final_grids = torch.cat([batch["multi_image_grid_thw"], batch["multi_image_grid_thw"]], dim=0)
        final_grids = None
        if "multi_image_grid_thw" in batch:
            # Grids 也是对应 Image 级别
            raw_grids = torch.cat([batch["multi_image_grid_thw"], batch["multi_image_grid_thw"]], dim=0)
            if raw_grids.dim() == 3:
                # [Fix] 之前缺少 Flatten，导致是 (2B, N, 3)，模型期望 (Total_Images, 3)
                final_grids = raw_grids.flatten(0, 1) 
            else:
                final_grids = raw_grids

        # Pixel Masks (Optional)
        # final_pixel_mask = None
        # if "multi_image_pixel_attention_mask" in batch:
        #      mask_list = [batch["multi_image_pixel_attention_mask"], batch["multi_image_pixel_attention_mask"]]
        #      raw_pm = torch.cat(mask_list, dim=0)
        #      if raw_pv.dim() >= 5 and raw_pm.dim() > 1:
        #          final_pixel_mask = raw_pm.flatten(0, 1)
        #      else:
        #          final_pixel_mask = raw_pm
    
        final_pixel_mask = None
        if "multi_image_pixel_attention_mask" in batch:
             mask_list = [batch["multi_image_pixel_attention_mask"], batch["multi_image_pixel_attention_mask"]]
             raw_pm = torch.cat(mask_list, dim=0)
             
             # [Fix] Mask 的展平逻辑必须严格跟随 Pixel Values
             if raw_pv.dim() == 6:
                 # PV 是 6D (B, N, T, C, H, W) -> flatten(0, 2)
                 # Mask 通常是 5D (B, N, T, H, W) -> 必须也是 flatten(0, 2)
                 if raw_pm.dim() == 5:
                     final_pixel_mask = raw_pm.flatten(0, 2)
                 else:
                     # 防御性：如果 Mask 维度定义不一致，尝试回退
                     final_pixel_mask = raw_pm.flatten(0, 1)
             elif raw_pv.dim() == 5 and raw_pm.dim() > 1:
                 # PV 是 5D -> flatten(0, 1)
                 final_pixel_mask = raw_pm.flatten(0, 1)
             else:
                 final_pixel_mask = raw_pm

        # =================================================================
        # 4. 手动构造 Concatenated Batch
        # =================================================================
        
        # A. 拼接文本:
        # default:
        all_prompts = chosen_prompts + rejected_prompts
        all_prompt_masks = chosen_prompt_masks + rejected_prompt_masks
        all_resps = chosen_resps + rejected_resps
        all_resp_masks = chosen_resp_masks + rejected_resp_masks
        
        prompt_ids = self.pad_and_cat(all_prompts, padding_side="left", padding_value=pad_token_id, device=model.device)
        prompt_mask = self.pad_and_cat(all_prompt_masks, padding_side="left", padding_value=0, device=model.device)
        resp_ids = self.pad_and_cat(all_resps, padding_side="right", padding_value=pad_token_id, device=model.device)
        resp_mask = self.pad_and_cat(all_resp_masks, padding_side="right", padding_value=0, device=model.device)
        
        # B. 拼接视觉: [Chosen_PV, Rejected_PV]
        # final_pv 已经是 2B (Task1+Task2)，再拼一次变成 4B
        concatenated_pv = torch.cat([final_pv, final_pv], dim=0)
        
        concatenated_sizes = None
        if final_sizes is not None:
            concatenated_sizes = torch.cat([final_sizes, final_sizes], dim=0)
            
        concatenated_grids = None
        if final_grids is not None:
            concatenated_grids = torch.cat([final_grids, final_grids], dim=0)

        concatenated_pm = None
        if final_pixel_mask is not None:
            concatenated_pm = torch.cat([final_pixel_mask, final_pixel_mask], dim=0)

        # D. 组装 Batch
        concatenated_batch = {
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "completion_input_ids": resp_ids,
            "completion_attention_mask": resp_mask,
            "pixel_values": concatenated_pv,
        }

        # 显式填入 key
        if concatenated_sizes is not None:
            concatenated_batch["image_sizes"] = concatenated_sizes
        if concatenated_grids is not None:
            concatenated_batch["image_grid_thw"] = concatenated_grids
        if concatenated_pm is not None:
            concatenated_batch["pixel_attention_mask"] = concatenated_pm
            
        # Token Types
        if "multi_image_token_type_ids" in batch:
            tt = batch["multi_image_token_type_ids"]
            tt2 = batch["multi_image_2_token_type_ids"]
            chosen_token_types = [tt, tt2]
            rejected_token_types = [tt, tt2]
            all_tt = chosen_token_types + rejected_token_types
            concatenated_batch["token_type_ids"] = self.pad_and_cat(all_tt, padding_side="left", padding_value=pad_token_id, device=model.device)

        policy_output = self.forward_with_pre_concatenated_batch(model, concatenated_batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_output = self.forward_with_pre_concatenated_batch(model, concatenated_batch)
            else:
                ref_output = self.forward_with_pre_concatenated_batch(self.ref_model, concatenated_batch)

        # =================================================================
        # 6. Loss 计算 & 日志
        # =================================================================
        all_losses, all_c_rewards, all_r_rewards = self.dpo_loss(
            policy_output["chosen_logps"], policy_output["rejected_logps"],
            ref_output["chosen_logps"], ref_output["rejected_logps"]
        )


        self.log_metrics(metrics, train_eval, "multi", all_losses, all_c_rewards, all_r_rewards)
        
        base_loss = all_losses.mean()
        aux_loss = base_loss.new_zeros(())
        anchor_loss = base_loss.new_zeros(())
        total_loss = base_loss

        if self.args.add_anchor_loss:
            anchor_losses = self.anchor_loss(
                policy_output["chosen_logps"], ref_output["chosen_logps"], self.args.anchor_delta
            )
            anchor_loss = anchor_losses.mean()
            metrics[f"{train_eval}_multi_anchor_loss"] = anchor_loss.item()
            total_loss += self.args.anchor_weight * anchor_loss

        logps_pack = {
            "chosen": policy_output["chosen_logps"],
            "rejected": policy_output["rejected_logps"],
            "ref_chosen": ref_output["chosen_logps"],
            "ref_rejected": ref_output["rejected_logps"]
        }

        loss_components = {
            "base_loss": base_loss,
            "aux_loss": aux_loss,
            "anchor_loss": anchor_loss,
        }

        return total_loss, logps_pack, loss_components
    def soft_label_preference_loss(
        self,
        student_chosen_logps: torch.Tensor,
        student_rejected_logps: torch.Tensor,
        student_ref_chosen_logps: torch.Tensor,
        student_ref_rejected_logps: torch.Tensor,
        teacher_chosen_logps: torch.Tensor,
        teacher_rejected_logps: torch.Tensor,
        teacher_ref_chosen_logps: torch.Tensor,
        teacher_ref_rejected_logps: torch.Tensor,
        beta: float = 0.1,
        temperature: float = 1.0,
        validation_threshold: float = 0.5, # 新增参数：过滤阈值
    ) -> tuple[torch.Tensor, dict]:
        """
        Visual Preference Distillation with Teacher Correctness Filtering.
        Only distills when Teacher agrees with the Ground Truth label (prob > threshold).
        """
        # 1. 计算 Student Margin
        student_logits = (student_chosen_logps - student_rejected_logps) - (student_ref_chosen_logps - student_ref_rejected_logps)
        student_margin = beta * student_logits

        # 2. 计算 Teacher Margin & Soft Label
        if self.args.vcdist_stopgrad:
            with torch.no_grad():
                teacher_logits = (teacher_chosen_logps - teacher_rejected_logps) - (teacher_ref_chosen_logps - teacher_ref_rejected_logps)
                scaled_teacher_logits = teacher_logits / temperature
                teacher_margin = beta * scaled_teacher_logits
                
                target_probs = torch.sigmoid(teacher_margin)
        else:
            teacher_logits = (teacher_chosen_logps - teacher_rejected_logps) - (teacher_ref_chosen_logps - teacher_ref_rejected_logps)
            scaled_teacher_logits = teacher_logits / temperature
            teacher_margin = beta * scaled_teacher_logits
            
            target_probs = torch.sigmoid(teacher_margin)

        # 3. 计算 Student 概率
        student_probs = torch.sigmoid(student_margin)

        # 4. 计算原始 BCE Loss (不进行 reduction，保持每个样本独立)
        raw_loss = F.binary_cross_entropy(student_probs, target_probs, reduction='none')

        teacher_acc_mask = (target_probs > validation_threshold).float()
        correctness_mask = (target_probs > validation_threshold).float()
        confidence_mask = (student_probs < target_probs).float()

        if self.args.vcdist_filter:
            active_mask = confidence_mask * correctness_mask
        else:
            active_mask = torch.ones_like(target_probs)

        filtered_loss = raw_loss * active_mask

        # 5. Loss Reduction (计算平均值)
        # 注意：分母应该是 valid_mask.sum() 还是 batch_size？
        # 推荐使用 valid_mask.sum()，这样 loss 的量级不会因为有效样本变少而由于被稀释变小
        num_valid = active_mask.sum()
        
        # 防止分母为 0
        if num_valid > 0:
            final_loss = filtered_loss.sum() / num_valid
        else:
            final_loss = torch.tensor(0.0, device=raw_loss.device, requires_grad=True)

        # # 6. Metrics 监控
        # kl_div = (target_probs * (torch.log(target_probs + 1e-6) - torch.log(student_probs + 1e-6)) + 
        #           (1 - target_probs) * (torch.log(1 - target_probs + 1e-6) - torch.log(1 - student_probs + 1e-6)))
        
        # # 只统计有效样本的 KL
        # valid_kl = (kl_div * valid_mask).sum() / (num_valid + 1e-8)

        # return final_loss, {
        #     "vcdist_kl": valid_kl.item(),
        #     "teacher_confidence": (target_probs - 0.5).abs().mean().item(),
        #     "teacher_acc": (target_probs > validation_threshold).float().mean().item(), # 监控 Teacher 的准确率
        #     "vcdist_valid_count": num_valid.item() # 监控有多少样本参与了蒸馏
        # }

        # 公式: P * log(P/Q) + (1-P) * log((1-P)/(1-Q))
        kl_div_element_wise = (target_probs * (torch.log(target_probs + 1e-6) - torch.log(student_probs + 1e-6)) + 
                               (1 - target_probs) * (torch.log(1 - target_probs + 1e-6) - torch.log(1 - student_probs + 1e-6)))
        
        # === A. 计算 Valid KL (原有逻辑) ===
        valid_kl_sum = (kl_div_element_wise * correctness_mask).sum()
        valid_count = correctness_mask.sum()

        active_kl_sum = (kl_div_element_wise * active_mask).sum()
        active_count = active_mask.sum()
        
        # === B. [新增] 计算 Global KL (所有样本) ===
        # 直接求和，不乘 mask
        global_kl_sum = kl_div_element_wise.sum()
        global_count = torch.tensor(kl_div_element_wise.numel(), device=kl_div_element_wise.device, dtype=torch.float)

        teacher_prob_sum = target_probs.sum()
        student_prob_sum = student_probs.sum()
        teacher_confidence_sum = (target_probs - 0.5).abs().sum()
        teacher_acc_count = teacher_acc_mask.sum()
        confident_wrong_count = (target_probs < 0.1).float().sum()
        ambiguous_count = ((target_probs > validation_threshold) & (target_probs <= validation_threshold + 0.1)).float().sum()
        gate_pass_count = correctness_mask.sum()
        trigger_count = (confidence_mask * correctness_mask).sum()

        metrics_payload = {
            "vcdist_valid_kl_sum": valid_kl_sum.detach(),
            "vcdist_valid_count": valid_count.detach(),
            "vcdist_active_kl_sum": active_kl_sum.detach(),
            "vcdist_active_count": active_count.detach(),
            "vcdist_global_kl_sum": global_kl_sum.detach(),
            "vcdist_global_count": global_count.detach(),
            "vcdist_teacher_prob_sum": teacher_prob_sum.detach(),
            "vcdist_student_prob_sum": student_prob_sum.detach(),
            "vcdist_teacher_confidence_sum": teacher_confidence_sum.detach(),
            "vcdist_teacher_acc_count": teacher_acc_count.detach(),
            "vcdist_confident_wrong_count": confident_wrong_count.detach(),
            "vcdist_ambiguous_count": ambiguous_count.detach(),
            "vcdist_gate_pass_count": gate_pass_count.detach(),
            "vcdist_trigger_count": trigger_count.detach(),
        }

        if target_probs.shape[0] % 2 == 0 and target_probs.shape[0] > 0:
            split_index = target_probs.shape[0] // 2
            for pos_name, pos_slice in [
                ("pos1", slice(0, split_index)),
                ("pos2", slice(split_index, None)),
            ]:
                pos_target_probs = target_probs[pos_slice]
                pos_teacher_acc_mask = teacher_acc_mask[pos_slice]
                pos_correctness_mask = correctness_mask[pos_slice]
                pos_trigger_mask = (confidence_mask * correctness_mask)[pos_slice]

                metrics_payload[f"vcdist_{pos_name}_teacher_prob_sum"] = pos_target_probs.sum().detach()
                metrics_payload[f"vcdist_{pos_name}_teacher_acc_count"] = pos_teacher_acc_mask.sum().detach()
                metrics_payload[f"vcdist_{pos_name}_gate_pass_count"] = pos_correctness_mask.sum().detach()
                metrics_payload[f"vcdist_{pos_name}_trigger_count"] = pos_trigger_mask.sum().detach()
                metrics_payload[f"vcdist_{pos_name}_global_count"] = torch.tensor(
                    pos_target_probs.numel(),
                    device=target_probs.device,
                    dtype=torch.float,
                )

        return final_loss, metrics_payload

    def anchor_loss(
        self,
        chosen_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        anchor_delta: float = 0.0,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the IC-VCO anchor regularization loss.
        
        Formula: L = -log(sigmoid(beta * log(pi/pi_ref) - delta))
        
        Args:
            The arguments match the underlying pairwise preference loss inputs.
        """
        device = self.accelerator.device

        # 1. 计算 Chosen 的 Log Ratios (log(pi) - log(pi_ref))
        # Anchor Loss 只关注 chosen response，因此我们主要计算这一部分
        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)

        # 3. 计算 Logits
        # 公式核心部分: beta * log(pi/pi_ref) - delta
        # 对应图片中的: beta * log(pi_theta / pi_ref) - delta
        logits = self.beta * chosen_logratios - anchor_delta

        # 4. 计算 Loss
        # 公式: -log(sigmoid(logits))
        # 使用 F.logsigmoid 计算更数值稳定: -F.logsigmoid(logits)
        losses = -F.logsigmoid(logits)

        return losses


    # -------------------------------------------------------------------------
    # 4. 辅助函数
    # -------------------------------------------------------------------------
    @staticmethod
    def pad_and_cat(tensors, padding_side="right", padding_value=0, device=None):
        """
        Helper: Pad list of tensors to max length and concatenate along dim 0.
        """
        if not tensors:
            return torch.tensor([], device=device)
        
        max_len = max(t.size(1) for t in tensors)
        padded_tensors = []
        for t in tensors:
            pad_len = max_len - t.size(1)
            if pad_len > 0:
                if padding_side == "right":
                    t = torch.nn.functional.pad(t, (0, pad_len), value=padding_value)
                elif padding_side == "left":
                    t = torch.nn.functional.pad(t, (pad_len, 0), value=padding_value)
            padded_tensors.append(t)
        
        return torch.cat(padded_tensors, dim=0)

    def log_metrics(self, metrics, prefix, task, losses, chosen_rewards, rejected_rewards):
        """
        Helper to log metrics with DDP support.
        Gathers tensors from all GPUs before computing the mean.
        """
        # 3. 定义一个内部辅助函数来处理 gather + mean
        def gather_and_mean(tensor):
            # (A) Detach: 切断梯度，防止显存泄漏 (关键!)
            tensor = tensor.detach()
            
            # (B) Gather: 从所有 GPU 收集数据
            # gather_for_metrics 会自动处理 DDP 中的 padding 问题
            all_tensors = self.accelerator.gather_for_metrics(tensor)
            
            # (C) Mean: 计算全局均值
            return all_tensors.mean().item()

         # 2. 构造全名前缀
        full_prefix = f"{prefix}_{task}_"
        # 1. 计算 Accuracy 和 Margin (Local)
        # 先在本地计算，保持维度一致，方便 gather
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        margins = chosen_rewards - rejected_rewards
        # 4. 记录日志
        metrics[f"{full_prefix}rewards/chosen"] = gather_and_mean(chosen_rewards)
        metrics[f"{full_prefix}rewards/rejected"] = gather_and_mean(rejected_rewards)
        metrics[f"{full_prefix}rewards/accuracies"] = gather_and_mean(reward_accuracies)
        metrics[f"{full_prefix}rewards/margins"] = gather_and_mean(margins)
        metrics[f"{full_prefix}loss"] = gather_and_mean(losses)

        if chosen_rewards.numel() % 2 == 0 and chosen_rewards.numel() > 0:
            split_index = chosen_rewards.shape[0] // 2
            for pos_name, pos_slice in [
                ("pos1", slice(0, split_index)),
                ("pos2", slice(split_index, None)),
            ]:
                pos_chosen_rewards = chosen_rewards[pos_slice]
                pos_rejected_rewards = rejected_rewards[pos_slice]
                pos_reward_accuracies = (pos_chosen_rewards > pos_rejected_rewards).float()
                pos_margins = pos_chosen_rewards - pos_rejected_rewards

                metrics[f"{full_prefix}{pos_name}_rewards/accuracies"] = gather_and_mean(pos_reward_accuracies)
                metrics[f"{full_prefix}{pos_name}_rewards/margins"] = gather_and_mean(pos_margins)
