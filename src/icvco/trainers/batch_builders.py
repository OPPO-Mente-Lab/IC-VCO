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

from typing import Callable, Mapping

import torch


PadAndCatFn = Callable[..., torch.Tensor]


def build_completion_token_mask(
    token_masks: list[torch.Tensor],
    *,
    pad_and_cat: PadAndCatFn,
    device,
) -> torch.Tensor | None:
    if not token_masks:
        return None
    return pad_and_cat(token_masks, padding_side="right", padding_value=0, device=device)


def build_single_image_completion_token_mask(
    batch: Mapping[str, torch.Tensor],
    *,
    pad_and_cat: PadAndCatFn,
    device,
) -> torch.Tensor | None:
    if "resp_1_token_mask" not in batch or "resp_2_token_mask" not in batch:
        return None

    token_masks = [
        batch["resp_1_token_mask"],
        batch["resp_2_token_mask"],
        batch["resp_2_token_mask"],
        batch["resp_1_token_mask"],
    ]

    return build_completion_token_mask(token_masks, pad_and_cat=pad_and_cat, device=device)
