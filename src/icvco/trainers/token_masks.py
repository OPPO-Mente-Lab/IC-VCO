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

from typing import Sequence


def prepare_token_mask(
    input_ids: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
    target_spans: Sequence[tuple[int, int]] | None,
) -> list[int]:
    seq_len = len(input_ids)
    mask = [0] * seq_len

    if not target_spans:
        return mask

    for idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_start == token_end:
            continue

        for span_start, span_end in target_spans:
            overlap_start = max(token_start, span_start)
            overlap_end = min(token_end, span_end)
            if overlap_start < overlap_end:
                mask[idx] = 1
                break

    return mask
