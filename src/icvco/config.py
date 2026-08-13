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

from .dataset_registry import load_dataset_registry


MODULE_KEYWORDS = {
    "llava-interleave": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"],
    },
    "llava-onevision": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"],
        "max_grid_side": 768,
    },
}


# Keep the legacy import surface stable while moving the actual dataset registry
# into configs/datasets/registry.json.
DATASET_KEYWORDS = load_dataset_registry()
