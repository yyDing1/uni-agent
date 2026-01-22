# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from .r2e_gym import R2EGymTestSpec
from .swe_bench import SWEBenchTestSpec

MAP_DATA_TO_TEST_SPEC = {
    "swe-bench-verified": SWEBenchTestSpec,
    "r2e-gym-subset": R2EGymTestSpec,
}
SWETestSpec = SWEBenchTestSpec


def make_test_spec(dataset_id: str, metadata: dict) -> SWETestSpec:
    assert dataset_id in MAP_DATA_TO_TEST_SPEC, f"Unknown dataset_id {dataset_id}"
    return MAP_DATA_TO_TEST_SPEC[dataset_id](metadata=metadata)
