# Copyright 2022 The T5X Authors.
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

"""Import API modules."""

import lit.t5x.adafactor
import lit.t5x.checkpoints
import lit.t5x.decoding
import lit.t5x.gin_utils
import lit.t5x.losses
import lit.t5x.models
import lit.t5x.partitioning
import lit.t5x.state_utils
import lit.t5x.train_state
import lit.t5x.trainer
import lit.t5x.utils

# Version number.
from t5x.version import __version__

# TODO(adarob): Move clients to t5x.checkpointing and rename
# checkpoints.py to checkpointing.py
checkpointing = t5x.checkpoints
