# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import os
from easydict import EasyDict as edict

"""System Paths Configuration
"""

__C = edict()
# All users should use cfg to get config options
cfg = __C

##########################################################################
# Dataset
##########################################################################
__C.DATASET = edict()
__C.DATASET.BASE_PATH = os.environ['DATA_DIR']
__C.DATASET.PATHS = edict()
__C.DATASET.PATHS.EXAMPLE_DATA = os.path.join(__C.DATASET.BASE_PATH, 'example_data')
__C.DATASET.PATHS.INFERENCE = os.path.join(__C.DATASET.BASE_PATH, 'inferences')
##########################################################################
# Manifest
##########################################################################
__C.MANIFEST = edict()
__C.MANIFEST.PATH = os.path.join(__C.DATASET.BASE_PATH, 'manifests')
__C.MANIFEST.FILE = 'manifest.json'
##########################################################################
# NeMo
##########################################################################
__C.NEMO = edict()
__C.NEMO.BASE_PATH = '/tmp/NeMo/'
__C.NEMO.TOOLS = os.path.join(os.environ['APP_DIR'], 'tools', 'NeMo')
__C.NEMO.PRETRAINED = os.path.join(os.environ['APP_DIR'], 'demo', 'quartznet15x5')
__C.NEMO.CONFIGS = os.path.join(__C.NEMO.BASE_PATH, 'examples', 'asr', 'configs')
##########################################################################
# Models
##########################################################################
__C.MODEL = edict()
##########################################################################
# Acoustic Model
##########################################################################
__C.MODEL.AM = edict()
__C.MODEL.AM.NAME = 'acoustic'
__C.MODEL.AM.CONFIG_FILES = os.path.join(__C.DATASET.BASE_PATH, 'config_files')
__C.MODEL.AM.TRAIN_SCRIPT = os.path.join(__C.NEMO.TOOLS, 'jasper_train.py')
__C.MODEL.AM.PATH = os.path.join(__C.DATASET.BASE_PATH, 'models', 'acoustic_models')
##########################################################################
# Language Model
##########################################################################
__C.MODEL.LM = edict()
__C.MODEL.LM.PATH = os.path.join(__C.DATASET.BASE_PATH, 'models', 'language_models')
__C.MODEL.LM.DECODERS = os.path.join(__C.NEMO.BASE_PATH, 'scripts','decoders')
__C.MODEL.LM.TRAIN_SCRIPT = os.path.join(__C.NEMO.TOOLS, 'build_lm.py')
__C.MODEL.LM.PRETRAINED = os.path.join(__C.DATASET.PATHS.EXAMPLE_DATA, 'language_model', '6-gram-lm.binary')
##########################################################################