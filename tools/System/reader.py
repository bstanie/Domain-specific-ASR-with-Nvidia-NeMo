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
import shutil
import pandas as pd
import subprocess
from shutil import copyfile
from easydict import EasyDict as edict

from tools.filetools import *
from tools.System.config import cfg
from tools.System.common_reader import CommonReader

import logging


class Reader(CommonReader):
  """
  A control system for Data and Model versioning through manifest
  """

  @staticmethod
  def new(project_id):
    """Create a new reader

    Arguments:
      project_id - unique project description
    """
    manifest = edict()
    manifest.is_built = False
    manifest.id = project_id
    manifest.manifest_path = os.path.join(cfg.MANIFEST.PATH, str(project_id) + '_' + cfg.MANIFEST.FILE)

    # eval datasets
    manifest.eval_dataset_names = []
    manifest.eval_datasets = []

    # Inference
    manifest.inference = edict()
    manifest.inference_params = edict()
    manifest.inference_params.load_dir = os.path.join(cfg.NEMO.PRETRAINED)
    manifest.inference_params.model_config = None
    manifest.inference_params.batch_size = None
    manifest.inference_params.amp_opt_level = None
    manifest.inference_params.save_results = os.path.join(cfg.DATASET.PATHS.INFERENCE, str(project_id))
    manifest.inference_params.lm_path = None
    manifest.inference_params.beam_width = None
    manifest.inference_params.alpha = None
    manifest.inference_params.beta = None

    # Acoustic model
    manifest.am = edict()
    manifest.am.train_dataset_names = []
    manifest.am.train_datasets = []

    manifest.am.finetuned_model_path = None

    manifest.am.train_cmd = None
    manifest.am.infer_cmd = None

    manifest.am.train_params = edict()
    manifest.am.train_params.num_gpus = 1
    manifest.am.train_params.batch_size = 16
    manifest.am.train_params.num_epochs = 1
    manifest.am.train_params.lr = None
    manifest.am.train_params.warmup_steps = None
    manifest.am.train_params.weight_decay = None
    manifest.am.train_params.model_config = None
    manifest.am.train_params.optimizer="novograd"
    manifest.am.train_params.amp_opt_level = 'O1'
    manifest.am.train_params.beta1 = None
    manifest.am.train_params.beta2 = None
    manifest.am.train_params.finetune = True
    manifest.am.train_params.load_encoder = os.path.join(cfg.NEMO.PRETRAINED,'JasperEncoder-STEP-247400.pt')
    manifest.am.train_params.load_decoder = os.path.join(cfg.NEMO.PRETRAINED,'JasperDecoderForCTC-STEP-247400.pt')
    manifest.am.train_params.work_dir = os.path.join(cfg.MODEL.AM.PATH, manifest.id)

    # Language Model
    manifest.lm = edict()
    manifest.lm.train_dataset_name = None
    manifest.lm.train_dataset = None
    manifest.lm.ngram = 6

    manifest.lm.finetuned_model = os.path.join(cfg.MODEL.LM.PATH, manifest.id + '_lm.binary')
    manifest.lm.train_cmd = None
    manifest.lm.infer_cmd = None

    return Reader(manifest)

  def __init__(self, manifest):
    """Initialize
    Arguments:
      manifest: Manifest describing the reader
    """
    super(Reader, self).__init__(manifest)
    if not self.manifest.is_built:
      self.load_am_config_file()
      self.load_inf_config_file()
      self.manifest.is_built = True
      self.save_manifest()
      print ('Manifest is saved', self.manifest.manifest_path, '\n')
    else:
      print ('Manifest restored from', self.manifest.manifest_path, '\n')

  def get_manifest_file_path(self):
    """Get manifest file path

    Returns:
      manifest_file_path: Path of manifest file
    """
    return self.manifest.manifest_path

  ##############################################################################
  # Inference
  ##############################################################################
  def load_inf_config_file(self, config_file=os.path.join(cfg.NEMO.CONFIGS,
                                                          'quartznet15x5.yaml')):
    """Loads default config file - uses the same as AM config
    Agruments:
       config_file - path to the config file
    """
    self.manifest.inference_params.model_config = self.manifest.am.train_params.model_config
    print ('Inference config file:',  self.manifest.inference_params.model_config)
    self.save_manifest()

  def get_inference_cmd(self, model_id):
    """Returns command to run inference
    Arguments:
      model_id: named used to identify model used by inference
    """
    config_changes = []

    if len(self.manifest.eval_datasets) == 0:
        print ('No evaluation dataset is added to the reader')
        return

    # configs
    for key, value in self.manifest.inference_params.items():
        if value is not None and value:
            config_changes.append('--' + key + '=' + str(value))

    # datasets
    config_changes.append('--eval_datasets=' + str(','.join(self.manifest.eval_datasets)))
    config_changes.append('--model_id='+ model_id)

    no_gpus = 1
    inf_file = os.path.join(cfg.NEMO.TOOLS, 'jasper_eval.py')

    if no_gpus == 1:
      # single gpu
      cmd = "python {} {}".format(inf_file, ' '.join(config_changes))
    else:
      # multi-gpu
      cmd = "python -m torch.distributed.launch --nproc_per_node={} {} {}".format(no_gpus, inf_file, ' '.join(config_changes))
    if self.manifest.inference_params.lm_path is not None:
      self.manifest.lm.infer_cmd = cmd
    else:
      self.manifest.am.infer_cmd = cmd

    self.save_manifest()
    return cmd

  ##############################################################################
  # Add Results
  ##############################################################################
  def add_inference_results(self):
    """Adds inference results to the project
    Arguments:
       self - project manifest
    """
    # list of inferences in project
    inference_results = os.listdir(self.manifest.inference_params.save_results)
    inference_results = [os.path.join(
      self.manifest.inference_params.save_results, s) for s in inference_results]
    # add results
    for inf_file in inference_results:
      inf = import_file_path(inf_file)

      dataset = inf['dataset']
      if dataset not in self.manifest.inference.keys():
        self.manifest.inference[dataset] = edict()

      model_id = inf['model_id']
      if model_id not in self.manifest.inference[dataset].keys():
        self.manifest.inference[dataset][model_id] = edict()
        self.manifest.inference[dataset][model_id].wer = inf['wer']
        self.manifest.inference[dataset][model_id].path = inf_file
        self.manifest.inference[dataset][model_id].lm_wer = inf['lm_wer']
        print("Added results for model {} - {}.".format(model_id, dataset))
      else:
        print("Results for model '{}' already exists - to replace delete prior results).".format(model_id))
    self.save_manifest()

  def get_inf_path(self, dataset, model_id):
    """Gets the path to inference results
    Arguments:
      dataset: path to evaluation dataset
      model_id: named used to id model when running inference
    """
    return self.manifest.inference[dataset][model_id].path

  ##############################################################################
  # Tools for acoustic model
  ##############################################################################
  def remove_decoder(self):
    self.manifest.am.train_params.load_decoder = None
    print("Decoder removed - load_decoder set to: {}".format(self.manifest.am.train_params.load_decoder))
    self.save_manifest()


  def load_am_config_file(self, config_file=os.path.join(cfg.NEMO.CONFIGS,
                                                         'quartznet15x5.yaml')):
    """Loads default config file,renames and saves it to project's config folder
    Agruments:
       config_file - path to the config file
    """
    mkdir_p(cfg.MODEL.AM.CONFIG_FILES)
    config_name = config_file.split('/')[-1]
    new_config_file_path = os.path.join(cfg.MODEL.AM.CONFIG_FILES,
                                        self.manifest.id+'_'+cfg.MODEL.AM.NAME+'_'+config_name)
    shutil.copy(config_file, new_config_file_path)
    self.manifest.am.train_params.model_config = new_config_file_path
    print ('Training config file:', self.manifest.am.train_params.model_config)
    self.save_manifest()

  def clean_am_workdir(self):
    """Cleans work dir for acoustic model
    """
    model_path = self._construct_name(name=self.manifest.id + '_finetuning')
    rm_rf(model_path)
    print (model_path, 'is now empty')

  def _construct_name(self, name=''):
    """Constructs name of acoustic model folder
    """
    if self.manifest.am.train_params.lr is None:
      lr = 0.02
    else:
      lr = self.manifest.am.train_params.lr
    if self.manifest.am.train_params.weight_decay is None:
      wd = 0.0
    else:
      wd = self.manifest.am.train_params.weight_decay
    model_folder = ("{0}-lr_{1}-bs_{2}-e_{3}-wd_{4}-opt_{5}".format(
        name,
        lr,
        self.manifest.am.train_params.batch_size,
        self.manifest.am.train_params.num_epochs,
        wd,
        self.manifest.am.train_params.optimizer))
    model_path = os.path.join(self.manifest.am.train_params.work_dir,
                              model_folder)
    self.manifest.am.finetuned_model_path = os.path.join(model_path,
                                                         "checkpoints")
    return model_path

  def get_am_train_cmd(self):
    """Returns command to train acoustic model
    """
    config_changes = []
    if len(self.manifest.am.train_datasets) == 0:
        print ('No training dataset is added to the reader')
        return

    # configs
    for key, value in self.manifest.am.train_params.items():
        if value is not None and value:
          if key == 'finetune':
            config_changes.append('--' + key)
          elif key == 'load_encoder' or key == 'load_decoder' and self.manifest.am.train_params.finetune:
            config_changes.append('--' + key + '=' + str(value))
          elif key == 'num_gpus':
            pass
          else:
            config_changes.append('--' + key + '=' + str(value))
    config_changes.append('--exp_name=' + self.manifest.id + '_finetuning')

    # datasets
    config_changes.append('--train_dataset=' + str(','.join(self.manifest.am.train_datasets)))
    if len(self.manifest.eval_datasets) != 0:
      config_changes.append('--eval_datasets=' + str(','.join(self.manifest.eval_datasets)))

    no_gpus = self.manifest.am.train_params.num_gpus

    if no_gpus == 1:
      # single gpu
      cmd = "python {} {}".format(cfg.MODEL.AM.TRAIN_SCRIPT, ' '.join(config_changes))
    else:
      # multi-gpu
      cmd = "python -m torch.distributed.launch --nproc_per_node={} {} {}".format(no_gpus, cfg.MODEL.AM.TRAIN_SCRIPT, ' '.join(config_changes))
    self.manifest.am.train_cmd = cmd

    path = self._construct_name(name=self.manifest.id + '_finetuning')

    self.save_manifest()
    return cmd

  def set_am_pretrained_model(self, model=cfg.NEMO.PRETRAINED):
      """Loads pretrained acoustic model
      Arguments:
         path to a pretrained acoustic model
      """
      self.manifest.am.train_params.pretrained_model = model
      self.save_manifest()

  def set_am_num_gpus(self, num_gpus=4):
      """Sets number of GPUs to use for training acoustic model
      Arguments:
        num_gpus: number of GPUs
      """
      self.manifest.am.train_params.num_gpus = num_gpus
      self.save_manifest()

  def set_am_batch_size(self, batch_size=4):
      """Sets the batch size per GPU for training acoustic model
      Arguments:
        batch_size: training batch_size
      """
      self.manifest.am.train_params.batch_size = batch_size
      self.save_manifest()

  def set_am_num_epochs(self, num_epochs):
      """Sets number of epochs for training acoustic model
      Arguments:
        num_epochs: number of training epochs
      """
      self.manifest.am.train_params.num_epochs = num_epochs
      self.save_manifest()

  def set_am_learning_rate(self, lr=0.0001):
      """Sets learning rate to use for training acoustic model
      Arguments:
        lr: training learning rate
      """
      self.manifest.am.train_params.lr = lr
      self.save_manifest()

  ##############################################################################
  # Tools for Datasets
  ##############################################################################
  def add_dataset(self, dataset_file, dataset_id, dataset_type="am-train"):
    """Add a dataset_file for training or evaluation
    Arguments:
    dataset_file: a dataset file to add as training or inference dataset
    dataset_id: id used to identify dataset
    dataset_type: am-train, lm-train or eval dataset
    """
    if dataset_type == "am-train":
      if dataset_id in self.manifest.am.train_dataset_names:
        print (dataset_id, 'is already included in the am-train dataset')
        return
      self.manifest.am.train_datasets.append(dataset_file)
      self.manifest.am.train_dataset_names.append(dataset_id)

    elif dataset_type == "eval":
      if dataset_id in self.manifest.eval_dataset_names:
        print (dataset_id, 'is already included in the eval dataset')
        return
      self.manifest.eval_datasets.append(dataset_file)
      self.manifest.eval_dataset_names.append(dataset_id)

    elif dataset_type == "lm-train":
      if dataset_id == self.manifest.lm.train_dataset_name:
        print (dataset_id, 'is already included in the lm-train dataset')
        return
      self.manifest.lm.train_dataset = dataset_file
      self.manifest.lm.train_dataset_name = dataset_id
    else:
      print("No dataset added - dataset_type must be: am-train, lm-train, eval.")
    self.save_manifest()

  ##############################################################################
  # Tools for language model
  ##############################################################################
  def set_n_gram(self, n):
    """Sets n-gram size for Language model
    Arguments:
    n: number of words in n-gram
    """
    self.manifest.lm.ngram = n
    self.save_manifest()

  def get_lm_train_cmd(self):
    """Returns command to train language model
    """
    cmd = "python {} {} --n={} --project_id={}".format(
      cfg.MODEL.LM.TRAIN_SCRIPT, self.manifest.lm.train_dataset,
      self.manifest.lm.ngram, self.manifest.id)
    self.manifest.lm.train_cmd = cmd
    self.save_manifest()
    return cmd

  def clean_lm_logdir(self):
    """Cleans log dir for language model
    """
    rm_rf(self.manifest.lm.finetuned_model)
    print (self.manifest.lm.finetuned_model, 'deleted')

  def get_path_to_lm(self, LM_finetuned=False):
    """Returns path to trained language model
    Arguments:
       LM_finetuned = True - LM finetuned model, False - default pretrained model
    """
    return self.manifest.lm.finetuned_model if LM_finetuned else cfg.MODEL.LM.PRETRAINED