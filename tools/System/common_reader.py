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

import abc

import os
import json
from easydict import EasyDict as edict

from tools.filetools import mkdir_p, rm_rf
from tools.System.autoloader import obj_to_class_str, str_to_class
from tools.System.config import cfg


class CommonReader(object):
  """Interface class for all readers
  """
  __metaclass__ = abc.ABCMeta

  @staticmethod
  def load_manifest(obj_id, reader_class):
    """Load manifest from file for a given object into a reader class

    Args:
      obj_id: String id of object
      reader_class: Class of type of object to construct

    Returns:
      obj: Instantiated object based on manifest
    """
    manifest_file = os.path.join(cfg.MANIFEST.PATH, str(obj_id) + '_' + cfg.MANIFEST.FILE)

    if os.path.exists(manifest_file) and os.path.isfile(manifest_file):
      with open(manifest_file, 'r') as f:
        manifest = edict(json.load(f))
        return str_to_class(manifest.reader_type)(manifest)
    else:
      raise IOError('Manifest file not found at %s' % manifest_file)

  def __init__(self, manifest):
    """Initialize

    Args:
      manifest: Dict containing manifest
    """
    self.opened = False
    self.manifest = manifest
    if not self.manifest.is_built:
      self.manifest.reader_type = obj_to_class_str(self)

  def save_manifest(self):
    """Save manifest file
    """
    # error check to make sure none of the dict items is empty:
    for k in self.manifest:
      if self.manifest[k] is None:
        raise AttributeError("No value set in manifest for key %s" % k)
    manifest_file = self.get_manifest_file_path()
    mkdir_p(os.path.dirname(manifest_file))
    with open(manifest_file, 'w') as f:
      json.dump(self.manifest, f, indent=4)

  def _ensure_opened(self):
    if not self.opened:
      raise RuntimeError("Please call open context manager prior to operation")

  @abc.abstractmethod
  def open(self):
    """Open file descriptor for reader
    """

  @abc.abstractmethod
  def close(self):
    """Close file descriptor for reader
    """

  def delete(self):
    """Delete this object and associated storage
    """
    obj_folder = os.path.join(self.storage_base_path(), str(self.manifest.id))
    rm_rf(obj_folder)

  def __enter__(self):
    """Context manager help - enter
    """
    self.open()
    return self

  def __exit__(self, type, value, traceback):
    """Context manager help - exit
    """
    self.close()