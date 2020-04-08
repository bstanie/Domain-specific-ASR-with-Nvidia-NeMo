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

import importlib


def str_to_class(fully_qualified_cls_name):
  """Given a string representing a fully qualified class name, return the class
  object.

  Args:
    fully_qualified_cls_name: module and class name with dot separator

  Returns:
    Class object which can be used to instantiate a new class
  """
  module_names = fully_qualified_cls_name.split(".")
  for idx, mn in enumerate(module_names):
    if idx is 0:
      continue
    module_name = '.'.join(module_names[0:idx])
    somemodule = importlib.import_module(module_name)
  return getattr(somemodule, module_names[-1])


def obj_to_class_str(obj):
  """Given an object, return a string representing a fully qualified class name.

  Args:
    obj: Instance of the class name requested

  Returns:
    String representing fully qualified class name
  """
  return obj.__module__ + "." + obj.__class__.__name__