# Copyright (c) 2019 NVIDIA Corporation
import pandas as pd
import os
import argparse
import shutil
import errno
from tools.System.config import cfg
from tools.filetools import mkdir_p

if __name__ == '__main__':
  """Build Languauge Model
  Arguments:
    text: text file with lm dataset
    n: number of words for n-gram
    project_id: used to identify model
  """
  # arguments text file
  parser = argparse.ArgumentParser(description='Build N-gram LM model from TXT files')
  parser.add_argument('text', metavar='text', type=str, help='text file')
  parser.add_argument('--n', type=int, help='n for n-grams', default=3)
  parser.add_argument('--project_id', type=str, help='project id', default='lm')
  args = parser.parse_args()

  # directories
  output_dir = cfg.MODEL.LM.PATH
  mkdir_p(output_dir)
  corpus_name = os.path.join(output_dir, args.project_id + '.txt')
  arpa_name = os.path.join(output_dir, args.project_id + '.arpa')
  lm_name = os.path.join(output_dir, args.project_id + '_lm.binary')
  trie_name = os.path.join(output_dir, args.project_id + '-lm.trie')

  # copy training .txt to project with correct name
  shutil.copy(args.text, corpus_name)

  # build lm model
  command = cfg.MODEL.LM.DECODERS + '/kenlm/build/bin/lmplz --text {} --arpa {} --o {}'.format(
  corpus_name, arpa_name, args.n)
  print(command)
  os.system(command)

  command = cfg.MODEL.LM.DECODERS + '/kenlm/build/bin/build_binary trie -q 8 -b 7 -a 256 {} {}'.format(arpa_name, lm_name)
  print(command)
  os.system(command)

  os.remove(arpa_name)