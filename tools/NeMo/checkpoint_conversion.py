import torch
from collections import OrderedDict

import argparse

parser = argparse.ArgumentParser(
    description='Convert NeMo checkpoints from 0.8.2 to master')
parser.add_argument('--prev', type=str,
    required=True, help='0.8.2 checkpoint')
parser.add_argument('--new', type=str,
    required=True, help='master checkpoint')
args = parser.parse_args()

prev_dict = torch.load(args.prev)
new_dict = OrderedDict()

for k in prev_dict:
    new_k = k.replace('.conv.', '.mconv.')
    if len(prev_dict[k].shape)==3:
        new_k = new_k.replace('.weight', '.conv.weight')
    new_dict[new_k] = prev_dict[k]
torch.save(new_dict, args.new)