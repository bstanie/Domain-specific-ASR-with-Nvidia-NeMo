
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
import difflib

from tools.filetools import import_file_path
"""Miscellaneous tools
"""

def show_diff(text, n_text):
  """
  Unify operations between two compared strings seqm is a difflib.
  SequenceMatcher instance whose a & b are strings
  """
  seqm = difflib.SequenceMatcher(None, text, n_text)
  output= []
  for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
    if opcode == 'equal':
      output.append(seqm.a[a0:a1])
    elif opcode == 'insert':
      output.append("<font color=#32CD32>^" + seqm.b[b0:b1] + "</font>")
    elif opcode == 'delete':
      output.append("<font color=red>^" + seqm.a[a0:a1] + "</font>")
    elif opcode == 'replace':
      # seqm.a[a0:a1] -> seqm.b[b0:b1]
      output.append("<b><font color=blue>^" + seqm.b[b0:b1] + "</font></b>")
    else:
      raise (RuntimeError, "unexpected opcode")
  return ''.join(output)

def get_transcript(path, lm=False):
  """Get transcript from inference results
    Arguments:
      path: path to inference results
      lm: True if want beam transcript
  """
  inf = import_file_path(path)
  if lm:
    return inf['beam transcript']
  else:
    return inf['transcript']

def get_gtruth(path):
  """Get ground truth from inference results
  Arguments:
  path: path to inference results
  """
  inf = import_file_path(path)
  return inf['gtruth']


def show_values_on_bars(axs):
  """Show bar values on barplot"""
  def _show_on_single_plot(ax):
    for p in ax.patches:
      _x = p.get_x() + p.get_width() / 2
      _y = p.get_y() + p.get_height()
      value = '{:.2f}'.format(p.get_height())
      ax.text(_x, _y, value, ha="center", fontdict={'weight':'bold','size':20})

  if isinstance(axs, np.ndarray):
    for idx, ax in np.ndenumerate(axs):
      _show_on_single_plot(ax)
  else:
    _show_on_single_plot(axs)

def parse_manifest_wer(inferences, sort_metric="", keep=[]):
    """Parse manifest Word Error Rate
    Arguments:
      inferences: manifest inference results
      sort_metric: column name to sort by
      keep: inference type or name to keep in dataframe
    """
    df = pd.DataFrame(inferences).T
    df = df.drop('path', 1)
    df.wer = df.wer.astype('float')
    df.lm_wer = df.lm_wer.astype('float')
    df['percentWER'] = np.where(df['lm_wer'].isnull(), df['wer']*100, df['lm_wer']*100)
    df = df.rename_axis('inference_types').reset_index()
    if keep:
      df = df[df.inference_types.isin(keep)]
    if sort_metric:
      df = df.sort_values(by=sort_metric, ascending=True)
    return df

def barplot_manifest(data, metric, title="", xlabel="", ylabel=""):
    """Plot manifest results
    Arguments:
      data: Parsed results dataframe
      metric: Metric to use to plot data
      title: plot title
      xlabel, ylabel: Labels to use for x and y axis
    """
    font_labels = {'weight':'bold','size':20}
    font_title = {'weight':'bold','size':30}
    ax = data.plot.bar(x='inference_types', y=metric, rot=1, figsize=(20, 9),
                       legend=False, grid=True, color=['C0', 'C1', 'C2', 'C3',
                                                       'C4', 'C6', 'C7', 'C9'])
    ax.set_title(title, font_title)
    ax.set_xlabel(xlabel, font_labels)
    ax.set_ylabel(ylabel, font_labels)
    ax.tick_params(axis="x", labelsize=15, labelrotation=-10)
    ax.tick_params(axis="y", labelsize=20)
    show_values_on_bars(ax)
    return ax

def create_lm_dataset(json_path, out_txt):
    """Create dataset for language model training
    Arguments:
      json_path: Path to json dataset
      out_txt: Path to output lm dataset (txt)
    """
    # read list of dicts
    lines = []
    for line in open(json_path, 'r'):
        lines.append(json.loads(line))

    # get text of each dict
    texts= [l['text'] for l in lines]

    # save output
    with open(out_txt, 'w') as filehandle:
        for listitem in texts:
            filehandle.write('%s\n' % listitem)
    print("Created lm dataset {}".format(out_txt))