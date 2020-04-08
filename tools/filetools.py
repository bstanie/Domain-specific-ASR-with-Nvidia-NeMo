import os
import shutil
import errno
import json

"""Common file tools that replicate shell functions
"""

def rm_rf(path):
  """Remove path
  """
  if not os.path.exists(path):
    print("{} does not exist.".format(path))
    return
  if os.path.isfile(path) or os.path.islink(path):
    os.unlink(path)
  else:
    shutil.rmtree(path)


def mkdir_p(path):
  """Make directory path
  """
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

def reset_path(path):
  """Remove path and recreate the same path
  """
  rm_rf(path)
  mkdir_p(path)


def symlink(src, dst):
  """Symlink wrapper
  """
  os.symlink(src, dst)


def file_exists(path):
  """Test if file exists

  Returns:
    bool: True if file exists
  """
  return os.path.exists(path) and (os.path.isfile(path) or os.path.islink(path))

def save_to_file(text, folder, filename):
  """Save text to file
  Arguments:
    text: text to save to file
    folder: path or folder to save file at
    filename: Name of file to save text to
  """
  mkdir_p(folder)
  out_file_path = os.path.join(folder, filename)
  with open(out_file_path, 'w') as fd:
    print(text, file=fd)

def display_file(folder, filename):
  """Display file from folder
  Arguments:
    folder: path or folder to save file at
    filename: Name of file to save text to
  """
  transcript = import_file(folder, filename)
  print("\n",filename)
  print(transcript)
  return transcript

def import_file_path(path):
  """Import file from path
  Arguments:
    path: path to file to import
  """
  with open(path, 'r') as file:
    if '.txt' in path:
      data = file.read()
    elif '.json' in path:
      data = json.load(file)
    else:
      assert False
  return data

def import_file(folder, filename):
  """Import file from folder
  Arguments:
    folder: path or folder
    filename: Name of file to import
  """
  path = os.path.join(folder, filename)
  with open(path, 'r') as file:
    if '.txt' in filename:
      data = file.read()
    elif '.json' in filename:
      data = json.load(file)
    else:
      assert False
  return data