# Utility functions to display and parse transcripts
import json
import os
from num2words import num2words

def to_lower(transcript):
  """Send string or list of strings to lowercase"""
  if isinstance(transcript, str):
    return transcript.lower()
  elif all(isinstance(item, str) for item in transcript):
    return [w.lower() for w in transcript]
  else:
    raise TypeError("Only accepts strings or list of strings")

def del_p(string, all_p=False):
  """Delete punctuation"""
  punctuations ='''!()-[]{};:"\,<>./?@#$%^&*_~“„’´ʻ…–'''
  if all_p:
    # also remove '
    punctuations ='''!()-[]{};:'"\,<>./?@#$%^&*_~“„’´ʻ…–'''
  for x in string:
    if x in punctuations:
      if x in '''-–''':
        string = string.replace(x, " ")
      else:
        string = string.replace(x, "")
  return string

def remove_punct(transcript, all_p=False):
  """Remove punctuation from string or list of strings"""
  # string, list of strings
  if isinstance(transcript, str):
    return del_p(transcript, all_p)
  elif all(isinstance(item, str) for item in transcript):
    return [del_p(string, all_p) for string in transcript]
  else:
    raise TypeError("Only accepts strings or list of strings")

def dig_to_words(string, ln='en'):
  """Convert digits to words"""
  punctuations = '''!()-[]{};:"\,<>./?@#$%^&*_~'''
  #print(string)
  words = string.split()
  for i, word in enumerate(words):
    if any(i.isdigit() for i in word):
      if word.isdigit():
        # digits only
        num =  num2words(word, lang=ln).replace("-"," ").replace(",","")
        if word[-1] in punctuations:
          # digit ends with punctuation must still add it back
          words[i] = num + word[-1]
        else:
          words[i] = num
      else:
        # digits with letters, e.g. 3d
        # identify letters/digits and split
        numbers = [x for x in word if x.isdigit()]
        numbers = ''.join(numbers)
        chars = [x for x in word if not x.isdigit()]
        chars = ''.join(chars)
        num_tmp =  num2words(numbers, lang=ln).replace("-"," ").replace(",","")
        words[i] = num_tmp + " " + chars
  return ' '.join(words)

def remove_digits(transcript, lang='en'):
  """Convert digits to words from string or list of strings.
  You can specify what language to use."""
  # string, list of strings
  if isinstance(transcript, str):
    return dig_to_words(transcript, ln=lang)
  elif all(isinstance(item, str) for item in transcript):
    return [dig_to_words(string, ln=lang) for string in transcript]
  else:
    raise TypeError("Only accepts strings or list of strings")


def remove_abbrv(transcript, old, new):
  """Remove abbreviations from string or list of strings."""
  # string, list of strings
  if isinstance(transcript, str):
    return transcript.replace(old, new)
  elif all(isinstance(item, str) for item in transcript):
    return [string.replace(old, new) for string in transcript]
  else:
    raise TypeError("Only accepts strings or list of strings")

def normalize(text, lang='en'):
  """Function to Normalize text (English digit convertion)"""
  return remove_digits(remove_punct(to_lower(text)), lang=lang)