"""
  -> this is a basic tokenizer without special character removal or regex
  -> uses set of unique characters present in training data as the initial vocab instead
  utf-8 encodings
  -> saves the trained model
"""

from tqdm import tqdm
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class BasicTokenizer:
  def __init__(self, train_text):
    super().__init__()
    self.chars = sorted(list(set(train_text)))
    self.vocab_size = len(self.chars)
    self.string_to_index = { ch:i for i,ch in enumerate(self.chars) }
    self.index_to_string = { i:ch for i,ch in enumerate(self.chars) }

  def encode(self, string):
    for char in string:
      s = self.string_to_index(char)
      return s
  
  def decode(self, integer):
    for no in integer:
      s = self.index_to_string(no)
      return s
  
  def get_stats(self, ids, counts=None):
    """
      takes list of integers and returns dictionary of counts of pairs(consecutive ones)
      eg: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
      allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
      counts[pair] = counts.get(pair, 0) + 1
    return counts
  
  def merge(self, ids, pair, idx):
    """
      in the list of integers, replaces all consecutive pair with the new integer token idx
      eg: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
      if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
        new_ids.append(idx)
        i += 2
      else:
        new_ids.append(ids[i])
        i += 1
    return new_ids
  
  def train_model(self, train_data, n_merges):
    pass