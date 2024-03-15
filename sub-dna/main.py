import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class DNAtokenizer:
  def __init__(self):
    super().__init__()
    self.vocab_size = 0
    self.chars = []
    self.vocab = {}
    self.merges = {}
    self.string_to_index = { ch:i for i,ch in enumerate(self.chars) }
    self.index_to_string = { i:ch for i,ch in enumerate(self.chars) }
  
  def _encode(self, string):
    encoded = [self.string_to_index[char] for char in string]
    return encoded
  
  def _decode(self, integer):
    decoded = ''.join([self.index_to_string[i] for i in integer])
    return decoded

  def _get_stats(self, ids, counts=None):
    """
      takes list of integers and returns dictionary of counts of pairs(consecutive ones)
      eg: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
      allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
      counts[pair] = counts.get(pair, 0) + 1
    return counts
  
  def _merge(self, ids, pair, idx):
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
  
  def train(self, train_data, vocab_size):
    
    pass