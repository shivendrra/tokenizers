"""
  --> uses utf-8 characters to build a vocab
  --> follows byte-pair algorithm, very basic, without regex or special characters
  --> save/load works properly
"""

import json
from tqdm import tqdm
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class BasicTokenizer:
  def __init__(self) -> None:
    super().__init__()
    self.vocab_size = 0
    self.vocab = {}
    self.merges = {}

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
  
  def _build_vocab(self, merges):
    """
      - takes merges after training or loading
      - builds the new vocab by merging the merge values together
      - returns the vocab
    """

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    
    return vocab
  
  def train(self, train_data, target_vocab):
    assert target_vocab >= 256
    n_merges = target_vocab - 256
    text_bytes = train_data.encode('utf-8')
    ids = list(text_bytes)
    
    merges = {}
    for i in tqdm(range(n_merges), desc='Training the tokenizer\t'):
      stats = self._get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = 256 + i
      ids = self._merge(ids, pair, idx)
      merges[pair] = idx
    
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]

    self.vocab = vocab
    self.merges = merges
    self.vocab_size = len(self.vocab)

    return self.vocab, self.merges
  
  def encode(self, text):
    """
      - takes in the input string, encodes it using 'utf-8' encodings
      - fetches merges from saved or loaded merges
      - returns the merges
      
      Args:
        train_data (str): string of dna sequence
        self.merges (dictonary): contains merges
    """
    text_bytes = text.encode('utf-8')
    ids = list(text_bytes)
    while len(ids) >= 2:
      stats = self._get_stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break

      idx = self.merges[pair]
      ids = self._merge(ids, pair, idx)
    return ids
  
  def decode(self, ids):
    """
      - takes in the input list
      - fetches the index from the vocab and joins the value
      - decodes the 'utf-8' tokens into string and returns it
      
      Args:
        train_data (list[int]): list containing integers
        self.vocab (dictonary): contains final vocab
    """
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    text = text_bytes.decode('utf-8', errors='replace')
    return text

  def save_model(self, prefix):
    """
      - basic save_model() funtion, saves two files, '.model' & 'vocab.json'
      - '.model' contains all the final merges, each on next line
      - 'vocab.json' contains the final vocab, for human interpretation

      Args:
        prefix (str): prefix along with the path
        self.merges (dict): contains final merges
        self.vocab (dict): contains final vocab
    """
    model_file = prefix + '.model'
    with open(model_file, 'w') as f:
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")
    
    vocab_file = prefix + '_vocab.json'
    with open(vocab_file, 'w') as f:
      serializable_vocab = {}
      for idx in self.vocab:
        try:
          serializable_vocab[idx] = str(self.vocab[idx], 'utf-8')
        except UnicodeDecodeError:
          pass
      json.dump(serializable_vocab, f)
  
  def load_model(self, model_file):
    """
      - loads the '.model' file
      - re-writes the merges in the new merges dict
      - builds the vocab again for further use

      Args:
        model_path (str): path to the '.model' file
    """
    assert model_file.endswith('.model')

    merges = {}
    idx = 256
    with open(model_file, 'r', encoding='utf-8') as f:
      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1

    self.merges = merges
    self.vocab = self._build_vocab(merges)