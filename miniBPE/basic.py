"""
  --> uses utf-8 characters to build a vocab
  --> follows byte-pair algorithm, very basic, without regex or special characters
  --> save/load works properly
"""

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
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    text = text_bytes.decode('utf-8', errors='replace')
    return text

  def save_model(self, prefix):
    """
      saves all the merges in .model file
    """
    model_file = prefix + '.model'
    with open(model_file, 'w') as f:
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")
  
  def load_model(self, model_file):
    """
      --> loads the saved model file
      --> re-builds the merges and vocab for later use
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