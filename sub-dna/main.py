import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from tqdm import tqdm

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

  def _build_vocab(self):
    return {i: ids for i, ids in enumerate(self.chars)}

  def train(self, train_data, target_vocab):
    self.chars = sorted(list(set(train_data)))
    self.string_to_index = {ch: i for i, ch in enumerate(self.chars)}
    self.index_to_string = {i: ch for i, ch in enumerate(self.chars)}
    vocab = self._build_vocab()
    vocab_size = len(vocab)
    
    tokens = self._encode(train_data)
    ids = list(tokens)
    merges = {}
    for i in tqdm(range(target_vocab), desc='Training the tokenizer\t'):
      stats = self._get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = vocab_size + i
      ids = self._merge(ids, pair, idx)
      merges[pair] = idx
    
    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    
    self.vocab = vocab
    self.merges = merges
    self.vocab_size = len(self.vocab)
    return self.vocab, self.merges
  
  def encode(self, text):
    tokens = self._encode(text)
    ids = list(tokens)
    while len(ids) >= 2:
      stats = self._get_stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break

      idx = self.merges[pair]
      ids = self._merge(ids, pair, idx)
    return ids

  def decode(self, ids):
    tokens = ''.join(self.vocab[idx] for idx in ids)
    seq = self._decode(tokens)
    return seq

with open('../train files/new_dna_1.txt', 'r', encoding='utf-8') as f:
  data = f.read()

token = DNAtokenizer()
vocab, size = token.train(data, 200)

sample = 'AAACGCTCACCGTAATGGTCAGCCAGAGTGTTGAACGTCTTGATTCGCTTGACGCTGCTG'
print(token.encode(sample))
print(sample == token.decode(token.encode(sample)))