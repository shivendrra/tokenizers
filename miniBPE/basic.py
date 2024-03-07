"""
  -> this is a basic tokenizer without special character removal or regex
  -> uses set of unique characters present in training data as the initial vocab instead
  utf-8 encodings
  -> saves the trained model
"""

from tqdm import tqdm
import multiprocessing
import json
import time
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class BasicTokenizer:
  def __init__(self):
    super().__init__()
    self.chars = []
    self.vocab_size = 0

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
  
  def _build_vocab(self, merges):
    vocab = {i: ids for i, ids in enumerate(self.chars)}
    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]

    return vocab
  
  def init_var(self, train_text=None):
    self.chars = sorted(list(set(train_text))) if train_text is not None else self.chars
    print(self.chars)
    self.train_data = train_text
    self.vocab_size = len(self.chars)
    self.string_to_index = { ch:i for i,ch in enumerate(self.chars) }
    self.index_to_string = { i:ch for i,ch in enumerate(self.chars) }

  def train(self, train_data, target_vocab):
    self.init_var(train_data)
    tokens = list(self._encode(self.train_data))
    ids = list(tokens)
    n_merges = target_vocab
    
    merges = {}
    iteration_times = []

    for i in tqdm(range(n_merges), desc='Training the tokenizer\t'):
      start_time = time.time()
      
      stats = self._get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = self.vocab_size + i
      ids = self._merge(ids, pair, idx)
      merges[pair] = idx
      
      iteration_times.append((time.time() - start_time))
    
    self.vocab = self._build_vocab(merges)
    self.merges = merges
    
    return iteration_times

  def _train_merge(self, merge_idx):
    stats = self._get_stats(self.ids)
    pair = max(stats, key=stats.get)
    idx = self.vocab_size + merge_idx
    ids = self._merge(self.ids, pair, idx)
    return pair, ids

  def train_parallel(self, target_vocab, num_processes=None):
    tokens = list(self._encode(self.train_data))
    self.ids = list(tokens)
    n_merges = target_vocab - self.vocab_size
    merge_indices = range(n_merges)

    with multiprocessing.Pool(processes=num_processes) as pool:
      results = list(tqdm(pool.imap(self._train_merge, merge_indices), total=n_merges, desc='Training the tokenizer\t'))

    # with multiprocessing.Pool(processes=num_processes) as pool:
    #   with tqdm(total=n_merges, desc='Training the tokenizer\t') as pbar:
    #     results = list(pool.imap(self._train_merge, merge_indices))
    #     pbar.update(n_merges)

    merges = {pair: ids for pair, ids in results}

    self.vocab = self._build_vocab(merges)
    self.merges = merges

    return self.vocab, self.merges

  def encode(self, en_text):
    tokens = list(self._encode(en_text))
    while True:
      stats = self._get_stats(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      tokens = self._merge(tokens, pair, idx)
    return tokens
  
  def decode(self, de_text):
    tokens = [self.vocab[idx] for idx in de_text]
    text = ''.join(tokens)
    return text

  def save_model(self, prefix):
    """
      Saves two files: file_prefix.vocab and file_prefix.model
      This is inspired (but not equivalent to!) sentencepiece's model saving:
      - model file is the critical one, intended for load()
      - vocab file is just a json version for human inspection only
    """
    file_name = prefix + '.model'
    with open(file_name, 'w', encoding='utf-8') as f:
      f.write("minibpe v1\n")
      f.write(f"chars: {self.chars}\n\n")
      for idx1, idx2 in tqdm(self.merges, desc='Saving model\t\t', unit='merge'):
        f.write(f"{idx1} {idx2}\n")

    vocab_file = prefix + '_vocab.json'
    with open(vocab_file, 'w') as f:
      json.dump(self.vocab, f)

    print(f'file saved successfully!')

  def load_model(self, model_file):
    """
      loads '.model' file
      returns vocab loaded from the file
    """
    assert model_file.endswith('.model')
    merges = {}
    idx = self.vocab_size
    with open(model_file, 'r', encoding='utf-8') as f:
      version = f.readline().strip()
      assert version == "minibpe v1"
      for line in tqdm(f, desc='Loading model\t\t'):
        if line.startswith('chars:'):
          self.chars = line.split(':')[1].strip().strip('[]').split(', ')
          print(self.chars)
          continue

        tokens = line.split()
        if len(tokens) != 2:
          continue  # Skip lines that do not contain two integers

        idx1, idx2 = map(int, tokens)
        merges[(idx1, idx2)] = idx
        idx += 1

        # idx1, idx2 = map(int, line.split())
        # merges[(idx1, idx2)] = idx
        # idx += 1
      self.chars = [char.strip("'") for char in self.chars]
      self.init_var()
      self.merges = merges
      self.vocab = self._build_vocab(self.merges)
    
    # vocab_file = prefix + '_vocab.json'
    # with open(vocab_file, 'r') as f:
    #   vocab = json.load(f)

    return self.vocab, self.merges