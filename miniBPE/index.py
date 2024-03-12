"""
  new kind of tokenizer, same as bpe but different vocab building
  still in progress
"""

from tqdm import tqdm
import json
import time
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class Tokenizer:
  def __init__(self):
    super().__init__()
    self.chars = []
    self.vocab = {}
    self.merges = {}
    self.vocab_size = len(self.chars)
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
  
  def _build_vocab(self, merges):
    init_vocab = {i: ids for i, ids in enumerate(self.chars)}
    for (p0, p1), idx in merges.items():
      init_vocab[idx] = init_vocab[p0] + init_vocab[p1]

    return init_vocab
  
  def load_model(self, file_prefix):
    """
      loads '.model' file that contains merges
      loads '.json' file that contains vocab 
    """
    model_file = file_prefix + '.model'
    vocab_file = file_prefix + '.json'
    merges = {}
    idx = self.vocab_size
    print(idx)
    with open(model_file, 'r', encoding='utf-8') as f:
      version = f.readline().strip()
      assert version == "minibpe v1"
      print('loading model....')
      for line in f:
        if line.startswith('chars:'):
          chars = line.split(':')[1].strip().strip('[]').split(', ')
          continue

        tokens = line.split()
        if len(tokens) != 2:
          continue  # Skip lines that do not contain two integers

        idx1, idx2 = map(int, tokens)
        merges[(idx1, idx2)] = idx
        idx += 1

    self.chars = [char.strip("'") for char in chars]
    self.merges = merges
    # self.vocab1 = self._build_vocab(merges1)
    
    with open(vocab_file, 'r') as f:
      vocab = json.load(f)
    self.vocab = {int(k): v for k, v in vocab.items()}

tokenizer = Tokenizer()
tokenizer.load_model('../models/base30k')
print("chars: ", tokenizer.chars)

text = 'can I talk to you, I wanted to tell you something'
print("encoded: \n", tokenizer.encode(text))
print("decoded: \n", tokenizer.decode(tokenizer.encode(text)))

import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

text = "Hello, nice to meet you"

tokenizer.encode(text)