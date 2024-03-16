"""
  from karpathy's minBPE
  doesn't works properly, have to fix it
"""

import regex as re
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from .base import get_stats, merge
regex_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:
  def __init__(self, pattern=None):
    super().__init__()
    self.pattern = regex_pattern if pattern is None else pattern
    self.compiled_pattern = re.compile(self.pattern)
    self.special_tokens = {}
    self.inverse_special_tokens = {}
  
  def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256
    n_merges = vocab_size - 256

    # text_chunks = re.findall(self.compiled_pattern, text)
    # ids = [list(ch.encode('utf-8') for ch in text_chunks)]
    tokens = list(text.encode('utf-8'))
    ids = [tokens]

    merges = {}
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i in range(n_merges):
      stats = {}
      for chunk_ids in ids:
        stats = get_stats(chunk_ids, stats)
      
      pair = max(stats, key=stats.get)
      idx = 256 + i
      ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
      merges[pair] = idx
    
      if verbose:
        pass
    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    
    self.vocab = vocab
    self.merges = merges
  
  def register_special_token(self, special_tokens):
    self.special_tokens = special_tokens
    self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
  
  def decode(self, ids):
    part_bytes = []
    for idx in ids:
      if idx in self.vocab:
        part_bytes.append(self.vocab[idx])
      elif idx in self.inverse_special_tokens:
        part_bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))
      else:
        raise ValueError(f"invalid token id: {idx}")
    
    text_bytes = b"".join(part_bytes)
    text = text_bytes.decode('utf-8', errors='replace')
    return text
  
  def _encode_chunk(self, text_bytes):
    ids = list(text_bytes)
    while len(ids) > 2:
      stats = get_stats(ids)
      pair = min(stats, key= lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break
      
      idx = self.merges[pair]
      ids = merge(ids, pair, idx)
    return ids
  
  def encode_ordinary(self, text):
    """Encoding that ignores any special tokens."""
    text_chunks = re.findall(self.compiled_pattern, text)
    ids = []
    for chunk in text_chunks:
      chunk_bytes = chunk.encode("utf-8")
      chunk_ids = self._encode_chunk(chunk_bytes)
      ids.extend(chunk_ids)
    return ids
  
  def encode(self, text, allowed_special="none_raise"):
    special = None
    if allowed_special == 'all':
      special = self.special_tokens
    elif allowed_special == 'none':
      special = {}
    elif allowed_special == 'none_raise':
      special = {}
      assert all(token not in text for token in self.special_tokens)
    elif isinstance(allowed_special, set):
      special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
    else:
      raise ValueError(f"allowed_special = {allowed_special} not understood")
    
    if not special:
      return self.encode_ordinary(text)
    
    special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
    special_chunks = re.split(special_pattern, text)

    ids = []
    for part in special_chunks:
      if part in special:
        ids.append(special[part])
      else:
        ids.extend(self.encode_ordinary(part))
    
    return ids