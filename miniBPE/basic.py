"""
  basic bpe tokenizer with encoder and decoder class
  doesn't use regex and special tokens
"""

import os

current_directory = os.path.dirname(os.path.relpath(__file__))
os.chdir(current_directory)

def get_stats(ids):
  counts = {}
  for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
  return counts

def merge(ids, pair, idx):
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

def encode(text):
  tokens = list(text.encode('utf-8'))
  while True:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float('inf')))
    if pair not in merges:
      break
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

def decode(ids):
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode('utf-8', errors='replace')
  return text

vocab_sie = 456
n_merges = vocab_sie - 256

merges = {}
for i in range(n_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  ids = merge(ids, pair, idx)
  merges[pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
  vocab[idx] = vocab[p0] + vocab[p1]