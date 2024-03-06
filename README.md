# Sypher-tokenizer

This repository contains per-character, sub-word and word level tokenizers.

## Per-Character
---
`PerCharTokenizer()` in `perChar` directory contains a character level tokenizer. It's very simple to understand and use

```python
# this is a basic character level tokeinzer

chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoder - decoder
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: takes a string, returns a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: takes a list of integers, returns a string
```

## Sub-word
---
A byte-pair encoding tokenizer, with a little different architecture. Instead of using `'utf-8'` encodings for the initial vocab of size 256, it uses the all the unique characters present in a data set for the initial vocab which means `vocab_size` can be larger or smaller than 256 at first. Then it adds rest of the pairs and merges vocab like usual byte-pair encoder.

``` python
# basic code

class BasicTokenizer:
  def __init__(self, train_text):
    super().__init__()
    self.chars = sorted(list(set(train_text)))
    self.train_data = train_text
    self.vocab_size = len(self.chars)
    self.string_to_index = { ch:i for i,ch in enumerate(self.chars) }
    self.index_to_string = { i:ch for i,ch in enumerate(self.chars) }

  def _build_vocab(self, merges):
    vocab = {i: ids for i, ids in enumerate(self.chars)}
    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

  def train(self, target_vocab):
    tokens = list(self._encode(self.train_data))
    ids = list(tokens)
    n_merges = target_vocab - self.vocab_size
    merges = {}
    for i in tqdm(range(n_merges), desc='Training the tokenizer\t'):
      stats = self._get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = self.vocab_size + i
      ids = self._merge(ids, pair, idx)
      merges[pair] = idx
    self.vocab = self._build_vocab(merges)
    self.merges = merges
   return self.vocab, self.merges

# ... continued
```

## Word level
---
still to build