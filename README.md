# tokenizers

This repository contains per-character, sub-word and word level tokenizers.

## Per-Character
---
`PerCharTokenizer()` in `perChar` directory contains a character level tokenizer. It's very simple to understand and use. Each unique character present in the `train_data` builds the vocab for the tokenizer.
Not very reliable for big projects, only good for training small models for experimentation.

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

### How to use

``` python
tokenizer = PerCharTokenizer()
tokenizer.train(train_text=train_data)
tokenizer.save_model(prefix='perChar')  # saves the model
tokenizer.load(model_path='../path_to_model')  # loads the model

text = "My name is Alan"
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
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

This one is better preferred for big applications like training a LLM model. Language models don't see text like you and I, instead they see a sequence of numbers (known as tokens). Byte pair encoding (BPE) is a way of converting text into tokens. It has a couple desirable properties:

1. It's reversible and lossless, so you can convert tokens back into the original text
2. It works on arbitrary text, even text that is not in the tokenizer's training data
3. It compresses the text: the token sequence is shorter than the bytes corresponding to the original text
4. It attempts to let the model see common sub-words. For instance, "ing" is a common sub-word in English, so BPE encodings will often split "encoding" into tokens like "encod" and "ing" (instead of e.g. "enc" and "oding"). Because the model will then see the "ing" token again and again in different contexts, it helps models generalize and better understand grammar.
### How to use:

``` python
from miniBPE import BasicTokenizer
name = '../models/basicCharMap'
tokenizer = BasicTokenizer(train_text)
tokenizer.train(target_vocab=4000)
tokenizer.save_model(name)  # saves the model
tokenizer.load('../path_to_model')  # loads the model

text = "My name is Alan"
print(tokenizer.encode(text))  # encoder
print(tokenizer.decode(tokenizer.encode(text)))  # decoder
```

## Word level
---
still to build