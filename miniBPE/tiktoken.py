import tiktoken
from .regex import RegexTokenizer
from .base import build_vocab, render_token

def bpe(mergeable_ranks, token, max_rank):
  parts = [bytes([b]) for b in token]
  while True:
    min_idx = None
    min_rank = None
    for i, pair in enumerate(zip(parts[:-1], parts[:1])):
      rank = mergeable_ranks.get(pair[0] + pair[1])
      if rank is not None and (min_rank is None or rank < min_rank):
        min_idx = i
        min_rank = rank
      
    if min_rank is None or (max_rank is not None and min_rank >= max_rank):
      break
    assert min_idx is not None
    parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
  return parts

def recover_merges(mergeable_ranks):
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        parts = bpe(mergeable_ranks, token, rank)
        if len(parts) != 2:
            continue
        pair = tuple(parts)
        try:
            ix0 = mergeable_ranks.get(pair[0])
            ix1 = mergeable_ranks.get(pair[1])
            if ix0 is not None and ix1 is not None:
                merges[(ix0, ix1)] = rank
        except KeyError:
            continue
    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4tokenizer(RegexTokenizer):
  def __init__(self, pattern=GPT4_SPLIT_PATTERN, special_tokens=GPT4_SPECIAL_TOKENS):
    super().__init__()
    enc = tiktoken.get_encoding("cl100k_base")
    mergeable_ranks = enc._mergeable_ranks
    self.merges = recover_merges(mergeable_ranks)
    self.pattern = pattern

    # vocab = {idx: bytes([idx]) for idx in range(256)}
    # for (p0, p1), idx in self.merges.items():
    #   vocab[idx] = vocab[p0] + vocab[p1]
    # self.vocab = vocab

    # self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
    # self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}  
    # self.register_special_tokens = special_tokens

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in self.merges.items():
      try:
        vocab[idx] = vocab[p0] + vocab[p1]
      except KeyError:
        continue
    self.vocab = vocab

    self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
    self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}  
    self.register_special_tokens = special_tokens

  def _encode_chunk(self, text_bytes):
    text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
    ids = super()._encode_chunk(text_bytes)
    return ids
  
  def decode(self, ids):
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
    text = text_bytes.decode("utf-8", errors="replace")
    return text
  
  def save(self, file_prefix):
    """
      Saves two files: file_prefix.vocab and file_prefix.model
      This is inspired (but not equivalent to!) sentencepiece's model saving:
      - model file is the critical one, intended for load()
      - vocab file is just a pretty printed version for human inspection only
      ## not suitable to be called, has some issues
    """
    model_file = file_prefix + ".model"
    inverted_merges = {idx: pair for pair, idx in self.merges.items()}
    with open(model_file, 'w') as f:
      f.write("minibpe v1\n")
      f.write(f"{self.pattern}\n")
      f.write(f"{len(self.special_tokens)}\n")
      for special, idx in self.special_tokens.items():
        f.write(f"{special} {idx}\n")
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")

    vocab_file = file_prefix + ".vocab"
    with open(vocab_file, 'w') as f:
      for idx, token in self.vocab.items():
        s = render_token(token)
        if idx in inverted_merges:
          idx0, idx1 = inverted_merges[idx]
          s0 = render_token(self.vocab[idx0])
          s1 = render_token(self.vocab[idx1])
          f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
        else:
          f.write(f"[{s}] {idx}\n")

  def load(self, model_file):
    """
      loads '.model' file
      returns vocab loaded from the file
      ## not suitable to be called, has some issues
    """
    assert model_file.endswith('.model')
    merges = {}
    special_tokens = {}
    idx = 256
    with open(model_file, 'r', encoding='utf-8') as f:
      version = f.readline().strip()
      assert version == "minbpe v1"

      self.pattern = f.readline().strip()
      num_special = int(f.readline().strip())
      for _ in range(num_special):
        special, special_idx = f.readline().strip().split()
        special_tokens[special] = int(special_idx)
      
      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
      
      merges = merges
      special_tokens = special_tokens
      vocab = build_vocab(merges, special_tokens)
    
    return vocab

  def save_vocab(self, vocab_file):
    vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
    for (p0, p1), idx in self.merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]

    inverted_merges = {idx: pair for pair, idx in self.merges.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
      for idx, token in vocab.items():
          s = render_token(token)
          if idx in inverted_merges:
              idx0, idx1 = inverted_merges[idx]
              s0 = render_token(vocab[idx0])
              s1 = render_token(vocab[idx1])
              f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
          else:
              f.write(f"[{s}] {idx}\n")