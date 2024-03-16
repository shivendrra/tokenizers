import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class PerCharTokenizer:
  def __init__(self):
    super().__init__()
    self.vocab_size = 0
    self.string_to_index = {}
    self.index_to_string = {}

  def encode(self, string):
    encoded = []
    for char in string:
      if char in self.string_to_index:
        encoded.append(self.string_to_index[char])
      else:
        special_index = len(self.string_to_index)
        self.string_to_index[char] = special_index
        self.index_to_string[special_index] = char
        encoded.append(special_index)
    return encoded
  
  def decode(self, integer):
    decoded = []
    for i in integer:
      if i in self.index_to_string:
        decoded.append(self.index_to_string[i])
      else:
        continue
    return ''.join(decoded)
  
  def _build_vocab(self):
    return {i: ids for i, ids in enumerate(self.chars)}

  def train(self, train_text):
    self.train_data = train_text
    self.chars = sorted(list(set(self.train_data)))
    self.string_to_index = {ch: i for i, ch in enumerate(self.chars)}
    self.index_to_string = {i: ch for i, ch in enumerate(self.chars)}
    self.vocab = self._build_vocab()
  
  def save_model(self, prefix):
    """
      - basic save_model() funtion, saves one file, '.model'
      - '.model' contains all the final merges, each on next line

      Args:
        prefix (str): prefix along with the path
        self.merges (dict): contains final merges
    """
    file_name = prefix + '.model'
    with open(file_name, 'w', encoding='utf-8') as f:
      f.write("per-char v1\n")
      for idx, ids in self.vocab.items():
        f.write(f"{idx} {ids}\n")
    print("file saved successfully!!")
  
  def load(self, model_path):
    """
      - loads the '.model' file
      - re-writes the merges in the new merges dict
      - builds the vocab again for further use

      Args:
        model_path (str): path to the '.model' file
    """
    assert model_path.endswith('.model')
    vocab = {}
    with open(model_path, 'r', encoding='utf-8') as f:
      version = f.readline().strip()
      assert version == 'per-char v1'
      for line in f:
        tokens = line.split()
        if len(tokens) == 2:
          idx, char = int(tokens[0]), tokens[1]
          vocab[idx] = char
        else:
          continue
    self.vocab = vocab
    self.chars = [char for idx, char in sorted(self.vocab.items())]
    self.vocab_size = len(self.chars)
    self.string_to_index = {char: idx for idx, char in enumerate(self.chars)}
    self.index_to_string = {idx: char for idx, char in enumerate(self.chars)}