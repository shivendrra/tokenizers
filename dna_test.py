import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from subDNA import DNAtokenizer

with open('train files/train_data.txt', 'r', encoding='utf-8') as f:
  data = f.read()

token = DNAtokenizer()
token.train(data, 1000)
token.save_model(model_prefix='subDNA/trained models/base_1k')
# out_vocab, merges = token.load_model(model_path='dna_200.model')

# sample = 'CCTCCTGCCTGGAACATCAGGCTCCATGTTCTTTGGCTTTTAGAC'
with open('train files/new_dna_1.txt', 'r', encoding='utf-8') as f:
  sample = f.read()
# print(token.encode(sample))
print(sample == token.decode(token.encode(sample)))
print(f"compression ration: {(len(sample) / len(token.encode(sample))):.2f}x")