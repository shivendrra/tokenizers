import timeit
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from subDNA import DNAtokenizer

start = timeit.default_timer()
# with open('train files/train_data.txt', 'r', encoding='utf-8') as f:
with open('train files/new_dna_1.txt', 'r', encoding='utf-8') as f:
  data = f.read()

start_token = timeit.default_timer()
print(f"file opened in {((start_token - start)/60):.2f} mins")

token = DNAtokenizer()
token.train(data, 100)
token.load_model(model_path='subDNA/trained models/base_1k.model')
token.continue_train(data, 200)
token.save_model(model_prefix='subDNA/trained models/base_2k')
end_token = timeit.default_timer()
print(f"tokenized in {((end_token - start_token)/3600):.2f} hrs")

sample = 'CCTCCTGCCTGGAACATCAGGCTCCATGTTCTTTGGCTTTTAGAC'
# with open('train files/new_dna_1.txt', 'r', encoding='utf-8') as f:
#   sample = f.read()

# print(token.encode(sample))
print(sample == token.decode(token.encode(sample)))
print(f"compression ration: {(len(sample) / len(token.encode(sample))):.2f}x")