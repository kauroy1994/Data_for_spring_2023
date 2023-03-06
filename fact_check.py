from random import choice
from tqdm import tqdm

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from fact_checking import FactChecker

'''
reading in the data
'''

import pickle
data_binary = open("concept_paths.pkl",'rb')
concept_paths = pickle.load(data_binary)
concepts = list(concept_paths.keys())[:]

'''
format paths as strings for prompting
'''

string_paths = []
for concept in concepts:
  concept_path=concept_paths[concept]
  if concept_path and concept_path[0] and len(concept_path) > 0:
    if not [x for x in concept_path if '[\'\'' in x]:
      for path in concept_path:
        if path.replace(',','').replace('\'','')[1:-1]:
          string_paths.append(path.replace(',','').replace('\'','')[1:-1])

'''
unit test prompt
'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
fact_checking_model = GPT2LMHeadModel.from_pretrained('fractalego/fact-checking')
fact_checker = FactChecker(fact_checking_model, tokenizer)

true_sums = []

for j in tqdm(range(10)):

  true_sum = 0

  for i in range(100):
    r = choice(string_paths)

    _evidence = "Lets talk about the concept "+r.split(' ')[0]

    _claim = r

    is_claim_true = fact_checker.validate(_evidence, _claim)
    true_sum += int(is_claim_true)

  true_sums.append(true_sum)

print (true_sums)
