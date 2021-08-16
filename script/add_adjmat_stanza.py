# This file can only be run in the allennlp environment.
# run: conda activate stanza

import numpy as np
import stanza
import spacy_stanza
import _pickle as cPickle
import torch
import torch.nn.functional as F
import re
import argparse
from pytorch_transformers.tokenization_bert import BertTokenizer
import os

print('Loading model...')
stanza.download("en")
nlp = spacy_stanza.load_pipeline("en")

print('Loading tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def create_map_to_index(sent):
  words = tokenizer.tokenize(sent)
  setwords = []
  for i, word in enumerate(words):
    if word.startswith('##'):
      setwords[-1] = setwords[-1]+word[2:]
    else:
      setwords.append(word)
  curridx = 0
  indexmap = {}
  for i, word in enumerate(setwords):
    tokens_len = len(tokenizer.encode(word))
    indexmap[i] = [i for i in range(curridx, curridx+tokens_len)]
    curridx+=tokens_len
  return indexmap, curridx, setwords

def get_true_span(span_map, span):
  ret = []
  for idx in span:
    ret+=(span_map[idx])
  return ret

def graph_gen(sent, type='child', symm=False):

  spanmap, l, words = create_map_to_index(sent)

  doc = nlp(sent)

  if len(doc) == 0 or len(words) != len(doc):
    adj1 = np.ones((l,l), dtype='int')
    adj2 = np.ones((l,l), dtype='int')
    return adj1, adj2, 1

  adj1 = np.zeros((l,l), dtype='int')
  adj2 = np.zeros((l,l), dtype='int')
  np.fill_diagonal(adj1, 1)
  np.fill_diagonal(adj2, 1)
  
  graph = {}

  for token in doc:
    
    if type == 'child':
      children = [child for child in token.children]
    elif type == 'ancestor':
      children = [token.head]
    else:
      children = []
    
    true_token_idxs = spanmap[token.i]

    span = []
    
    for child in children:
      span += spanmap[child.i]

    for idx in true_token_idxs:
      graph[idx] = span

  for i in graph.keys():
    for j in graph[i]:
      adj1[i][j] = 1
      adj2[i][j] = 1
      for k in graph[j]:
        adj2[i][k] = 1

  if symm:
    adj1 = np.tril(adj1) + np.triu(adj1.T, 1)
    adj2 = np.tril(adj2) + np.triu(adj2.T, 1)

  return adj1, adj2, 0

def main(datapath, type='child', symm=False):
  entries = cPickle.load(open(datapath+'.pkl', "rb"))
  max_len = len(entries[0]['input_mask'])
  print("max dimention of adj matrix: "+str(max_len))
  num_entries = len(entries)
  issues_count = 0
  issues_count_t = 0

  for i, entry in enumerate(entries):

    adj1, adj2, isc = graph_gen(entry['caption'], type, symm)
    adj1 = adj1[:max_len-2, :max_len-2]
    adj2 = adj2[:max_len-2, :max_len-2]
    adj1 = torch.from_numpy(adj1)
    adj2 = torch.from_numpy(adj2)
    issues_count_t += isc

    a1 = F.pad(input=adj1, pad=(1, 1, 1, 1), mode='constant', value=1)
    a2 = F.pad(input=adj2, pad=(1, 1, 1, 1), mode='constant', value=1)

    tokens_len = len([i for i in entry['input_mask'] if i != 0])
    if a1.shape[0] != tokens_len:
      print(entry)
      print(tokens_len)
      print(a1.shape[0])
      exit()
      issues_count += 1
      a1 = torch.ones((tokens_len, tokens_len), dtype=int)
      a2 = torch.ones((tokens_len, tokens_len), dtype=int)

    if a1.shape[0] < max_len:
      s = max_len-a1.shape[0]
      a1 = F.pad(input=a1, pad=(0, s, 0, s), mode='constant', value=0)
      a2 = F.pad(input=a2, pad=(0, s, 0, s), mode='constant', value=0)
    entry['adj1'] = a1
    entry['adj2'] = a2

    if (i % 1000) == 0:
      print(str(i)+' of '+str(num_entries))

  if symm:
    sym = "_symm"
  else:
    sym = ""
  
  print("saving to pkl...")
  cPickle.dump(entries, open(datapath+'_graph'+'_'+type+sym+'.pkl', "wb"))
  print("Issues with mismatched tokenization lengths: "+str(issues_count_t))
  print("Issues with wrong sized adj matrix formed: "+str(issues_count))
  print("Done!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--f", default=None, type=str, help="Path to pickle file"
  )
  parser.add_argument(
      "--type", default=None, type=str, help="Path to pickle file"
  )
  parser.add_argument(
      "--symm", action="store_true", help="Produce symmetric matrix"
  )
  parser.add_argument(
      "--dataset", default='refcocog', type=str, help="Dataset name"
  )
  args = parser.parse_args()
  f = args.f
  t = args.type
  s = args.symm
  
  paths = []
  types = []
  
  if f is not None:
    paths.append(f)
  else:
    for fname in os.listdir("../datasets/refcoco/cache"):
      if fname.endswith("cleaned.pkl") and fname.startswith(args.dataset):
        paths.append(os.path.join("../datasets/refcoco/cache", fname))

  if t is not None:
    types.append(t)
  else:
    types+=['child', 'ancestor']

  for path in paths:
    if path.endswith('.pkl'):
      path = path[:len(path)-4]
    for ty in types:
      print("Adding adj matrices to: "+path)
      main(path, ty, s)