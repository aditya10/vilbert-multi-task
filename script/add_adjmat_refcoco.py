# This file can only be run in the allennlp environment.
# run: conda activate allennlp

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import spacy
import numpy as np
import _pickle as cPickle
import torch
import torch.nn.functional as F
import re
import argparse
from pytorch_transformers.tokenization_bert import BertTokenizer
import os

print('Loading SRL model...')
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", cuda_device=0)
nlp = spacy.load("en_core_web_sm")

print('Loading tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def create_role_graph_data(srl_data):
  words = srl_data['words']
  verb_items = srl_data['verbs']
    
  graph_nodes = {}
  graph_edges = []
    
  root_name = 'ROOT'
  graph_nodes[root_name] = {'words': words, 'spans': list(range(0, len(words))), 'role': 'ROOT'}
    
  # parse all verb_items
  phrase_items = []
  for i, verb_item in enumerate(verb_items):
    tags = verb_item['tags']
    tag2idxs = {}
    tagname_counter = {} # multiple args of the same role
    for t, tag in enumerate(tags):
      if tag == 'O':
        continue
      if t > 0 and tag[0] != 'B':
        # deal with some parsing mistakes, e.g. (B-ARG0, O-ARG1)
        # change it into (B-ARG0, B-ARG1)
        if tag[2:] != tags[t-1][2:]:
          tag = 'B' + tag[1:]
      tagname = tag[2:]
      if tag[0] == 'B':
        if tagname not in tagname_counter:
          tagname_counter[tagname] = 1
        else:
          tagname_counter[tagname] += 1
      new_tagname = '%s:%d'%(tagname, tagname_counter[tagname])
      tag2idxs.setdefault(new_tagname, [])
      tag2idxs[new_tagname].append(t)
    if len(tagname_counter) > 1 and 'V' in tagname_counter and tagname_counter['V'] == 1:
      phrase_items.append(tag2idxs)

  node_idx = 1
  spanrole2nodename = {}
  for i, phrase_item in enumerate(phrase_items):
    # add verb node to graph
    tagname = 'V:1'
    role = 'V'
    spans = phrase_item[tagname]
    spanrole = '-'.join([str(x) for x in spans] + [role])
    if spanrole in spanrole2nodename:
      continue
    node_name = str(node_idx)
    tag_words = [words[idx] for idx in spans]
    graph_nodes[node_name] = {
      'role': role, 'spans': spans, 'words': tag_words,
    }
    spanrole2nodename[spanrole] = node_name
    verb_node_name = node_name
    node_idx += 1
    
    # add arg nodes and edges of the verb node
    for tagname, spans in phrase_item.items():
      role = tagname.split(':')[0]
      if role != 'V':
        spanrole = '-'.join([str(x) for x in spans] + [role])
        if spanrole in spanrole2nodename:
          node_name = str(spanrole2nodename[spanrole])
        else:
          # add new node or duplicate a node with a different role
          node_name = str(node_idx)
          tag_words = [words[idx] for idx in spans]
          graph_nodes[node_name] = {
            'role': role, 'spans': spans, 'words': tag_words,
          }
          spanrole2nodename[spanrole] = node_name
          node_idx += 1
        # add edge
        graph_edges.append((verb_node_name, node_name, role))
            
  return graph_nodes, graph_edges

def create_spacy(sent, graph):
  nodes, edges = graph
  node_idx = len(nodes)
                        
  # add noun and verb word node if no noun and no noun phrases
  if len(nodes) == 1:
    sent = re.sub(' +', ' ', sent)
    doc = nlp(sent)
    assert len(doc) == len(nodes['ROOT']['words']), sent
    
    # add noun nodes
    for w in doc.noun_chunks:
      node_name = str(node_idx)
      nodes[node_name] = {
        'role': 'NOUN', 'spans': np.arange(w.start, w.end).tolist()
      }
      nodes[node_name]['words'] = [doc[j].text for j in nodes[node_name]['spans']]
      node_idx += 1
    if len(nodes) == 1:
      for w in doc:
        node_name = str(node_idx)
        if w.tag_.startswith('NN'):
          nodes[node_name] = {
            'role': 'NOUN', 'spans': [w.i], 'words': [w.text],
          }
          node_idx += 1
    
    # add verb nodes
    for w in doc:
      node_name = str(node_idx)
      if w.tag_.startswith('VB'):
        nodes[node_name] = {
          'role': 'V', 'spans': [w.i], 'words': [w.text],
        }
        node_idx += 1
    
  return nodes, edges

def convert_edges(e):
  edges = {}
  for t in e:
    fr, to, _ = t
    if fr in edges:
      edges[fr].append(to)
    else:
      edges[fr] = []
      edges[fr].append(to)
  return edges

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

def graphgen(n,e,span_map,l):

  nodegraphs = {}
  edgegraphs = {}

  n.pop('ROOT', None)

  if len(n) == 0:
    adj1 = np.ones((l,l), dtype='int')
    adj2 = np.ones((l,l), dtype='int')
    return adj1, adj2

  for key in n.keys():

    # 1. Generate a matrix for the node
    graph = np.zeros((l,l), dtype='int')
    
    spans = set(n[key]['spans'])
    true_span = get_true_span(span_map, spans)

    for i in true_span:
      for j in true_span:
        graph[i][j] = 1
    
    nodegraphs[key] = graph

    # 2. If the matrix is verb, then make a secondary edge graph matrix
    if key in e.keys():
      
      graph = np.zeros((l,l), dtype='int')

      for node in e[key]:
        spans.update(n[node]['spans'])
    
      true_span = get_true_span(span_map, spans)
      
      for i in true_span:
        for j in true_span:
          graph[i][j] = 1
      
      edgegraphs[key] = graph

  adj1 = np.zeros((l,l), dtype='int')
  np.fill_diagonal(adj1, 1)
  adj2 = np.zeros((l,l), dtype='int')
  np.fill_diagonal(adj2, 1)
  
  for key in n.keys():
    adj1 += nodegraphs[key]
    if key in e.keys():
      adj2 += edgegraphs[key]
  
  adj1[adj1 > 1] = 1
  adj2[adj2 > 1] = 1

  if len(e) == 0:
    adj2 = np.ones((l,l), dtype='int')

  return adj1, adj2

def get_graph(sent):
  data = predictor.predict(sentence=sent)
  spanmap, l, words = create_map_to_index(sent)
  if len(words)<len(data["words"]):
    adj1 = np.ones((l,l), dtype='int')
    adj2 = np.ones((l,l), dtype='int')
    return adj1, adj2, 1
  n, e = create_spacy(sent, create_role_graph_data(data))
  e = convert_edges(e)
  adj1, adj2 = graphgen(n,e,spanmap,l)
  return adj1, adj2, 0

def main(datapath):
  entries = cPickle.load(open(datapath+'.pkl', "rb"))
  max_len = len(entries[0]['input_mask'])
  print("max dimention of adj matrix: "+str(max_len))
  num_entries = len(entries)
  issues_count = 0
  issues_count_t = 0

  for i, entry in enumerate(entries):

    adj1, adj2, isc = get_graph(entry['caption'])
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

  print("saving to pkl...")
  cPickle.dump(entries, open(datapath+'_graph.pkl', "wb"))
  print("Issues with mismatched tokenization lengths: "+str(issues_count_t))
  print("Issues with wrong sized adj matrix formed: "+str(issues_count))
  print("Done!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--f", default=None, type=str, help="Path to pickle file"
  )
  args = parser.parse_args()
  f = args.f
  
  paths = []
  
  if f is not None:
    paths.append(f)
  else:
    for fname in os.listdir("/ubc/cs/research/shield/projects/aditya10/vilbert-multi-task/datasets/refcoco/cache"):
      if fname.endswith("cleaned.pkl"):
        paths.append(os.path.join("/ubc/cs/research/shield/projects/aditya10/vilbert-multi-task/datasets/refcoco/cache", fname))

  for path in paths:
    if path.endswith('.pkl'):
      path = path[:len(path)-4]
    print("Adding adj matrices to: "+path)
    main(path)