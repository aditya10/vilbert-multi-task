import numpy as np
import _pickle as cPickle
import torch

datapath = "../datasets/refcoco/cache/"
dataset = 'refcocog'
split = 'train'
setinfo = '_20_101_cleaned'

entries = cPickle.load(open(datapath+dataset+'_'+split+setinfo+'.pkl', "rb"))
entries_graph = cPickle.load(open(datapath+dataset+'_'+split+setinfo+'_graph'+'.pkl', "rb"))
entries_graph_child = cPickle.load(open(datapath+dataset+'_'+split+setinfo+'_graph_child'+'.pkl', "rb"))
entries_graph_child_symm = cPickle.load(open(datapath+dataset+'_'+split+setinfo+'_graph_child_symm'+'.pkl', "rb"))
entries_graph_ancestor = cPickle.load(open(datapath+dataset+'_'+split+setinfo+'_graph_ancestor'+'.pkl', "rb"))
entries_graph_ancestor_symm = cPickle.load(open(datapath+dataset+'_'+split+setinfo+'_graph_ancestor_symm'+'.pkl', "rb"))

num_entries = len(entries)

# for i, entry in enumerate(entries):

#     graph_data = {}

#     graph_data['graph'] = {}
#     graph_data['graph']['adj1'] = entries_graph[i]['adj1']
#     graph_data['graph']['adj2'] = entries_graph[i]['adj2']

#     graph_data['graph_child'] = {}
#     graph_data['graph_child']['adj1'] = entries_graph_child[i]['adj1']
#     graph_data['graph_child']['adj2'] = entries_graph_child[i]['adj2']

#     graph_data['graph_child_symm'] = {}
#     graph_data['graph_child_symm']['adj1'] = entries_graph_child_symm[i]['adj1']
#     graph_data['graph_child_symm']['adj2'] = entries_graph_child_symm[i]['adj2']

#     graph_data['graph_ancestor'] = {}
#     graph_data['graph_ancestor']['adj1'] = entries_graph_ancestor[i]['adj1']
#     graph_data['graph_ancestor']['adj2'] = entries_graph_ancestor[i]['adj2']

#     graph_data['graph_ancestor_symm'] = {}
#     graph_data['graph_ancestor_symm']['adj1'] = entries_graph_ancestor_symm[i]['adj1']
#     graph_data['graph_ancestor_symm']['adj2'] = entries_graph_ancestor_symm[i]['adj2']

#     entry['graph_data'] = graph_data
    
#     if (i % 1000) == 0:
#       print(str(i)+' of '+str(num_entries))

# for i, entry in enumerate(entries):

#     entry['graph_adj1'] = entries_graph[i]['adj1']
#     entry['graph_adj2'] = entries_graph[i]['adj2']

#     entry['graph_child_adj1'] = entries_graph_child[i]['adj1']
#     entry['graph_child_adj2'] = entries_graph_child[i]['adj2']

#     entry['graph_child_symm_adj1'] = entries_graph_child_symm[i]['adj1']
#     entry['graph_child_symm_adj2'] = entries_graph_child_symm[i]['adj2']

#     entry['graph_ancestor_adj1'] = entries_graph_ancestor[i]['adj1']
#     entry['graph_ancestor_adj2'] = entries_graph_ancestor[i]['adj2']

#     entry['graph_ancestor_symm_adj1'] = entries_graph_ancestor_symm[i]['adj1']
#     entry['graph_ancestor_symm_adj2'] = entries_graph_ancestor_symm[i]['adj2']
    
#     if (i % 1000) == 0:
#       print(str(i)+' of '+str(num_entries))


for i, entry in enumerate(entries):


    adj1 = [entries_graph[i]['adj1'], 
            entries_graph_child[i]['adj1'], 
            entries_graph_child_symm[i]['adj1'],
            entries_graph_ancestor[i]['adj1'],
            entries_graph_ancestor_symm[i]['adj1']]
    
    adj2 = [entries_graph[i]['adj2'], 
            entries_graph_child[i]['adj2'], 
            entries_graph_child_symm[i]['adj2'],
            entries_graph_ancestor[i]['adj2'],
            entries_graph_ancestor_symm[i]['adj2']]

    entry['graph_adj1'] = torch.stack(adj1)
    entry['graph_adj2'] = torch.stack(adj2)
    
    if (i % 1000) == 0:
      print(str(i)+' of '+str(num_entries))

cPickle.dump(entries, open(datapath+dataset+'_'+split+setinfo+'_graph_data.pkl', "wb"))