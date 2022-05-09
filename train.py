import dgl
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import itertools
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
import shutil

import seaborn as sns
sns.set(font_scale = 1.5)
sns.set_theme()




from neodgl import edge_list 
from model import *
from utils import *

#Print out the number of nodes and edges in the newly constructed graph
hello = edge_list()
G = hello.dgl_graph_from_cypher(hello.get_edge_list())
nx_G = G.to_networkx().to_undirected()
print('We have %d nodes.'% G.number_of_nodes())
print('We have %d edges.'% G.number_of_edges())

pos = nx.kamada_kawai_layout(nx_G)
#nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
plt.savefig('graph_vis/graph.png')

################################
#train the model

embed = nn.Embedding(34, 5) # 34 nodes with embedding dim equal to 5
G.ndata['feat'] = embed.weight
print(G.ndata['feat'][2])
print(G.ndata['feat'][[10, 11]])


net = GCN(5, 5, 2)
print('net:', net)

inputs = embed.weight
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
all_logits = []
for epoch in range(200):

    logits = net(G, inputs)

    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)

    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch %d | Loss: %.4f'% (epoch, loss.item()))



##############################################
#Visualize

fig = plt.figure(dpi=150, figsize=(16,10))
fig.clf()
ax = fig.subplots()

#prelabeled nodes
nodelist = [0, 33]

if not os.path.exists("graph_vis"):
    os.mkdir("graph_vis")

#save into a video
ani = animation.FuncAnimation(fig, draw, fargs = (all_logits, nodelist, nx_G, ax), frames=len(all_logits), interval=200)
vid_name = "graph_vis/graph.gif"
ani.save(vid_name, writer = "Pillow", fps = 80)
print("Saved Video:", vid_name)

#draw the graphs
draw(0, all_logits, nodelist, nx_G, ax)
plt.savefig("graph_vis/Initial_Embedding.png")

draw(199, all_logits, nodelist, nx_G, ax)
plt.savefig("graph_vis/Final_Iteration.png")
print("Visualizations saved to", "/graph_vis")

hello.close()