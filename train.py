from warnings import filterwarnings
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
import argparse
import time

import seaborn as sns
sns.set(font_scale = 1.5)
sns.set_theme()

from neodgl import edge_list 
from model import *
from utils import *

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the GCN model using either the test data or neo4j dbms")
    parser.add_argument("--test", default=False, action="store_true", help = "train the test data!")
    parser.add_argument("--uri", type=str, help="neo4j dbms uri", default="")
    parser.add_argument("--password", type=str, help="neo4j dbms password", default="")
    args = parser.parse_args()


    if not args.test and (not args.uri or not args.password):
        print("\nno testing argument given")
        print("missing args: uri or password")
        print("terminating process: invalid args")
        quit()

    elif args.test:
        print("\nTraining with Test Data: Zach's Karate Club\n")
        G = build_karate_club_graph()
        time.sleep(2)

    elif args.uri and args.password:
        print("\nconnecting to", args.uri)
        hello = edge_list(args.uri, args.password)
        G = hello.dgl_graph_from_cypher(hello.get_edge_list())
        hello.close()
        print("connection success\n")
        time.sleep(2)

    #TODO: ADD ARGsS

    #how many labels there are
    num_labels = 3

    #note that only the nodes with the id of 0 and 33 (the president and the instructor) are labeled
    #for labeling, make sure that the node ids (or names, etc.) are unique
    prelabeled_nodes = [0, 33, 56]

    #here, we are giving the label 0 to the node 0 and 1 to the node 33
    #for 4 groups, it would look like [0, 1, 2, 3], where each one is the class
    classes = [0, 1, 2]

    #MATCH COLORS TO NUMBER OF CLASSES, COLORS CAN BE ANYTHIN (HTMNL COLORS) BUT DIMENSTION MATCH
    #ex) number of labels = 3 -> number of colors = 3
    colors = ["#BE6D5C", "#3EFF00", "#1D43F0"]

    nx_G = G.to_networkx().to_undirected()
    print("network data:")
    print('   %d nodes.'% G.number_of_nodes())
    print('   %d edges.\n'% G.number_of_edges())

    #create embeddings using kamada kawaii
    pos = nx.kamada_kawai_layout(nx_G)

    #train the model

    embed = nn.Embedding(nx_G.number_of_nodes(), 5) # 34 nodes with embedding dim equal to 5
    G.ndata['feat'] = embed.weight

    #create the model
    net = GCN(5, 5, num_labels)
    print('gcn:', net)

    inputs = embed.weight
    labeled_nodes = torch.tensor(prelabeled_nodes)
    labels = torch.tensor(classes)  

    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)

    #save the embeddings and output tensor in list
    all_logits = []

    print("\nmodel training")
    time.sleep(2)
    #start the training
    for epoch in range(200):

        logits = net(G, inputs)

        #save the output
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)

        # we only compute loss for labeled nodes
        loss = F.nll_loss(logp[labeled_nodes], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f'% (epoch, loss.item()))

    print("success")
    #Visualize

    fig = plt.figure(dpi=150, figsize=(16,10))
    fig.clf()
    ax = fig.subplots()

    if not os.path.exists("graph_vis"):
        os.mkdir("graph_vis")

    #save into a video
    print("\nwriting visualizations ...")


    if len(colors) > 2:
        print("number of labels is greater than 2, applying dimension reduction")
    print("colors for each label:", colors)

    ani = animation.FuncAnimation(fig, draw, fargs = (all_logits, prelabeled_nodes, nx_G, ax, colors), frames=len(all_logits), interval=200)
    vid_name = "graph_vis/graph.gif"
    ani.save(vid_name, writer = "Pillow", fps = 80)
    print("saved video:", vid_name)

    #draw the graphs
    draw(0, all_logits, prelabeled_nodes, nx_G, ax, colors)
    plt.savefig("graph_vis/Initial_Embedding.png")

    draw(199, all_logits, prelabeled_nodes, nx_G, ax, colors)
    plt.savefig("graph_vis/Final_Iteration.png")
    print("visualizations saved to", "/graph_vis")



