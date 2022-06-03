import dgl
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from sklearn.metrics import confusion_matrix

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

    #----------------------------------------------------------
    #ARGUMENTS FOR USING THIS ON DATARES NETWORK

    #how many labels there are
    num_labels = 3

    #note that only the nodes with the id of 0 and 33 (the president and the instructor) are labeled
    #for labeling, make sure that the node ids
    #IN NEO4J THESE ARE CHARACTERIZED BY <id> data field
    prelabeled_nodes = [0, 29, 56]

    #for 4 groups, it would look like [0, 1, 2, 3], where each one is the class
    #here, node 0 is labeled class 0, node 33 is labeled class 1, node 56 is labeled class 2
    classes = [0, 1, 2]

    #MATCH COLORS TO NUMBER OF CLASSES, COLORS CAN BE ANYTHIN (HTMNL COLORS) BUT DIMENSTION MATCH
    #ex) number of labels = 3 -> number of colors = 3
    #choose any color!
    colors = ["#BE6D5C", "#3EFF00", "#1D43F0"]

    #classes that correspond to each true class
    class_rel = {0: "UCLA Athletics", 1: "Research", 2: "Data Blog"}
    #----------------------------------------------------------------------

    print("-----------------------------------------------")

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
        prelabeled_nodes = [0, 33]
        classes = [0, 1]
        colors = ["#BE6D5C", "#3EFF00"]
        num_labels = 2

    elif args.uri and args.password:
        print("\nconnecting to", args.uri)
        hello = edge_list(args.uri, args.password)
        G = hello.dgl_graph_from_cypher(hello.get_edge_list())
        true_labels = hello.get_true_labels()
        hello.close()
        print("connection success\n")
        time.sleep(2)


    nx_G = G.to_networkx().to_undirected()
    print("network data:")
    print('   %d nodes.'% G.number_of_nodes())
    print('   %d edges.'% G.number_of_edges())
    print("prelabeled nodes:", prelabeled_nodes)

    #create embeddings using kamada kawaii
    #pos = nx.kamada_kawai_layout(nx_G)

    #train the model

    
    embed = nn.Embedding(nx_G.number_of_nodes(), 5) # 34 nodes with embedding dim equal to 5
    G.ndata['feat'] = embed.weight

    #create the model
    net = GCN(5, 5, num_labels)
    print('\ngcn:', net)

    inputs = embed.weight
    labeled_nodes = torch.tensor(prelabeled_nodes)
    labels = torch.tensor(classes)  

    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)

    #save the embeddings and output tensor in list
    all_logits = []

    print("\n-----------------------------------------------\n")
    print("model training")
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

    #---------------------------------------------------------
    #Visualize

    fig = plt.figure(dpi=150, figsize=(16,10))
    fig.clf()
    ax = fig.subplots()

    if not os.path.exists("graph_vis"):
        os.mkdir("graph_vis")
    print("\n-----------------------------------------------\n")
    #save into a video
    print("analyzing graph")
    print("writing visualizations ...")


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

    #---------------------------------------
    #accuracy
    #f1 and confusion matrix

    print("\n-----------------------------------------------\n")

    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (16, 10))

    print("analyzing accuracy")

    if not os.path.exists("accuracy_vis"):
        os.mkdir("accuracy_vis")

    print("writing visualizations ... ")

    #confusion matrix for last iteration
    write_confusion_matrix(199, all_logits, nx_G, true_labels, class_rel, ax, cbar_ax)
    plt.savefig("accuracy_vis/confusion_matrix.png")

    ani = animation.FuncAnimation(fig, write_confusion_matrix, fargs = (all_logits, nx_G, true_labels, class_rel, ax, cbar_ax), frames=len(all_logits), interval=200)
    vid_name = "accuracy_vis/confusion_matrix.gif"
    ani.save(vid_name, writer = "Pillow", fps = 80)
    print("saved video:", vid_name)

    print("\naccuracy saved to", "/accuracy_vis/")


