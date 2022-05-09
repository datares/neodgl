import dgl
import numpy as np
import networkx as nx
import torch

import seaborn as sns
sns.set(font_scale = 1.5)
sns.set_theme()



def build_karate_club_graph():
    '''
    All 78 edges are stored in two numpy arrays, one for the source endpoint and the other for the target endpoint
    '''
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    #Edges are directional in DGL; make them bidirectional
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    #Building diagram
    return dgl.DGLGraph((u, v))



"""
logits: [ [[embeddings], [probability tensor]] x 200 epochs]
"""
#iterate through each epoch
def draw(iteration: int, all_logits: torch.tensor, nodelist: [int], nx_G, ax) -> None:
    """
    draws graphs from epoch iteration, given the node list of prelabeled nodes

    ARGS:
        all_logits: embeddings generated by model
        iteration: the iteration number for drawing
        nodelist: the prelabeled nodes from the training step
    
    """
    #HTML emcoding of nodes
    cls1 ="#BE6D5C"
    cls2 ="#3EFF00"

    #save the embedding
    embed = {}
    colors = []

    #for each epoch, 34 vectors for each node, iterate through the vectors
    for j in range(34):
        embed[j] = all_logits[iteration][j].numpy()
        cls = embed[j].argmax()
        colors.append(cls1 if cls else cls2)

    #clear from previous graph
    ax.cla()

    #edit the plot
    ax.set_title('Epoch: %d'% iteration, fontdict = {"fontsize": 8})

    #draw the nodes classified from the ML
    nx.draw_networkx(nx_G.to_undirected(), embed, node_color=colors,
            with_labels=False, node_size=20, ax=ax, width = 0.03, edgecolors="grey")

    #draw the prelabeled nodes
    nx.draw_networkx_nodes(nx_G.to_undirected(), embed, nodelist = nodelist, node_color="black", node_size=50, ax=ax)

    ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True) 
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    #ax.legend()



if __name__ == "__main__":
    G = build_karate_club_graph()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    #Visualize the graph by converting it into a networkx graph
    nx_G = G.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.savefig('graph.png')