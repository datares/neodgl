import networkx as nx
import numpy as np
import pandas as pd

from node2vec import Node2Vec



def embed_node2vec(nx_G: nx.DiGraph) -> np.array:

    #generate object
    node2vec = Node2Vec(nx_G, dimensions=5, walk_length=10, num_walks=50, workers=1)
    
    #generate embeddings
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node2vec_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in nx_G.nodes()], index = nx_G.nodes))

    node2vec_embeddings = node2vec_df.to_dict(orient='split')
    node2vec_embeddings = np.array(node2vec_embeddings["data"])
    return node2vec_embeddings