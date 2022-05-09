import neo4j
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship
import pandas as pd
import dgl
import matplotlib.pyplot as plt
import numpy as np


class edge_list():
    """Class to gather the edge list and create dgl graph"""
    def __init__(self, uri: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=("neo4j", password))

    def close(self) -> None:
        self.driver.close()

    @classmethod
    def edge_list(cls, tx) -> any:
        query = ("""
                    MATCH path=(m)--(n)
                    RETURN m.id AS u, n.id AS v
                """)
        result = tx.run(query)
        #return a dataframe
        return result.data() 

    def get_edge_list(self) -> any:
        result = self.driver.session().write_transaction(self.edge_list)
        return pd.DataFrame(result)

    def dgl_graph_from_cypher(self, data: pd.DataFrame) -> None:
        """
        Takes the whole graph and creates dgl graph

        ARGS:
            data is the edge list in a pandas df
            
        RETURNS: dgl graph
        """
        #THIS IS UNDIRECTED GRAPH
        u = data["u"].to_numpy()
        v = data["v"].to_numpy()
        #Building diagram
        return dgl.graph((u, v))



if __name__ == "__main__":
    hello = edge_list()
    data = hello.dgl_graph_from_cypher(hello.get_edge_list())
    print('We have %d nodes.'% data.number_of_nodes())
    print('We have %d edges.'% data.number_of_edges())
