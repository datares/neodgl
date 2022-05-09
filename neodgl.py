from neo4j import GraphDatabase
from neo4j.types.graph import Node, Relationship
import pandas as pd
import networkx as nx

def graph_from_cypher(data):
    """Constructs a networkx graph from the results of a neo4j cypher query.
    Example of use:
    >>> result = session.run(query)
    >>> G = graph_from_cypher(result.data())

    Nodes have fields 'labels' (frozenset) and 'properties' (dicts). Node IDs correspond to the neo4j graph.
    Edges have fields 'type_' (string) denoting the type of relation, and 'properties' (dict)."""

    G = nx.Graph()
    def add_node(node):
        # Adds node id it hasn't already been added
        u = node.id
        if G.has_node(u):
            return
        G.add_node(u, labels=node._labels, properties=dict(node))

    def add_edge(relation):
        # Adds edge if it hasn't already been added.
        # Make sure the nodes at both ends are created
        for node in (relation.start_node, relation.end_node):
            add_node(node)
        # Check if edge already exists
        u = relation.start_node.id
        v = relation.end_node.id
        eid = relation.id
        if G.has_edge(u, v, key=eid):
            return
        # If not, create it
        G.add_edge(u, v, key=eid, type_=relation.type, properties=dict(relation))

    for d in data:
        for entry in d.values():
            # Parse node
            if isinstance(entry, Node):
                add_node(entry)

            # Parse link
            elif isinstance(entry, Relationship):
                add_edge(entry)
            else:
                raise TypeError("Unrecognized object")
    return G

driver = GraphDatabase.driver('bolt://localhost:7687', auth=("neo4j", "hunter2"))

query = """
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE toLower(m.title) CONTAINS "you"
RETURN *
"""

with driver.session() as session:
    result = session.run(query)

# This works
G = graph_from_cypher(result.data())