![GitHub](https://img.shields.io/github/license/datares/neodgl) ![GitHub last commit](https://img.shields.io/github/last-commit/datares/neodgl)

# Seemless integration of DGL on Neo4j DBMS
---------------
By integrating the DGL framework with the Neo4j DBMS, analysis of network data is more wholistic, combining the visualization prowess and "classical" graph algorithms of Neo4j and the end-to-end deep learning models of DGL. 

Overall, this repository not only solves both Neo4j and DGL's shortcomings, but also provides visualizations of model training for enhanced understanding of GCN/GNNs. 

### Setup
-------------------

Create Conda Env and install dependencies:

<code> conda activate dgl </code>

<code> pip install -r requirements.txt </code>

Start Training:

<code> python3 train.py </code>


To use your own database, edit <code> \_\_init__ </code> in <code> neodgl.py </code>:

 <code> self.driver = GraphDatabase.driver("YOUR OWN URI", auth=("neo4j", "PASSWORD"))
</code>

### NOTE: UNFINISHED REPOS
