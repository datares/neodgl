<img src="https://ucladatares.com/static/media/logo.416d2c1d.svg" width="9%"></img>  &ensp; ![GitHub](https://img.shields.io/github/license/datares/neodgl) ![GitHub last commit](https://img.shields.io/github/last-commit/datares/neodgl) ![Own Badge](https://img.shields.io/badge/Research%20Head-Irsyad%20%3A\)\)\)-blue) ![Own Badge](https://img.shields.io/badge/dependencies-7-brightgreen)  



# Seemless integration of DGL on Neo4j DBMS
By integrating the DGL framework with the Neo4j DBMS, analysis of network data is more wholistic, combining the visualization prowess and "classical" graph algorithms of Neo4j and the end-to-end deep learning models of DGL. 

Overall, this repository not only solves both Neo4j and DGL's shortcomings, but also provides visualizations of model training for enhanced understanding of GCN/GNNs. 

### Setup
-------
Create new conda env:

<code> conda create -n dgl </code>

Switch to conda env and install dependencies:

<code> conda activate dgl </code>

<code> git clone git@github.com:datares/neodgl.git</code>

<code> pip install -r requirements.txt </code>

Start Training with Zach's Karate Club Test Data:

<code> python3 train.py --test</code>


To use your own database:

 <code> python3 train.py --uri [DBMS URI] --password [DBMS PASSWORD]</code>

Arguments: 

```
usage: train.py [-h] [--test] [--uri URI] [--password PASSWORD]

Train the GCN model using either the test data or neo4j dbms

optional arguments:
  -h, --help           show this help message and exit
  --test               train the test data!
  --uri URI            neo4j dbms uri
  --password PASSWORD  neo4j dbms password
```

### Visualizations
---------------
Semi-Supervised Node Classification Model Training: 
![](graph_vis/graph.gif)




### NOTE: UNFINISHED REPOS
