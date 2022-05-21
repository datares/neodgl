import pandas as pd
import numpy as np
import neo4j

from tqdm import tqdm
from neo4j import GraphDatabase

from graph_create_utils import *

import argparse
import time
import warnings




if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Create a neo4j graph from DataRes Data")
    parser.add_argument("--uri", type=str, help="neo4j dbms uri", default="")
    parser.add_argument("--password", type=str, help="neo4j dbms password", default="")
    args = parser.parse_args()

    if not args.uri and not args.password:
        print("invalid parameters")
        print("terminating")
        quit()

    # Seen from :server status
    uri = args.uri

    password = args.password
    # default user for graph database is neo4j
    # auth = ("neo4j", "password")
    auth = ("neo4j", password)

    print("starting datares graph create \n")

    driver = GraphDatabase.driver(uri = uri, auth = auth)
    print("connecting:", driver.verify_connectivity())

    #create constraint
    create_constraint(driver)

    data = data_preprocessing()

    #upload nodes
    responses = upload_nodes(data, driver)

    #upload relationships
    upload_relationship(data, responses, driver)

    #delete nodes
    filter_data(driver)

    #get stats
    complete(driver)
    




