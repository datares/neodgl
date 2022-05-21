import pandas as pd
import numpy as np
import neo4j

from tqdm import tqdm
from neo4j import GraphDatabase



def create_constraint(driver: neo4j.Driver) -> None:
    print("creating uniqueness constraint")
    try:
        name_query = "CREATE CONSTRAINT ON (p:Person) ASSERT p.name IS UNIQUE"       

        info = driver.session().run(name_query)
        response = driver.session().run("CALL db.constraints").data()
        print(response)

    except:
        name_query = "CALL db.constraints;"
        info = driver.session().run(name_query)
        response = driver.session().run("CALL db.constraints").data()
        print("uniqueness constraint already exists:", response)
    print("success\n")

def data_preprocessing():
    data = pd.read_excel("form_responses.xlsx")
    data.fillna("No")
    return data

def create_nodes(tx, name, team) -> None:
    """
    parameters of create_nodes are metadata for nodes
    """
    query = """
            MERGE (p:Person {name: $name, team: $team})
            """
    tx.run(query, name = name, team = team)

def upload_nodes(data: pd.DataFrame, driver: neo4j.Driver) -> [str]:
    responses = []

    for i in tqdm(range(len(data)), desc = "deploying person nodes"):   
        name = data["First and Last Name:"][i]
        team = data["Your DataRes Team for S22"][i]
        name = name.lower()
        responses.append(name)
        driver.session().write_transaction(create_nodes, name, team)

    print("success\n", flush=True)
    return responses

def create_relationships(tx, name1, name2) -> None:
    """
    Args:
        id1 is the id of first node
        id2 is id of second 
        NOTE: SRC-->DEST
    """
    query = """
            MATCH (p:Person {name: $name1})
            MATCH (n:Person {name: $name2})
            MERGE (p)-[:KNOWS]-(n)
            """
    tx.run(query, name1 = name1, name2 = name2)

def upload_relationship(data: pd.DataFrame, responses: [str], driver: neo4j.Driver) -> None:

    #lowercase the columns
    data.columns = data.columns.str.lower()

    #change colin and madison to the headings
    new_headings = data.columns.to_list()
    new_headings[5] = "madison kohls"
    new_headings[6] = 'colin curtis'
    data.columns = new_headings

    #names as the index
    data.index = data["first and last name:"]
    data.index = data.index.str.lower()

    #filter out data
    wrong_names = ['ziyue jin', 'tian ouyang', 'natalia f', 'shiyu murashima', 'trent bellinger', 'christine shen ', 'daniel fsng', 'kowoon jeong ']
    new_headings = [item for item in responses if item not in wrong_names]
    data = data[new_headings]

    #filter out colin and mads data
    data["colin curtis"] = data["colin curtis"].replace(to_replace = "literally a genius", value = "Yes")
    data["madison kohls"] = data["madison kohls"].replace(to_replace = "yup super cool", value = "Yes")

    for column_name in tqdm(data.columns, desc = "deploying relationships"):
        for i in data.index:
            if data.loc[i, column_name] == "Yes":
                if i != column_name:
                    driver.session().write_transaction(create_relationships, i, column_name)


    print("success\n")



def filter_data(driver):
    print("filtering nodes")
    query = """
            MATCH (n)
            WHERE size((n)--())=0
            DELETE (n)
            RETURN COUNT(n)
            """
    info = driver.session().run(query)
    info = info.data()[0]["COUNT(n)"]

    if info == 0:
        print("no nodes deleted: fully connected")
    
    else:
        print("deleted nodes:", info)


def complete(driver):
    print("\ngraph creation completed")
    query = """
            MATCH (n)
            RETURN COUNT(n) as nodes
            """
    info = driver.session().run(query)
    info = info.data()[0]["nodes"]
    print("node count", info)

    query = """
            MATCH ()-[r]->() RETURN count(*) as count
            """
    info = driver.session().run(query)
    info = info.data()[0]["count"]
    print("relationship count", info)


    