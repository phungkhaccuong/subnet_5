import os

import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

index_name = "test_vectors"


def init_index(search_client):
    """Initialize Eth Denver index in Elasticsearch"""

    if not search_client.indices.exists(index=index_name):
        print("creating index...", index_name)
        search_client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "price": {
                            "type": "integer"
                        },
                        "title_vector": {
                            "type": "dense_vector",
                            "dims": 3,
                            "index": False
                        }
                    }
                }
            },
        )
        print("Index created: ", index_name)
    else:
        print("Index already exists: ", index_name)


def insert_data(search_client):
    print('Insert data')
    # Index some vectors
    vectors = [
        {"title_vector": [2.2, 4.3, 1.8], "price": 23},
        {"title_vector": [3.1, 0.7, 8.2], "price": 9},
        {"title_vector": [1.4, 5.6, 3.9], "price": 124},
        {"title_vector": [1.1, 4.4, 2.9], "price": 1457}
    ]

    for i, vector in enumerate(vectors):
        search_client.index(index=index_name, id=i, body=vector)


def query(search_client):
    print('search query')
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.queryVector, 'title_vector') + 1.0",
                "params": {
                    "queryVector": [2.2, 4.3, 1.8]
                }
            }
        }
    }

    search_results = search_client.search(index=index_name, body={"query": script_query})
    print(f"RESULT:{search_results}")

def drop_index(search_client):
    if search_client.indices.exists(index=index_name):
        search_client.indices.delete(index=index_name)
        print("Index deleted: ", index_name)
    else:
        print("Index does not exist: ", index_name)



def search(search_client):
    query = {
        "query": {
            "match_all": {}
        }
    }

    print(f"es_query: {query}")

    try:
        response = search_client.search(
            index=index_name,
            body=query
        )
        print(f"RESPONSE::::{response}")

    except Exception as e:
        print("recall error...", e)

if __name__ == '__main__':
    load_dotenv()

    search_client = Elasticsearch(
        os.environ["ELASTICSEARCH_HOST"],
        basic_auth=(
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )

    drop_index(search_client)
    # # create index
    init_index(search_client)
    #
    # # insert data
    insert_data(search_client)
    #

    search(search_client)
    # # query data
    query(search_client)
