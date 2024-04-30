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
                        "vector": {
                            "type": "dense_vector",
                            "dims": 10,
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
    vectors = [
        {"price": 20, "vector": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"price": 19, "vector": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
        {"price": 18, "vector": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}
    ]
    for i, vector in enumerate(vectors):
        search_client.index(index=index_name, id=i, body=vector)


def query(search_client):
    print('search query')
    query_vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_results = search_client.search(index=index_name, body={"query": script_query})
    print(f"search_results:::{search_results}")
    for hit in search_results['hits']['hits']:
        print(f"HIT:::{hit}")
        print(f"Vector: {hit['_source']['text']}, Score: {hit['_score']}")

def drop_index(search_client):
    if search_client.indices.exists(index=index_name):
        search_client.indices.delete(index=index_name)
        print("Index deleted: ", index_name)
    else:
        print("Index does not exist: ", index_name)

def main(search_client):
    search_client.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "price": {
                        "type": "integer"
                    },
                    "vector": {
                        "type": "dense_vector",
                        "dims": 10,
                        "index": False
                    }
                }
            }
        },
    )

    # Index some vectors
    vectors = [
        {"price": 20, "vector": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
        {"price": 19, "vector": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]},
        {"price": 18, "vector": [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]}
    ]

    for i, vector in enumerate(vectors):
        search_client.index(index=index_name, id=i, body=vector)

    # Perform vector similarity search
    query_vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                "params": {"queryVector": query_vector}
            }
        }
    }

    search_results = search_client.search(index=index_name, body={"query": script_query})
    print(f"RESULT:{search_results}")
    for hit in search_results['hits']['hits']:
        print(f"Vector: {hit['_source']['text']}, Score: {hit['_score']}")


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
    # init_index(search_client)
    #
    # # insert data
    # insert_data(search_client)
    #
    # # query data
    # query(search_client)
    main(search_client)
