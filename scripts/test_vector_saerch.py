import os

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
                        "text": {"type": "text"},
                        "vector": {
                            "type": "dense_vector",
                            "dims": 9
                        }
                    }
                }
            },
        )
        print("Index created: ", index_name)
    else:
        print("Index already exists: ", index_name)


def insert_data(search_client):
    vectors = [
        {"text": "vector 1", "vector": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"text": "vector 2", "vector": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
        {"text": "vector 3", "vector": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}
    ]
    for i, vector in enumerate(vectors):
        search_client.index(index=index_name, id=i, body=vector)


def query(search_client):
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
    for hit in search_results['hits']['hits']:
        print(f"HIT:::{hit}")
        print(f"Vector: {hit['_source']['text']}, Score: {hit['_score']}")

def drop_index(search_client):
    if search_client.indices.exists(index=index_name):
        search_client.indices.delete(index=index_name)
        print("Index deleted: ", index_name)
    else:
        print("Index does not exist: ", index_name)


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
    # create index
    init_index(search_client)

    # insert data
    insert_data(search_client)

    # query data
    query(search_client)
