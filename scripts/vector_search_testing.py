import os
from pathlib import Path
import json

import numpy as np
from elasticsearch.helpers import scan
from tqdm import tqdm
import torch
import openai
from sentence_transformers import SentenceTransformer

from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

from openkaito.tasks import generate_question_from_eth_denver_segments
from openkaito.utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM

root_dir = __file__.split("scripts")[0]
index_name = "eth_denver_vector_search_v1"


### Extract Eth Denver dataset
def extract_dataset():
    """Extract Eth Denver dataset to datasets/eth_denver_dataset directory"""

    if os.path.exists(root_dir + "datasets/eth_denver_dataset"):
        print(
            "Eth Denver data already extracted to: ",
            root_dir + "datasets/eth_denver_dataset",
        )
    else:
        import tarfile

        with tarfile.open(
                root_dir + "datasets/eth_denver_dataset.tar.gz", "r:gz"
        ) as tar:
            tar.extractall(root_dir + "datasets")

        print(
            "Eth Denver data extracted to: ", root_dir + "datasets/eth_denver_dataset"
        )

    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)
    print(f"{len(list(dataset_path.glob('*.json')))} files in {dataset_dir}")


def init_index(search_client):
    """Initialize Eth Denver index in Elasticsearch"""

    if not search_client.indices.exists(index=index_name):
        print("creating index...", index_name)
        search_client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "episode_id": {"type": "keyword"},
                        "segment_id": {"type": "long"},
                        "episode_title": {"type": "text"},
                        "episode_url": {"type": "text"},
                        "created_at": {"type": "date"},
                        "company_name": {"type": "keyword"},
                        "segment_start_time": {"type": "float"},
                        "segment_end_time": {"type": "float"},
                        "text": {"type": "text"},
                        "speaker": {"type": "keyword"},
                        "question": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1024,
                        },
                    }
                }
            },
        )
        print("Index created: ", index_name)
    else:
        print("Index already exists: ", index_name)


def drop_index(search_client, index_name):
    """Drop index in Elasticsearch"""

    if search_client.indices.exists(index=index_name):
        search_client.indices.delete(index=index_name)
        print("Index deleted: ", index_name)
    else:
        print("Index does not exist: ", index_name)


def indexing_docs(search_client):
    """Index documents in Elasticsearch"""

    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)

    num_files = len(list(dataset_path.glob("*.json")))
    print(f"Indexing {num_files} files in {dataset_dir}")
    i = 0
    for doc_file in tqdm(
            dataset_path.glob("*.json"), total=num_files, desc="Indexing docs"
    ):
        with open(doc_file, "r") as f:
            doc = json.load(f)
            segments = [doc]
            question = generate_question_from_eth_denver_segments(
                llm_client, segments
            )
            doc['question'] = question
            print(f"DOC::::::::::{doc}")
            search_client.index(index=index_name, body=doc, id=doc["doc_id"])
            i = i + 1

        if i == 80:
            break


# def update_questions(llm_client, search_client):
#     """Index documents in Elasticsearch"""
#     print(f"helpers.scan total:::{search_client.count(index=index_name)['count']}")
#     for doc in tqdm(
#             helpers.scan(search_client, index=index_name),
#             desc="update_questions",
#             total=search_client.count(index=index_name)["count"],
#     ):
#         segments = [doc["_source"]]
#         doc_id = doc["_id"]
#         question = generate_question_from_eth_denver_segments(
#             llm_client, segments
#         )
#         print(f"DOC::::::::::{doc} - question::::{question}")
#
#         search_client.update(
#             index=index_name,
#             id=doc_id,
#             body={"doc": {"question": question}, "doc_as_upsert": True},
#         )


def text_to_embedding(text):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model.encode(text).tolist()


def indexing_embeddings(search_client):
    """Index embeddings of documents in Elasticsearch"""
    for doc in tqdm(
            helpers.scan(search_client, index=index_name),
            desc="Indexing embeddings",
            total=search_client.count(index=index_name)["count"],
    ):
        doc_id = doc["_id"]
        text = doc["_source"]["question"]
        emb = text_embedding(text)[0]
        emb = pad_tensor(emb, max_len=MAX_EMBEDDING_DIM)

        search_client.update(
            index=index_name,
            id=doc_id,
            body={"doc": {"embedding": emb.tolist()}, "doc_as_upsert": True},
        )




# def vector_search(search_client, query_embedding, index, top_n=10):
#     # script_query = {
#     #     "script_score": {
#     #         "query": {"match_all": {}},
#     #         "script": {
#     #             "source": "cosineSimilarity(params.query_vector, doc['embedding']) + 1.0",
#     #             "params": {"query_vector": query_embedding}
#     #         }
#     #     }
#     # }
#     #
#     # search_query = {
#     #     "size": top_n,
#     #     "query": script_query,
#     #     "_source": ["episode_id", "episode_title", "episode_url", "created_at", "company_name",
#     #                 "segment_start_time", "segment_end_time", "text", "speaker", "segment_id", "doc_id"]
#     # }
#     #
#     # res = search_client.search(index=index, body=search_query)
#     # return res['hits']['hits']
#
#     search_query = {
#         "size": top_n,
#         "query": {
#             "script_score": {
#                 "query": {"match_all": {}},
#                 "script": {
#                     "source": "cosineSimilarity(params.query_vector, doc['question_embedding']) + 1.0",
#                     "params": {"query_vector": query_embedding.tolist()}
#                 }
#             }
#         },
#         # "_source": ["episode_id", "episode_title", "episode_url", "created_at", "company_name", "segment_start_time",
#         #             "segment_end_time", "text", "speaker", "segment_id", "doc_id"]
#     }
#
#     res = search_client.search(index=index, body=search_query)
#     return res['hits']['hits']

def search_similar_questions(search_client, query_embedding, top_n=10):
    """Search similar questions based on the query embedding"""
    try:
        # query = {
        #     "size": top_n,
        #     "query": {
        #         "script_score": {
        #             "query": {"match_all": {}},
        #             "script": {
        #                 "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
        #                 "params": {"query_vector": query_embedding.tolist()}
        #             }
        #         }
        #     }
        # }

        # query = {
        #     "size": top_n,
        #     "query": {
        #         "knn": {
        #             "embedding": {
        #                 "vector": query_embedding.tolist(),
        #                 "k": top_n
        #             }
        #         }
        #     }
        # }

        query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding.tolist(),
                "k": 5,
                "num_candidates": 5 * 5,
            },
            "_source": {
                "excludes": ["embedding"],
            },
        }
        res = search_client.search(index=index_name, body=query)
        return res["hits"]["hits"]
    except Exception as inst:
        print(f'TYPE ERROR:::{type(inst)}')
        print(f'ERROR:::{json.dumps(inst.args)}')


def search(search_client):
    # query = {
    #     "query": {
    #         "match": {
    #             "speaker": "John Paller"
    #         }
    #     }
    # }

    query = {
        "_source": {
            "excludes": ["embedding"]
        },
        "query": {
            "match_all": {}
        }
    }

    print(f"es_query: {query}")

    try:
        response = search_client.search(
            index=index_name,
            body=query,
            size=10000
        )
        documents = response["hits"]["hits"]
        for i, document in enumerate(documents):
            print(f"INDEX::::{i} - DOC:::{document}")

    except Exception as e:
        print("recall error...", e)


if __name__ == "__main__":
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

    llm_client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        organization=os.getenv("OPENAI_ORGANIZATION"),
        max_retries=3,
    )

    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)

    num_files = len(list(dataset_path.glob("*.json")))

    # extract_dataset()
    #
    # drop_index(search_client, index_name)
    # init_index(search_client)
    #
    # r = search_client.count(index=index_name)
    # indexing_docs(search_client)
    #
    # indexing_embeddings(search_client)
    #
    # search(search_client)

    #Example query
    query_text = "How can building good UX for smart contract wallets benefit global users?"
    embedding = text_embedding(query_text)[0]
    embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
    print(f"query_embedding:::{embedding.tolist()}")

    # Perform vector search
    results = search_similar_questions(search_client, embedding)
    print(f"RESULT::::{results}")
    # Display results
    for result in results:
        print(f"RESULT::::{result}")
        print(f"Score: {result['_score']}")
        print()


