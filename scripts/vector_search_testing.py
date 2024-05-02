import os
from pathlib import Path
import json

import numpy as np
from elasticsearch.helpers import scan
from tqdm import tqdm
import torch
import openai
import csv
from sentence_transformers import SentenceTransformer

from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

from openkaito.tasks import generate_question_from_eth_denver_segments
from openkaito.utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM
from openkaito.evaluation.evaluator import Evaluator
from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.utils.uids import get_random_uids
from openkaito.utils.version import get_version

from openkaito.tasks import (
    generate_author_index_task,
    generate_question_from_eth_denver_segments,
    generate_structured_search_task,
    random_eth_denver_segments,
    random_query,
    generate_semantic_search_task,
)

root_dir = __file__.split("scripts")[0]
index_name = "eth_denver_v1"

scroll_timeout = '30m'  # Adjust as needed
batch_size = 800  # Adjust as needed


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

    for doc_file in tqdm(
            dataset_path.glob("*.json"), total=num_files, desc="Indexing docs"
    ):
        with open(doc_file, "r") as f:
            doc = json.load(f)
            if doc['episode_id'] in ['_cCwx5zaz1I', "_aRTKs6AmvI", "_ikuHdB0GSk", "_nNl0XqM8r4", "_92SG2nJOro",
                                     'A8BVAdn7mV4', 'sQ38uwJNnxM', 'qc9p8emCjD8', 'Ee6RLuq-myE', 'TUCHkExPURw', 'qMjrvUmr_j4', 'XoyvZxOe4Ww', 'S4Lgb0a9VmM', 'dYGj7Ugk75E',
                                     'e-ack9r3WSI', 'P6TFcBEXESk', 'CnFH6bYUsNw', 'ivYuGMYh6e0', 'rhueyjK1-78', 'KeV7qd4r2Og', 'kFXrrO95Wuw', 'nyhw-KNx12k', '6fjTd7L9DHQ']:
                segments = [doc]
                question = generate_question_from_eth_denver_segments(
                    llm_client, segments
                )
                doc['question'] = question
                print(f"DOC::::::::::{doc}")
                search_client.index(index=index_name, body=doc, id=doc["doc_id"])


def text_to_embedding(text):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model.encode(text).tolist()


# def indexing_embeddings(search_client):
#     """Index embeddings of documents in Elasticsearch"""
#     for doc in tqdm(
#             helpers.scan(search_client, index=index_name),
#             desc="Indexing embeddings",
#             total=search_client.count(index=index_name)["count"],
#     ):
#         try:
#             doc_id = doc["_id"]
#             text = doc["_source"]["question"] if (doc["_source"]["question"] is not None) else ""
#             emb = text_embedding(text)[0]
#             emb = pad_tensor(emb, max_len=MAX_EMBEDDING_DIM)
#
#             search_client.update(
#                 index=index_name,
#                 id=doc_id,
#                 body={"doc": {"embedding": emb.tolist()}, "doc_as_upsert": True},
#             )
#         except Exception as e:
#             print(f"Exception:::{e}")

def indexing_embeddings(search_client, index_name, text_embedding, pad_tensor, MAX_EMBEDDING_DIM):
    # Get total count of documents
    total_docs = search_client.count(index=index_name)["count"]

    # Initialize the scroll
    scroll = helpers.scan(
        search_client,
        index=index_name,
        scroll=scroll_timeout,
        size=batch_size,
        clear_scroll=False
    )

    pbar = tqdm(total=total_docs, desc="Indexing embeddings")

    # Iterate over documents
    for doc in scroll:
        doc_id = doc["_id"]
        text = doc["_source"]["question"] if (doc["_source"]["question"] is not None) else ""
        embedding = text_embedding(text)[0]
        embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
        search_client.update(
            index=index_name,
            id=doc_id,
            body={"doc": {"embedding": embedding.tolist()}, "doc_as_upsert": True},
        )
        pbar.update(1)

    pbar.close()



def search_similar_questions(search_client, query_embedding, top_n=5):
    """Search similar questions based on the query embedding"""
    try:
        episode_ids = get_episode_ids(query_text)
        print(f"episode_ids::::::::{episode_ids}")
        if episode_ids is None:
            query = {
                "size": top_n,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding.tolist()}
                        }
                    }
                },
                "_source": {
                    "excludes": ["embedding"]
                }
            }
        else:
            query = {
                    "size": top_n,
                    "query": {
                        "bool": {
                            "filter": {
                                "terms": {
                                    "episode_id": episode_ids
                                }
                            },
                            "should": [
                                {
                                    "script_score": {
                                        "query": {
                                            "match_all": {}
                                        },
                                        "script": {
                                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                            "params": {
                                                "query_vector": query_embedding.tolist()
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "_source": {
                        "excludes": ["embedding"]
                    }
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

def list_file():
    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)

    patterns = [
        "_cCwx5zaz1I.*.json",
        "_aRTKs6AmvI.*.json",
        "_ikuHdB0GSk.*.json",
        "_nNl0XqM8r4.*.json"
    ]

    # Create a list of file paths
    file_list = [file for pattern in patterns for file in dataset_path.glob(pattern)]

    print(f"number:::::{len(file_list)}")

def load_speaker_dict():
    with open('speaker_dict.json', 'r') as json_file:
        speaker_dict = json.load(json_file)

    return speaker_dict

def get_episode_ids(query):
    speaker_dict = load_speaker_dict()
    for key, value in speaker_episode_dict.items():
        if key in query:
            return value
    return None

def rank(evaluator, query_text):
    embedding = text_embedding(query_text)[0]
    embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
    results = search_similar_questions(search_client, embedding)
    print(f"RESULTS::::{results}")
    responses = []
    for i, result in enumerate(results):
        print(f"INDEX::{i} -- DOC::{result}")
        responses.append(result['_source'])

    search_query = generate_semantic_search_task(
        query_string=query_text,
        index_name=index_name,
        version=get_version(),
    )

    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    rewards = evaluator.evaluate_semantic_search(
        search_query, [responses], dataset_dir
    )

    print(rewards)

def get_distinct_episode_ids():
    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)
    num_files = len(list(dataset_path.glob("*.json")))

    episode_ids_set = set()
    for doc_file in tqdm(
            dataset_path.glob("*.json"), total=num_files, desc="Indexing docs"
    ):
        with open(doc_file, "r") as f:
            doc = json.load(f)
            episode_ids_set.add(doc["episode_id"])

    print(f"episode_ids_set:::::{episode_ids_set}")


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

    twitter_crawler = ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])

    evaluator = Evaluator(llm_client, twitter_crawler)

    #execute query
    query_text = "What does Benny Giang consider unchangeable in talking about game tokenomics?"
    rank(evaluator, query_text)



