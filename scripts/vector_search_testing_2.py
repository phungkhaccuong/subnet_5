import json
import os
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.evaluation.evaluator import Evaluator
from openkaito.tasks import (
    generate_question_from_eth_denver_segments,
)
from openkaito.utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

root_dir = __file__.split("scripts")[0]
index_name = "eth_denver_v1"

scroll_timeout = '30m'  # Adjust as needed
batch_size = 800  # Adjust as needed


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
                            "dims": MAX_EMBEDDING_DIM,
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


def indexing_docs(search_client, episode_ids):
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
            if doc['episode_id'] in episode_ids:
                segments = [doc]
                question = generate_question_from_eth_denver_segments(
                    llm_client, segments
                )
                doc['question'] = question
                search_client.index(index=index_name, body=doc, id=doc["doc_id"])


def text_to_embedding(text):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model.encode(text).tolist()


def indexing_embeddings(search_client, index_name, text_embedding, pad_tensor, MAX_EMBEDDING_DIM, episode_ids):
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
        if doc["_source"]["episode_id"] in episode_ids:
            print("OK")
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
        else:
            print("NOT OKEEEEEEEEEEEEEEEEE")

    pbar.close()


def search_similar_questions(search_client, query_embedding, top_n=5):
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
        #     },
        #     "_source": {
        #         "excludes": ["embedding"]
        #     }
        # }

        query = {
            "size": top_n,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "dotProduct(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding.tolist()}
                    }
                }
            },
            "_source": {
                "excludes": ["embedding"]
            }
        }

        # query = {
        #     "knn": {
        #         "field": "embedding",
        #         "query_vector": query_embedding.tolist(),
        #         "k": 5,
        #         "num_candidates": 5 * 5,
        #     },
        #     "_source": {
        #         "excludes": ["embedding"],
        #     },
        # }
        res = search_client.search(index=index_name, body=query)
        return res["hits"]["hits"]
    except Exception as inst:
        print(f'TYPE ERROR:::{type(inst)}')
        print(f'ERROR:::{json.dumps(inst.args)}')


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

    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)

    num_files = len(list(dataset_path.glob("*.json")))

    extract_dataset()

    #drop_index(search_client, index_name)
    #init_index(search_client)

    episode_ids = ['Vk57ZGJM0Rg', 'PO3r4nfTp4U', 'gRRQQyQE_9w', '5yyaf6dyUmA', 'hTVkYVzfMII', 'i2U5QTepDwA',
                   'J4KMTk8Rq_4',
                   'hHFl02APV0M', 'v9_FmVPpkHY', 'Ck9q-bn_7Gg', '-yhm-hBoPvg', 'UDISNqga8w0', 'bD5RYXxJKBc',
                   'tObiEV-mppE',
                   'vb6gvCgl-uc', '5GJnk66znfc', 'FQk4ZG0SHRs', 'iaaf7DOdLvc', 'E2vR1r_599E', 'skgwA9Ht5Gc',
                   'H_FQaR6whNg',
                   'HkoZnGUEIbs', 'bhPWgyN-iq0', 'n_lRelJSslk', 'Va_tv0-vflg', 'sjDdDuhADKg', 'YhjRlHfmqHM',
                   'fJJukF1j9Lw',
                   'munvUoEpT5A', 'kP7yn_60cQE']
    indexing_docs(search_client, episode_ids)

    #indexing_embeddings(search_client, index_name, text_embedding, pad_tensor, MAX_EMBEDDING_DIM, episode_ids)

    print("DONE..........................................................")
    time.sleep(60 * 60 * 24)
