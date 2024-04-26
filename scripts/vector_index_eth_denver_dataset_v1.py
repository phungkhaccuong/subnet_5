"""Vector Search of Eth Denver dataset using Elasticsearch and Transformers

This script extracts the Eth Denver dataset (open-sourced by https://portal.kaito.ai/events/ETHDenver2024 ), indexes the documents in Elasticsearch, and indexes the embeddings of the documents in Elasticsearch.
It also provides a test query to retrieve the top-k similar documents to the query.

This script is intentionally kept transparent and hackable, and miners may do their own customizations.
"""

import os
from pathlib import Path
import json
from tqdm import tqdm
import torch

from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

from openkaito.utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM

root_dir = __file__.split("scripts")[0]
index_name = "eth_denver"
# Define your scroll timeout and batch size
scroll_timeout = '10m'  # Adjust as needed
batch_size = 1000  # Adjust as needed


### Extract Eth Denver dataset
def extract_eth_denver_dataset():
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


def init_eth_denver_index(search_client):
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
            search_client.index(index=index_name, body=doc, id=doc["doc_id"])


def indexing_embeddings(search_client):
    """Index embeddings of documents in Elasticsearch"""

    for doc in tqdm(
            helpers.scan(search_client, index=index_name),
            desc="Indexing embeddings",
            total=search_client.count(index=index_name)["count"],
    ):
        doc_id = doc["_id"]
        text = doc["_source"]["text"]
        embedding = text_embedding(text)[0]
        embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
        search_client.update(
            index=index_name,
            id=doc_id,
            body={"doc": {"embedding": embedding.tolist()}, "doc_as_upsert": True},
        )


def test_retrieval(search_client, query, topk=5):
    """Test retrieval of top-k similar documents to query"""

    embedding = text_embedding(query)[0]
    embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
    body = {
        "knn": {
            "field": "embedding",
            "query_vector": embedding.tolist(),
            "k": topk,
            "num_candidates": 5 * topk,
        },
        # "_source": {
        #     "excludes": ["embedding"],
        # },
    }

    response = search_client.search(index=index_name, body=body)
    return response


# Function to index embeddings
def index_embeddings(search_client, index_name, text_embedding, pad_tensor, MAX_EMBEDDING_DIM):
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

    # Iterate over batches of documents
    while True:
        try:
            batch = next(scroll)
            batch_updates = []

            for doc in batch:
                if isinstance(doc, dict):
                    doc = doc["_source"]
                else:
                    continue

                doc_id = doc["_id"]
                text = doc["text"]

                # Ensure that text_embedding returns a list or array
                embedding = text_embedding(text)
                if isinstance(embedding, str):
                    embedding = [embedding]  # Convert to list if embedding is a string
                elif isinstance(embedding, list) and isinstance(embedding[0], str):
                    embedding = [float(emb) for emb in embedding]  # Convert string elements to float

                embedding = pad_tensor(embedding[0], max_len=MAX_EMBEDDING_DIM)
                batch_updates.append({
                    "_op_type": "update",
                    "index": index_name,  # Use "index" instead of "_index"
                    "id": doc_id,
                    "doc": {"embedding": embedding.tolist()},
                    "doc_as_upsert": True
                })

            # Bulk update
            if batch_updates:
                helpers.bulk(search_client, batch_updates)

            pbar.update(len(batch))

        except StopIteration:
            break

    pbar.close()


if __name__ == "__main__":
    load_dotenv()

    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)
    print(f"dataset_path:::{dataset_path}")

    num_files = len(list(dataset_path.glob("*.json")))
    print(f"num_files:::{num_files}")
    extract_eth_denver_dataset()

    search_client = Elasticsearch(
        os.environ["ELASTICSEARCH_HOST"],
        basic_auth=(
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )

    drop_index(search_client, index_name)
    init_eth_denver_index(search_client)

    r = search_client.count(index=index_name)
    if r["count"] != num_files:
        print(
            f"Number of docs in {index_name}: {r['count']} != total files {num_files}, reindexing docs..."
        )
        indexing_docs(search_client)
    else:
        print(
            f"Number of docs in {index_name}: {r['count']} == total files {num_files}, no need to reindex docs"
        )
    # Call the function to index embeddings
    index_embeddings(search_client, index_name, text_embedding, pad_tensor, MAX_EMBEDDING_DIM)
    #indexing_embeddings(search_client)

    query = "What is the future of blockchain?"
    response = test_retrieval(search_client, query, topk=5)
    # print(response)
    for response in response["hits"]["hits"]:
        print(response["_source"]["created_at"])
        print(response["_source"]["episode_title"])
        print(response["_source"]["speaker"])
        print(response["_source"]["text"])
        print(response["_score"])
