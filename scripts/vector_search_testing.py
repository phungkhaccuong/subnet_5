import os
from pathlib import Path
import json
from tqdm import tqdm
import torch
import openai
from sentence_transformers import SentenceTransformer

from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

from openkaito.tasks import generate_question_from_eth_denver_segments
from openkaito.utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM

root_dir = __file__.split("scripts")[0]
index_name = "eth_denver_vector_search"


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
                            "dims": 384,
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
        i = i + 1
        with open(doc_file, "r") as f:
            doc = json.load(f)
            search_client.index(index=index_name, body=doc, id=doc["doc_id"])

        if i ==1000:
            break


def update_questions(llm_client, search_client):
    """Index documents in Elasticsearch"""

    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)

    num_files = len(list(dataset_path.glob("*.json")))
    print(f"Indexing {num_files} files in {dataset_dir}")
    i = 0

    for doc in tqdm(
            helpers.scan(search_client, index=index_name),
            desc="update_questions",
            total=search_client.count(index=index_name)["count"],
    ):
        i = i + 1
        segments = []
        with open(doc, "r") as f:
            segments.append(doc["_source"])
            doc_id = doc["_id"]
            question = generate_question_from_eth_denver_segments(
                llm_client, segments
            )
            print(f"DOC::::::::::{doc} - question::::{question}")

            search_client.update(
                index=index_name,
                id=doc_id,
                body={"doc": {"question": question}, "doc_as_upsert": True},
            )

        if i == 100:
            break


def indexing_embeddings(search_client):
    """Index embeddings of documents in Elasticsearch"""
    for doc in tqdm(
            helpers.scan(search_client, index=index_name),
            desc="Indexing embeddings",
            total=search_client.count(index=index_name)["count"],
    ):
        doc_id = doc["_id"]
        text = doc["_source"]["text"]

        # Initialize the SentenceTransformer model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        embedding = text_embedding(text)[0]
        embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
        search_client.update(
            index=index_name,
            id=doc_id,
            body={"doc": {"embedding": embedding.tolist()}, "doc_as_upsert": True},
        )


if __name__ == "__main__":
    load_dotenv()

    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    dataset_path = Path(dataset_dir)

    num_files = len(list(dataset_path.glob("*.json")))

    extract_dataset()

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

    drop_index(search_client, index_name)
    init_index(search_client)

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

    update_questions(llm_client, search_client)

    # indexing_embeddings(search_client)
