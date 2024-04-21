import os
import time

import openai
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.evaluation.evaluator import Evaluator
from openkaito.protocol import SearchSynapse, StructuredSearchSynapse
from openkaito.search.ranking import HeuristicRankingModel
from openkaito.search.structured_search_engine import StructuredSearchEngine
import bittensor as bt


class CrawlJob():
    def __init__(self):
        self.twitter_usernames = None
        load_dotenv()

        self.search_client = Elasticsearch(
            os.environ["ELASTICSEARCH_HOST"],
            basic_auth=(
                os.environ["ELASTICSEARCH_USERNAME"],
                os.environ["ELASTICSEARCH_PASSWORD"],
            ),
            verify_certs=False,
            ssl_show_warn=False,
        )

        # for ranking results evaluation
        llm_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            max_retries=3,
        )

        # # for integrity check
        # twitter_crawler = ApifyTwitterCrawler(os.environ["APIFY_API_KEY"])
        self.evaluator = Evaluator(llm_client, None)

        # for ranking recalled results
        self.ranking_model = HeuristicRankingModel(length_weight=0.8, age_weight=0.2)

        # optional, for crawling data
        self.twitter_crawler = (
            ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])
            if os.environ.get("APIFY_API_KEY")
            else None
        )

        self.structured_search_engine = StructuredSearchEngine(
            search_client=self.search_client,
            relevance_ranking_model=self.ranking_model,
            twitter_crawler=self.twitter_crawler
        )

    def load_authors(self):
        with open("twitter_usernames.txt") as f:
            twitter_usernames = f.read().strip().splitlines()
        self.twitter_usernames = ['AllThingsETH']

    def run(self, evaluator):
        self.load_authors()
        bt.logging.info(f"load usernames successful")
        for i in range(0, len(self.twitter_usernames)):
            # get data from elas
            search_query = StructuredSearchSynapse(
                author_usernames=[self.twitter_usernames[i]],
                size=90,
            )
            results_search = self.structured_search_engine.search_and_mark(search_query)
            processed_docs = evaluator.llm_author_index_data_evaluation_optimize(results_search)
            self.save(processed_docs)

    def save(self, processed_docs):
        if len(processed_docs) > 0:
            try:
                bt.logging.info(f"bulk indexing {len(processed_docs)} docs")
                bulk_body = []
                for doc in processed_docs:
                    bulk_body.append(
                        {
                            "update": {
                                "_index": "twitter",
                                "_id": doc["id"],
                            }
                        }
                    )
                    bulk_body.append(
                        {
                            "doc": doc,
                            "doc_as_upsert": True,
                        }
                    )

                r = self.search_client.bulk(
                    body=bulk_body,
                    refresh=True,
                )
                bt.logging.trace("bulk update response...", r)
                if not r.get("errors"):
                    bt.logging.info("bulk update succeeded")
                else:
                    bt.logging.error("bulk update failed: ", r)
            except Exception as e:
                bt.logging.error("bulk update error...", e)


if __name__ == "__main__":
    bt.logging.info(f"Start job...")
    job = CrawlJob()
    while True:
        job.run(job.evaluator)
        time.sleep(600 * 60)
