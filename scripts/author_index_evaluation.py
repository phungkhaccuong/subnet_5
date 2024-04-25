import os

import openai
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.evaluation.evaluator import Evaluator
from openkaito.search.ranking import OptimizeRankingModel, OptimizeRankingModelV1, HeuristicRankingModel
from openkaito.search.structured_search_engine import StructuredSearchEngine
from openkaito.tasks import generate_author_index_task


def main():
    load_dotenv()
    # bt.logging.set_debug(True)
    # bt.logging.set_trace(True)

    # for ranking results evaluation
    llm_client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION"),
        max_retries=3,
    )

    twitter_crawler = None
    # # for integrity check
    twitter_crawler = ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])
    evaluator = Evaluator(llm_client, twitter_crawler)

    search_client = Elasticsearch(
        os.environ["ELASTICSEARCH_HOST"],
        basic_auth=(
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )

    ranking_models = [
        HeuristicRankingModel(length_weight=0.8, age_weight=0.2),
        OptimizeRankingModel(length_weight=0.8, age_weight=0.2),
        OptimizeRankingModelV1(length_weight=0.77, age_weight=0.23),
    ]
    search_engines = [
        StructuredSearchEngine(
            search_client=search_client,
            relevance_ranking_model=model,
            twitter_crawler=None,
            recall_size=100
        )

        for model in ranking_models
    ]

    # search_query = StructuredSearchSynapse(
    #     size=10, author_usernames=["elonmusk", "nftbadger"]
    # )
    search_query = generate_author_index_task(size=10, num_authors=2)
    print(search_query)

    responses = [search_engine.search(search_query=search_query) for search_engine in search_engines]

    rewards = evaluator.evaluate(search_query, responses)
    print(f"Scored responses: {rewards}")


if __name__ == "__main__":
    main()
