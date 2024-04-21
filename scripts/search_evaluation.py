import argparse
import os
import random

import bittensor as bt
import openai
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from openkaito.crawlers.twitter.microworlds import MicroworldsTwitterCrawler
from openkaito.evaluation.evaluator import Evaluator
from openkaito.protocol import SearchSynapse, SortType, StructuredSearchSynapse
from openkaito.search.ranking import OptimizeRankingModelV1
from openkaito.search.ranking.heuristic_ranking import HeuristicRankingModel
from openkaito.search.structured_search_engine import StructuredSearchEngine
from openkaito.tasks import generate_author_index_task


def parse_args():
    parser = argparse.ArgumentParser(description="Miner Search Ranking Evaluation")
    parser.add_argument("--query", type=str, default="BTC", help="query string")
    parser.add_argument(
        "--size", type=int, default=5, help="size of the response items"
    )
    # parser.add_argument('--crawling', type=bool, default=False, action='store_true', help='crawling data before search')

    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))
    load_dotenv()
    bt.logging.set_trace(True)

    # for ranking results evaluation
    llm_client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION"),
        max_retries=3,
    )

    # # for integrity check
    # twitter_crawler = ApifyTwitterCrawler(os.environ["APIFY_API_KEY"])
    evaluator = Evaluator(llm_client, None)

    search_client = Elasticsearch(
        os.environ["ELASTICSEARCH_HOST"],
        basic_auth=(
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )

    # for ranking recalled results
    ranking_model = OptimizeRankingModelV1(length_weight=0.8, age_weight=0.2)

    search_engine = StructuredSearchEngine(
        search_client=search_client,
        relevance_ranking_model=ranking_model,
        twitter_crawler=None,
    )

    search_query = SearchSynapse(
        query_string=args.query,
        size=args.size,
    )

    # search_query = generate_author_index_task(size=10, num_authors=2)
    #
    # print(search_query)
    #
    # ranked_docs = search_engine.search(search_query=search_query)
    # print("======ranked documents======")
    # print(ranked_docs)

    ranked_docs = [{'id': '1712956916459053483', 'text': "Oppa's Friend Tech holders are diamond handed 💎\n\nOppa is proud of finding such a good group\n\nOppa needs your help to find more (3,3)s 🔑\n\nPlease help Oppa retweet for awareness while Oppa does some shopping for keys this weekend \U0001faf0 https://t.co/Da9srSS09e", 'created_at': '2023-10-13T22:22:08+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1712956916459053483', 'quote_count': 0, 'reply_count': 13, 'retweet_count': 6, 'favorite_count': 27, 'choice': 'insightless', 'reason': 'Repetitive positive feedback without new information.'},

{'id': '1712651702904451316', 'text': "Oppa can't wait to buy some more keys over the weekend\n\nOppa is proud of having diamond-handed holders 🔑\n\nIf you want to (3,3), Oppa is ready \n\nPlease share this for awareness and Oppa will support you too 🙏 https://t.co/wbf0dxEpqG", 'created_at': '2023-10-13T02:09:20+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1712651702904451316', 'quote_count': 0, 'reply_count': 16, 'retweet_count': 7, 'favorite_count': 30, 'choice': 'insightless', 'reason': 'Lacks depth or valuable insights.'},

 {'id': '1712317718571970929', 'text': "Clocked in 18+ hours at the firm today\n\nChecked to see that Oppa's Friend Tech holders are diamond-handed \n\nIf you want to (3,3), please DM Oppa and let's get the show on the road \n\nOppa is on a mission and no jeets can get in the way \U0001fae1 https://t.co/ahGqCY41ks", 'created_at': '2023-10-12T04:02:12+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1712317718571970929', 'quote_count': 0, 'reply_count': 8, 'retweet_count': 7, 'favorite_count': 18, 'choice': 'insightless', 'reason': 'Primarily positive feedback without additional insights.'},

 {'id': '1712098686954508780', 'text': "Oppa will not be stopped by the Friend Tech jeets\n\nIf you're not a jeet and want to (3,3) please let me know\n\nOppa will also be buying some keys from the comments of this post who have been supporting Oppa 🔑\n\nPlease help Oppa retweet for awareness 🙏\U0001faf6 https://t.co/qFSvRtErBI", 'created_at': '2023-10-11T13:31:50+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1712098686954508780', 'quote_count': 0, 'reply_count': 14, 'retweet_count': 4, 'favorite_count': 36, 'choice': 'insightless', 'reason': 'Primarily consists of generic positive feedback.'},

{'id': '1711760943305126348', 'text': "Oppa is going to make the jeets regret \n\nLooking to (3,3) with any trustworthy keys \n\nIf you want to be Oppa's fren and (3,3), drop your key link below and let's make a deal 🙏 \n\nAlso buying a couple keys that retweet this post 🫂 https://t.co/CorgPc9exb", 'created_at': '2023-10-10T15:09:46+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1711760943305126348', 'quote_count': 1, 'reply_count': 22, 'retweet_count': 16, 'favorite_count': 34, 'choice': 'insightless', 'reason': 'Lacks substantial information or insights.'},

{'id': '1711073487048089606', 'text': 'Oppa is lighting it up on Friend Tech\n\nLooking for more high quality (3,3)s- NO jeets please 🙏 \n\nGot 21 E to deploy\n\nIf you’re not a jeet, comment your key link and retweet this post, Oppa will come you scoop up 🔑\U0001faf0 https://t.co/hIskpOd0In', 'created_at': '2023-10-08T17:38:04+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1711073487048089606', 'quote_count': 1, 'reply_count': 16, 'retweet_count': 7, 'favorite_count': 31, 'choice': 'insightless', 'reason': 'Lacks substantial information or insights.'},

 {'id': '1710816118619693265', 'text': 'Time to LIGHT IT UP on Friend Tech!\n\nGot 23 E left, I want to run it up \n\nLooking for quality (3,3) and new friends 🤝\n\nDrop a link to your key below and look out for a DM 🔽\n\nRetweet, like and comment your key on this post 🔑 https://t.co/DY8qZyUJgV', 'created_at': '2023-10-08T00:35:22+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1710816118619693265', 'quote_count': 0, 'reply_count': 38, 'retweet_count': 17, 'favorite_count': 60, 'choice': 'insightless', 'reason': 'Lacks substantial information or insights.'},

{'id': '1710358099251339539', 'text': 'Friend Tech is so addicting😅 Loaded up 25 E more\n\nHelp me find new friends to (3,3) 🗝️\n\nIf your key is under .5 E🔽 LMK in the comments below\n\nRetweet, like and comment your key on this post https://t.co/PWysDhNVNz', 'created_at': '2023-10-06T18:15:22+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1710358099251339539', 'quote_count': 1, 'reply_count': 64, 'retweet_count': 24, 'favorite_count': 94, 'choice': 'insightless', 'reason': 'Lacks substantial information or insights.'},

 {'id': '1710136556718067915', 'text': 'It’s time to be serious on Friend Tech and pick up keys! \n\nStarting with $10k. I want to buy upcoming creators \n\nIf your key is under .1 E LMK in the comments below \n\nRetweet, like and comment your key on this post! https://t.co/xoQITcw6oU', 'created_at': '2023-10-06T03:35:02+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1710136556718067915', 'quote_count': 0, 'reply_count': 42, 'retweet_count': 56, 'favorite_count': 70, 'choice': 'insightless', 'reason': 'Primarily consists of generic positive feedback.'},

 {'id': '1671569719772729344', 'text': 'Every speaker in this space is supporting an artist whom they know has stolen artwork but still claims to be an original artist.\n\nBlock all of them, and do not look back:\n@ironspidernft @abstraordinals @TO @businessmaneth @LeonidasNFT @KingAnt777 @OriginalMurr @ElenaaETH https://t.co/KBcRGwjakk', 'created_at': '2023-06-21T17:24:12+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1671569719772729344', 'quote_count': 0, 'reply_count': 2, 'retweet_count': 1, 'favorite_count': 6, 'choice': 'insightless', 'reason': 'Primarily positive feedback without additional insights.'}]
    # note this is the llm score, skipped integrity check and batch age score
    score = evaluator.llm_keyword_ranking_evaluation(args.query, ranked_docs)
    print("======LLM Score======")
    print(score)


if __name__ == "__main__":
    main()
