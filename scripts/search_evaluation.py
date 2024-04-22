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

    ranked_docs = [{'id': '1782041491717898462', 'text': 'Big Brain Holdings: we invested into the ZigZag in 2022, which is a financial loss. Some of the previous founders of ZigZag are now part of the ZKasino, which appears to be fraudulent. We have never invested in ZKasino but were offered a pro-rata token distribution that we have not received and will not opt to receive. https://t.co/YsY9nMLaN9', 'created_at': '2024-04-21T13:39:35+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1782041491717898462', 'quote_count': 4, 'reply_count': 9, 'retweet_count': 5, 'favorite_count': 45, 'choice': 'insightful', 'reason': 'Provides information about investments in ZigZag and ZKasino, highlighting potential fraudulent activities.'},

{'id': '1782013139732861294', 'text': 'MEXC said: We are just one of the investors, and the behavior of the project side has nothing to do with us. As investors, we are also victims. ZKasino previously announced the completion of Series A financing at a valuation of US$350 million, with participation from MEXC, Big Brain Holdings, Trading_axe, Pentoshi and Sisyphus. https://t.co/DeMOxjP2W5', 'created_at': '2024-04-21T11:46:55+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1782013139732861294', 'quote_count': 8, 'reply_count': 25, 'retweet_count': 14, 'favorite_count': 95, 'choice': 'insightful', 'reason': "MEXC's statement regarding the project side's behavior and the Series A financing of ZKasino involving various investors."},

{'id': '1782006926257312240', 'text': "Asia's weekly TOP10 crypto news: Japan Releases Web3 White Paper,  Russia Supports Cryptocurrency for International Settlements and Top10 News（0415-0421）\nhttps://t.co/iDSXdDRlWu https://t.co/SFKRxULHBx", 'created_at': '2024-04-21T11:22:13+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1782006926257312240', 'quote_count': 0, 'reply_count': 4, 'retweet_count': 3, 'favorite_count': 11, 'choice': 'somewhat insightful', 'reason': "Mentions various crypto news from Asia, including Japan's Web3 White Paper and Russia's support for cryptocurrency."},

{'id': '1781878036947976672', 'text': 'According to glassnode, affected by the Runes minting activity, on April 20, Bitcoin miner revenue reached US$106.7 million, of which 75.444% came from network transaction fees, both reaching record highs. https://t.co/lVSyqn1UaE https://t.co/xjkkTor2I9', 'created_at': '2024-04-21T02:50:04+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1781878036947976672', 'quote_count': 8, 'reply_count': 9, 'retweet_count': 16, 'favorite_count': 115, 'choice': 'insightful', 'reason': 'Shares data from glassnode about Bitcoin miner revenue and transaction fees, highlighting record highs.'},

 {'id': '1781876293598167452', 'text': 'According to Reddit, some MtGox creditors said that the official website form has been updated recently, including the expected number of tokens to be withdrawn (BTC BCH Yen) and the payment status. It is expected that Mt Gox will distribute its holdings of 142,000 BTC, 143,000 BCH and 69 billion yen to creditors before October 31, 2024. https://t.co/krrXZOlU0N', 'created_at': '2024-04-21T02:43:08+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1781876293598167452', 'quote_count': 20, 'reply_count': 31, 'retweet_count': 40, 'favorite_count': 213, 'choice': 'insightful', 'reason': 'Provides updates on MtGox creditors and the expected distribution of BTC, BCH, and yen to creditors.'},
{'id': '1712956916459053483', 'text': "Oppa's Friend Tech holders are diamond handed 💎\n\nOppa is proud of finding such a good group\n\nOppa needs your help to find more (3,3)s 🔑\n\nPlease help Oppa retweet for awareness while Oppa does some shopping for keys this weekend \U0001faf0 https://t.co/Da9srSS09e", 'created_at': '2023-10-13T22:22:08+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1712956916459053483', 'quote_count': 0, 'reply_count': 13, 'retweet_count': 6, 'favorite_count': 27, 'choice': 'insightless', 'reason': 'Repetitive positive feedback without new information.'},

{'id': '1712651702904451316', 'text': "Oppa can't wait to buy some more keys over the weekend\n\nOppa is proud of having diamond-handed holders 🔑\n\nIf you want to (3,3), Oppa is ready \n\nPlease share this for awareness and Oppa will support you too 🙏 https://t.co/wbf0dxEpqG", 'created_at': '2023-10-13T02:09:20+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1712651702904451316', 'quote_count': 0, 'reply_count': 16, 'retweet_count': 7, 'favorite_count': 30, 'choice': 'insightless', 'reason': 'Lacks depth or valuable insights.'},

 {'id': '1712317718571970929', 'text': "Clocked in 18+ hours at the firm today\n\nChecked to see that Oppa's Friend Tech holders are diamond-handed \n\nIf you want to (3,3), please DM Oppa and let's get the show on the road \n\nOppa is on a mission and no jeets can get in the way \U0001fae1 https://t.co/ahGqCY41ks", 'created_at': '2023-10-12T04:02:12+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1712317718571970929', 'quote_count': 0, 'reply_count': 8, 'retweet_count': 7, 'favorite_count': 18, 'choice': 'insightless', 'reason': 'Primarily positive feedback without additional insights.'},

 {'id': '1712098686954508780', 'text': "Oppa will not be stopped by the Friend Tech jeets\n\nIf you're not a jeet and want to (3,3) please let me know\n\nOppa will also be buying some keys from the comments of this post who have been supporting Oppa 🔑\n\nPlease help Oppa retweet for awareness 🙏\U0001faf6 https://t.co/qFSvRtErBI", 'created_at': '2023-10-11T13:31:50+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1712098686954508780', 'quote_count': 0, 'reply_count': 14, 'retweet_count': 4, 'favorite_count': 36, 'choice': 'insightless', 'reason': 'Primarily consists of generic positive feedback.'},

{'id': '1711760943305126348', 'text': "Oppa is going to make the jeets regret \n\nLooking to (3,3) with any trustworthy keys \n\nIf you want to be Oppa's fren and (3,3), drop your key link below and let's make a deal 🙏 \n\nAlso buying a couple keys that retweet this post 🫂 https://t.co/CorgPc9exb", 'created_at': '2023-10-10T15:09:46+00:00', 'username': 'MrOppa_Eth', 'url': 'https://x.com/MrOppa_Eth/status/1711760943305126348', 'quote_count': 1, 'reply_count': 22, 'retweet_count': 16, 'favorite_count': 34, 'choice': 'insightless', 'reason': 'Lacks substantial information or insights.'}
]


    # note this is the llm score, skipped integrity check and batch age score
    score = evaluator.llm_keyword_ranking_evaluation(args.query, ranked_docs)
    print("======LLM Score======")
    print(score)


if __name__ == "__main__":
    main()
