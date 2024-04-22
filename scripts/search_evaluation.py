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

    ranked_docs = [{'id': '1781548495994081646', 'text': '@federalreserve released their financial stability report. Remarkable that in just a few years stablecoins have reached sufficient scale to be listed alongside other major demand sinks for treasuries. \n\nRegulators really putting the hard sell on Hill to finalize a reg framework. https://t.co/X8MXPJZLGB', 'created_at': '2024-04-20T05:00:35+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1781548495994081646', 'quote_count': 1, 'reply_count': 1, 'retweet_count': 2, 'favorite_count': 12, 'choice': 'insightful', 'reason': 'Highlights the significance of stablecoins in the financial stability report.'},
 {'id': '1780674310207639566', 'text': 'Worth calling out that this is the 3rd or 4th letter with either just Warren, or Warren + one other Member on it. \n\nWhile she’s certainly out on an island here (with Wally and Brown on it too 🏝️), don’t underestimate ability to pull the Overton Window in her direction', 'created_at': '2024-04-17T19:06:53+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1780674310207639566', 'quote_count': 0, 'reply_count': 3, 'retweet_count': 4, 'favorite_count': 44, 'choice': 'somewhat insightful', 'reason': 'Provides some insight into the dynamics of political influence.'},

{'id': '1780655222890860889', 'text': 'Warren: financial regulators have determined “that crypto assets issued over public blockchains are highly unlikely to be consistent with safety and soundness principles.”\n\nSo private blockchains by banks like JP Morgan are ok then? \n\nWhere did Warren the bank cop go? We miss her', 'created_at': '2024-04-17T17:51:02+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1780655222890860889', 'quote_count': 5, 'reply_count': 14, 'retweet_count': 9, 'favorite_count': 85, 'choice': 'somewhat insightful', 'reason': "Raises questions about Senator Warren's stance on crypto assets."},
{'id': '1780655219392836029', 'text': 'Senator Warren really ramping up the anti-stablecoin letters. Here’s another one to Treasury Secretary Janet Yellen, calling on her to push for including “nodes in the DeFi system” [sic] in the AML sections of any stablecoin legislation https://t.co/a14fEztrFW', 'created_at': '2024-04-17T17:51:01+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1780655219392836029', 'quote_count': 11, 'reply_count': 40, 'retweet_count': 48, 'favorite_count': 189, 'choice': 'somewhat insightful', 'reason': "Provides insight into Senator Warren's actions regarding stablecoins."},

 {'id': '1778877236327649494', 'text': 'Another 🔥 @HesterPeirce speech: “the SEC has met inquiries about blockchain technology with a stubborn and decidedly uncreative insistence that existing rules work just fine. We have not worked seriously with people in the industry to figure out how to achieve our regulatory objectives in a way that is tailored to the opportunities and risks of the technology.”', 'created_at': '2024-04-12T20:05:57+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1778877236327649494', 'quote_count': 4, 'reply_count': 15, 'retweet_count': 48, 'favorite_count': 225, 'choice': 'insightful', 'reason': "Highlights @HesterPeirce's critique of the SEC's approach to blockchain technology."},
{'id': '1782041491717898462', 'text': 'Big Brain Holdings: we invested into the ZigZag in 2022, which is a financial loss. Some of the previous founders of ZigZag are now part of the ZKasino, which appears to be fraudulent. We have never invested in ZKasino but were offered a pro-rata token distribution that we have not received and will not opt to receive. https://t.co/YsY9nMLaN9', 'created_at': '2024-04-21T13:39:35+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1782041491717898462', 'quote_count': 4, 'reply_count': 9, 'retweet_count': 5, 'favorite_count': 45, 'choice': 'insightful', 'reason': 'Provides information about investments in ZigZag and ZKasino, highlighting potential fraudulent activities.'},

{'id': '1782013139732861294', 'text': 'MEXC said: We are just one of the investors, and the behavior of the project side has nothing to do with us. As investors, we are also victims. ZKasino previously announced the completion of Series A financing at a valuation of US$350 million, with participation from MEXC, Big Brain Holdings, Trading_axe, Pentoshi and Sisyphus. https://t.co/DeMOxjP2W5', 'created_at': '2024-04-21T11:46:55+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1782013139732861294', 'quote_count': 8, 'reply_count': 25, 'retweet_count': 14, 'favorite_count': 95, 'choice': 'insightful', 'reason': "MEXC's statement regarding the project side's behavior and the Series A financing of ZKasino involving various investors."},

{'id': '1782006926257312240', 'text': "Asia's weekly TOP10 crypto news: Japan Releases Web3 White Paper,  Russia Supports Cryptocurrency for International Settlements and Top10 News（0415-0421）\nhttps://t.co/iDSXdDRlWu https://t.co/SFKRxULHBx", 'created_at': '2024-04-21T11:22:13+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1782006926257312240', 'quote_count': 0, 'reply_count': 4, 'retweet_count': 3, 'favorite_count': 11, 'choice': 'somewhat insightful', 'reason': "Mentions various crypto news from Asia, including Japan's Web3 White Paper and Russia's support for cryptocurrency."},

{'id': '1781878036947976672', 'text': 'According to glassnode, affected by the Runes minting activity, on April 20, Bitcoin miner revenue reached US$106.7 million, of which 75.444% came from network transaction fees, both reaching record highs. https://t.co/lVSyqn1UaE https://t.co/xjkkTor2I9', 'created_at': '2024-04-21T02:50:04+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1781878036947976672', 'quote_count': 8, 'reply_count': 9, 'retweet_count': 16, 'favorite_count': 115, 'choice': 'insightful', 'reason': 'Shares data from glassnode about Bitcoin miner revenue and transaction fees, highlighting record highs.'},

 {'id': '1781876293598167452', 'text': 'According to Reddit, some MtGox creditors said that the official website form has been updated recently, including the expected number of tokens to be withdrawn (BTC BCH Yen) and the payment status. It is expected that Mt Gox will distribute its holdings of 142,000 BTC, 143,000 BCH and 69 billion yen to creditors before October 31, 2024. https://t.co/krrXZOlU0N', 'created_at': '2024-04-21T02:43:08+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1781876293598167452', 'quote_count': 20, 'reply_count': 31, 'retweet_count': 40, 'favorite_count': 213, 'choice': 'insightful', 'reason': 'Provides updates on MtGox creditors and the expected distribution of BTC, BCH, and yen to creditors.'}
]


    # note this is the llm score, skipped integrity check and batch age score
    score = evaluator.llm_keyword_ranking_evaluation(args.query, ranked_docs)
    print("======LLM Score======")
    print(score)


if __name__ == "__main__":
    main()
