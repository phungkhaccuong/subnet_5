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

{'id': '1781869394639143294', 'text': 'Satoshi Nakamoto’s first collaborator, sophomore Martti Malmi, disclosed his emails with Satoshi. Satoshi mentioned that there is no need to promote "anonymity" and Bitcoin as an investment. Bitcoin POW consumes less energy than the traditional banking system. POW is the only solution that can make a P2P electronic cash system work without a trusted third party. https://t.co/Oiavf8TLEO', 'created_at': '2024-04-21T02:15:43+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1781869394639143294', 'quote_count': 3, 'reply_count': 15, 'retweet_count': 32, 'favorite_count': 198, 'choice': 'insightful', 'reason': "Reveals insights from Satoshi Nakamoto's first collaborator, Martti Malmi, regarding Bitcoin's energy consumption and design."},

 {'id': '1781867793174114533', 'text': 'Bitcoin developer Luke Dashjr said that to filter Runes spam using either BitcoinKnots or Bitcoin Core, the only approach right now is to set datacarriersize=0 in your bitcoin.conf file (or the equivalent GUI option in Knots only).\n\nAfter Runes went online yesterday, it once pushed the Bitcoin network transaction fee to more than 2,000 sat/vb. It has now fallen back to about 100 sat/vb (about $9). https://t.co/5nuqHgLqj4', 'created_at': '2024-04-21T02:09:22+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1781867793174114533', 'quote_count': 2, 'reply_count': 7, 'retweet_count': 9, 'favorite_count': 46, 'choice': 'insightful', 'reason': "Discusses Bitcoin developer Luke Dashjr's comments on filtering Runes spam and its impact on transaction fees."},

{'id': '1781862517217825101', 'text': 'The IDO platform Ape Terminal issued a statement that the ZKasino IDO has been cancelled, and all participants will receive refunds. In addition, ZKasino has transferred 10,515 ETH (approximately $32.9 million) from the deposit contract to a 3/4 multisig contract (0x79…c491), and subsequently deposited them all into Lido. \nhttps://t.co/613qM1yjnf\nhttps://t.co/hVVOoHCJxe', 'created_at': '2024-04-21T01:48:24+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1781862517217825101', 'quote_count': 6, 'reply_count': 13, 'retweet_count': 16, 'favorite_count': 76, 'choice': 'insightful', 'reason': 'Highlights the cancellation of ZKasino IDO, fund transfers, and deposits into Lido.'},

{'id': '1781712167282332053', 'text': 'Exclusive: Victory Securities internally released the Hong Kong Bitcoin Ethereum spot ETF subscription guide and disclosed its charging standards. Hong Kong securities firms are selling to potential clients.\n\nThe original text of the picture is in Chinese, using Google Translate @EricBalchunas', 'created_at': '2024-04-20T15:50:57+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1781712167282332053', 'quote_count': 8, 'reply_count': 9, 'retweet_count': 16, 'favorite_count': 102, 'choice': 'somewhat insightful', 'reason': 'Shares information about Victory Securities releasing a Hong Kong Bitcoin Ethereum spot ETF guide.'},

 {'id': '1781635210171609351', 'text': 'The Chinese Embassy reminds that Angola’s “Law on the Prohibition of Cryptocurrency and Other Virtual Asset Mining” has officially come into effect on April 10. This law stipulates that cryptocurrency mining is a crime and may be punished with a prison sentence of 1 to 12 years. https://t.co/4buW8WUNvF', 'created_at': '2024-04-20T10:45:09+00:00', 'username': 'WuBlockchain', 'url': 'https://x.com/WuBlockchain/status/1781635210171609351', 'quote_count': 6, 'reply_count': 8, 'retweet_count': 11, 'favorite_count': 76, 'choice': 'insightful', 'reason': "Highlights Angola's cryptocurrency mining prohibition law and its implications."}]

    # note this is the llm score, skipped integrity check and batch age score
    score = evaluator.llm_keyword_ranking_evaluation(args.query, ranked_docs)
    print("======LLM Score======")
    print(score)


if __name__ == "__main__":
    main()
