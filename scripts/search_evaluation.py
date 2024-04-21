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

 {'id': '1778804674528673976', 'text': 'Towards the bottom of the proposal here, you can see not only the quality of work that @fund_defi has done, but the sheer quantity of it as well 🤯\n\nThere is no fiercer advocate for the decentralized future', 'created_at': '2024-04-12T15:17:37+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1778804674528673976', 'quote_count': 0, 'reply_count': 2, 'retweet_count': 2, 'favorite_count': 34, 'choice': 'insightful', 'reason': 'Highlights the extensive work and advocacy of @fund_defi for the decentralized future.'},

 {'id': '1778436723912892611', 'text': '@PatrickMcHenry: “this is par for the course with the current SEC—whether it’s inadequate public engagement, failure to comply with APA, attacking innovation, or issuing rules that exceed its authority, Chair Gensler’s SEC clearly thinks it’s above the law.” https://t.co/cxGfbnhg38', 'created_at': '2024-04-11T14:55:31+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1778436723912892611', 'quote_count': 1, 'reply_count': 1, 'retweet_count': 6, 'favorite_count': 21, 'choice': 'somewhat insightful', 'reason': "Comments on the SEC's actions and their impact on innovation."},

{'id': '1778436725783646643', 'text': '@PatrickMcHenry highlights recent comments re the @SECGov in Debt Box (“gross abuse of power”), @Ripple (“lack of faithful allegiance to the law”), &amp; @Grayscale (“arbitrary and capricious”). \n\n“Chair Gensler’s SEC is tarnishing the reputation of this very important institution.”', 'created_at': '2024-04-11T14:55:31+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1778436725783646643', 'quote_count': 0, 'reply_count': 0, 'retweet_count': 4, 'favorite_count': 9, 'choice': 'somewhat insightful', 'reason': 'Highlights criticisms of the SEC under Chair Gensler.'},

 {'id': '1777671041516552240', 'text': 'Things to watch for in today’s Senate Banking hearing with Treasury Deputy Secretary Wally Adeyemo:\n\n-Adeyemo calling on Chair Sherrod Brown to grant Treasury/FinCEN additional authority (potentially followed by Brown dropping a bill to do just that)\n\n-Warren bashing stablecoins and stablecoin bill, Lummis defending \n\n-Tillis, Hagerty, and other Rs lining up behind the new Tillis/Hagerty ENFORCE Act (which is focused on centralized actors) — as a potential alternative to leg like DAAMLA or CANSEE\n\nTL;DR: a lot of posturing, not a lot of action (with possible exception of stablecoins) this Congress', 'created_at': '2024-04-09T12:12:58+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1777671041516552240', 'quote_count': 1, 'reply_count': 1, 'retweet_count': 3, 'favorite_count': 31, 'choice': 'somewhat insightful', 'reason': 'Provides a preview of a Senate Banking hearing on crypto issues.'},

 {'id': '1777652757060554862', 'text': 'As a meta point, typically when Members have opinions about other Members’ bills, they share them directly or via staff, rather than through performative letters they “exclusively” give to @politico to run in a morning newsletter', 'created_at': '2024-04-09T11:00:19+00:00', 'username': 'AlexanderGrieve', 'url': 'https://x.com/AlexanderGrieve/status/1777652757060554862', 'quote_count': 0, 'reply_count': 1, 'retweet_count': 4, 'favorite_count': 37, 'choice': 'somewhat insightful', 'reason': 'Comments on the communication dynamics among Members.'}]



    # note this is the llm score, skipped integrity check and batch age score
    score = evaluator.llm_keyword_ranking_evaluation(args.query, ranked_docs)
    print("======LLM Score======")
    print(score)


if __name__ == "__main__":
    main()
