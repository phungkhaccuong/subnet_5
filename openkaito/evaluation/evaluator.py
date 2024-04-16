import json
import os
import random
from datetime import datetime, timezone
from traceback import print_exception

import bittensor as bt
import openai
import torch

from openkaito.protocol import SortType

from .utils import (
    ndcg_score,
    parse_llm_result,
    parse_llm_result_for_author_index,
    tweet_url_to_id,
)


class Evaluator:
    def __init__(self, llm_client, twitter_crawler=None) -> None:
        # for ranking results evaluation
        self.llm_client = llm_client

        # for integrity check
        self.twitter_crawler = twitter_crawler

        with open("twitter_usernames.txt", "r") as f:
            self.credit_twitter_author_usernames = set(f.read().strip().splitlines())
        bt.logging.info(
            f"loaded {len(self.credit_twitter_author_usernames)} credit_twitter_author_usernames"
        )

    def evaluate(self, query: bt.Synapse, responses: list):
        query_string = query.query_string
        size = query.size

        scores = torch.zeros(len(responses))

        zero_score_mask = torch.ones(len(responses))

        bt.logging.info(f"[CST] zero_score_mask start : {zero_score_mask}")

        rank_scores = torch.zeros(len(responses))

        bt.logging.info(f"[CST] rank_scores start : {rank_scores}")

        avg_ages = torch.zeros(len(responses))
        bt.logging.info(f"[CST] avg_ages start : {avg_ages}")
        avg_age_scores = torch.zeros(len(responses))
        uniqueness_scores = torch.zeros(len(responses))

        bt.logging.info(f"[CST] uniqueness_scores start : {uniqueness_scores}")
        credit_author_scores = torch.zeros(len(responses))
        bt.logging.info(f"[CST] credit_author_scores start : {credit_author_scores}")

        now = datetime.now(timezone.utc)
        max_avg_age = 0

        spot_check_id_dict = dict()
        # quick integrity check and get spot_check_id_dict
        utcnow = datetime.now(timezone.utc)
        for i, response in enumerate(responses):
            try:
                if response is None or not response or len(response) > size:
                    zero_score_mask[i] = 0
                    continue
                for doc in response:
                    doc_id = doc["id"]
                    url_id = tweet_url_to_id(doc["url"])
                    if doc_id != url_id:
                        bt.logging.info(
                            f"Document id {doc_id} not match url id {url_id}"
                        )
                        zero_score_mask[i] = 0
                        break
                    if datetime.fromisoformat(doc["created_at"].rstrip("Z")) > utcnow:
                        bt.logging.info(
                            f"created_at {doc['created_at']} is in the future"
                        )
                        zero_score_mask[i] = 0
                        break

                spot_check_id_dict[i] = random.choice(response)["id"]
                bt.logging.info(f"[CST] random.choice(response)['id']: {spot_check_id_dict[i]}")
            except Exception as e:
                bt.logging.error(
                    f"Error while intitial checking {i}-th response: {e}, 0 score"
                )
                bt.logging.debug(print_exception(type(e), e, e.__traceback__))
                zero_score_mask[i] = 0

        if self.twitter_crawler is not None:
            bt.logging.debug(f"[CST] spot_check_id_dict: {spot_check_id_dict}")
            bt.logging.debug(f"[CST] spot_check_id_dict.values(): {list(set(spot_check_id_dict.values()))}")
            groundtruth_docs = self.twitter_crawler.get_tweets_by_ids_with_retries(
                list(set(spot_check_id_dict.values())), retries=2
            )
            bt.logging.debug(f"[CST] groundtruth_docs: {groundtruth_docs}")
            groundtruth_check = len(groundtruth_docs) > 0
            if not groundtruth_check:
                bt.logging.warning(
                    "groundtruth_docs is empty, apify scraper is likely to be down, skipping check"
                )
        else:
            groundtruth_check = False
            bt.logging.warning(
                "Twitter crawler is not initialized. spot content check is skipped."
            )
        bt.logging.info(f"[CST] groundtruth_check: {groundtruth_check}")
        for i, response in enumerate(responses):
            try:
                if zero_score_mask[i] == 0:
                    continue

                bt.logging.trace(f"[CST] Processing {i}-th response")
                if groundtruth_check:
                    # the spot check doc did not get fetched
                    if spot_check_id_dict[i] not in groundtruth_docs:
                        bt.logging.info(
                            f"spot check id {spot_check_id_dict[i]} can not be fetched in groundtruth_docs"
                        )
                        zero_score_mask[i] = 0
                        continue

                    # check all docs against groundtruth, if fetched
                    for doc in response:
                        if doc["id"] in groundtruth_docs:
                            bt.logging.trace(f"[CST] Checking doc {doc['id']}")
                            if not self.check_document(
                                doc, groundtruth_docs[doc["id"]]
                            ):
                                zero_score_mask[i] = 0
                                break

                if query.name == "StructuredSearchSynapse":
                    # for author index task
                    # check if the response is from the request author list
                    if query.author_usernames is not None:
                        if not all(
                            doc["username"] in query.author_usernames
                            for doc in response
                        ):
                            zero_score_mask[i] = 0
                            continue

                    # check if the response is within the time range filter
                    if query.earlier_than_timestamp is not None:
                        if not all(
                            get_datetime(doc["created_at"]).timestamp()
                            < query.earlier_than_timestamp
                            for doc in response
                        ):
                            zero_score_mask[i] = 0
                            continue
                    if query.later_than_timestamp is not None:
                        if not all(
                            get_datetime(doc["created_at"]).timestamp()
                            > query.later_than_timestamp
                            for doc in response
                        ):
                            zero_score_mask[i] = 0
                            continue

                    bt.logging.debug(
                        f"[CST] Integrity check passed for {i}-th response: ", response
                    )

                id_set = set()
                credit_username_count = 0
                for doc in response:
                    bt.logging.info(f"[CST] for doc in response: {doc}")
                    avg_ages[i] += (
                        now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))
                    ).total_seconds()
                    bt.logging.info(f"[CST] avg_ages[i] +=: {avg_ages[i]}")
                    id_set.add(doc["id"])
                    if doc["username"] in self.credit_twitter_author_usernames:
                        credit_username_count += 1
                    bt.logging.info(f"[CST] credit_username_count += 1: {credit_username_count}")

                bt.logging.info(f"[CST] avg_ages[i] total ==: {avg_ages[i]}")
                avg_ages[i] /= len(response)
                bt.logging.info(f"[CST] avg_ages[i] /=: {avg_ages[i]}")
                max_avg_age = max(max_avg_age, avg_ages[i])

                bt.logging.info(f"[CST] max_avg_age=: {max_avg_age}")

                uniqueness_scores[i] = len(id_set) / size
                bt.logging.info(f"[CST] uniqueness_scores[i]=: {uniqueness_scores[i]}")
                credit_author_scores[i] = credit_username_count / size
                bt.logging.info(f"[CST] credit_author_scores[i]=: {credit_author_scores[i]}")
                # index author data task
                if (
                    query.name == "StructuredSearchSynapse"
                    and query.author_usernames is not None
                ):
                    bt.logging.info(f"[CST] HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                    bt.logging.info(f"[CST] llm_author_index_data_evaluation start")
                    llm_ranking_scores = self.llm_author_index_data_evaluation(response)
                    # mean quality score
                    bt.logging.info(f"[CST] llm_author_index_data_evaluation.llm_ranking_scores: {llm_ranking_scores}")
                    rank_scores[i] = sum(llm_ranking_scores) / len(llm_ranking_scores)
                    bt.logging.info(f"[CST] llm_author_index_data_evaluation.llm_ranking_scores rank_scores[i]: {rank_scores[i]}")
                else:
                    bt.logging.info(f"[CST] HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
                    bt.logging.info(f"[CST] llm_keyword_ranking_evaluation start")
                    llm_ranking_scores = self.llm_keyword_ranking_evaluation(
                        query_string, response
                    )
                    bt.logging.info(f"[CST] llm_keyword_ranking_evaluation.llm_ranking_scores: {llm_ranking_scores}")
                    rank_scores[i] = ndcg_score(llm_ranking_scores, size)
                    bt.logging.info(f"[CST] llm_keyword_ranking_evaluation.rank_scores[i]: {rank_scores[i]}")
                bt.logging.info(f"[CST] Quality score: {rank_scores[i]}")
            except Exception as e:
                bt.logging.error(f"Error while processing {i}-th response: {e}")
                bt.logging.debug(print_exception(type(e), e, e.__traceback__))
                zero_score_mask[i] = 0

        # age contribution to encourage recency
        bt.logging.info(f"[CST] avg_ages final: {avg_ages}")
        bt.logging.info(f"[CST] max_avg_age final: {max_avg_age}")
        avg_age_scores = 1 - (avg_ages / (max_avg_age + 1))
        bt.logging.info(f"[CST] avg_age_scores final: {avg_age_scores}")
        bt.logging.info(f"[CST] before compute scores avg_age_scores: {avg_age_scores}, rank_scores: {rank_scores}, credit_author_scores: {credit_author_scores}")
        scores = avg_age_scores * 0.2 + rank_scores * 0.7 + credit_author_scores * 0.1
        bt.logging.info(f"[CST] uniqueness_scores final: {uniqueness_scores}")
        scores = scores * uniqueness_scores
        bt.logging.info(f"[CST] scores final1: {scores}")
        # relative scores in a batch
        scores = scores / (scores.max() + 1e-5)
        bt.logging.info(f"[CST] scores final2 with scores.max: {scores.max()}: {scores}")
        return scores * zero_score_mask

    def check_document(self, doc, groundtruth_doc):
        """
        This function checks the integrity of the document.
        """
        try:
            check_fields = ["text", "username"]
            for field in check_fields:
                if doc[field] != groundtruth_doc[field]:
                    bt.logging.info(
                        f"Document {field} {doc[field]} does not match ground truth {groundtruth_doc[field]}"
                    )
                    return False
            if datetime.fromisoformat(
                doc["created_at"].rstrip("Z")
            ) != datetime.fromisoformat(groundtruth_doc["created_at"].rstrip("Z")):
                bt.logging.info(
                    f"Document created_at {doc['created_at']} does not match ground truth {groundtruth_doc['created_at']}"
                )
                return False
            return True
        except Exception as e:
            bt.logging.error(f"Error while checking integrity of document: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            return False

    def llm_keyword_ranking_evaluation(self, query_string, docs, retries=3):
        """
        This function evaluates the ranking of the documents using the LLM.
        """
        try:
            newline = "\n"
            prompt_docs = "\n\n".join(
                [
                    f"ItemId: {i}\nTime: {doc['created_at'].split('T')[0]}\nText: {doc['text'][:1000].replace(newline, '  ')}"
                    for i, doc in enumerate(docs)
                ]
            )
            bt.logging.info(
                f"[CST] prompt_docs = {prompt_docs}"
            )
            bt.logging.debug(
                f"[CST] Querying LLM of {query_string} with docs:\n" + prompt_docs
            )
            output = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """Below are the metrics and definations: 
outdated: Time-sensitive information that is no longer current or relevant.
off topic: Superficial content lacking depth and comprehensive insights.
somewhat relevant: Offers partial insight but lacks depth and comprehensive coverage.
relevant: Comprehensive, insightful content suitable for informed decision-making.""",
                    },
                    {
                        "role": "system",
                        "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
                    },
                    {
                        "role": "system",
                        "content": """
Example 1:
ItemId: 0
Time: "2023-11-25" 
Text: Also driving the charm is Blast's unique design: Depositors start earning yields on the transferred ether alongside BLAST points. "Blast natively participates in ETH staking, and the staking yield is passed back to the L2's users and dapps," the team said in a post Tuesday. 'We've redesigned the L2 from the ground up so that if you have 1 ETH in your wallet on Blast, over time, it grows to 1.04, 1.08, 1.12 ETH automatically."
As such, Blast is invite-only as of Tuesday, requiring a code from invited users to gain access. Besides, the BLAST points can be redeemed starting in May.Blast raised over $20 million in a round led by Paradigm and Standard Crypto and is headed by pseudonymous figurehead @PacmanBlur, one of the co-founders of NFT marketplace Blur.
@PacmanBlur said in a separate post that Blast was an extension of the Blur ecosystem, letting Blur users earn yields on idle assets while improving the technical aspects required to offer sophisticated NFT products to users.
BLUR prices rose 12%% in the past 24 hours following the release of Blast

Query: Blast

Output:
item_id: 0
choice: relevant
reason: It is relevant as it deep dives into the Blast project.

Example 2:
ItemId: 1
Time: "2023-11-15"
Text: To celebrate, we've teamed up with artist @debbietea8 to release a commemorative piece of art on @arbitrum! 😍
Now available for free, exclusively in app! 🥳

Query: Arbitrum

Output:
item_id: 1
choice: off topic
reason: It is not directly related to Arbitrum as it just uses the arbitrum app.
""",
                    },
                    {
                        "role": "user",
                        "content": f"You will be given a list of documents with id and you have to rate them based on the relevance to the query. The documents are as follows:\n"
                        + prompt_docs,
                    },
                    {
                        "role": "user",
                        "content": f"Use the metric choices [outdated, off topic, somewhat relevant, relevant] to evaluate the text toward '{query_string}'?",
                    },
                    {
                        "role": "user",
                        "content": "Must answer in JSON format of a list of choices with item ids for all the given items: "
                        + "{'results': [{'item_id': the item id of choice, e.g. 0, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }, {'item_id': 1, 'reason': explanation, 'choice': answer } , ... ] } ",
                    },
                ],
                temperature=0,
            )
            bt.logging.debug(f"[CST] LLM response: {output.choices[0].message.content}")
            bt.logging.debug(
                f"LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
            )
        except Exception as e:
            bt.logging.error(f"Error while querying LLM: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            return 0

        try:
            result = json.loads(output.choices[0].message.content)
            bt.logging.debug(f"[CST] LLM result: {result}")
            ranking = parse_llm_result(result)
            bt.logging.info(f"[CST] LLM ranking: {ranking}")
            if len(ranking) != len(docs):
                raise ValueError(
                    f"Length of ranking {len(ranking)} does not match input docs length {len(docs)}"
                )
            # ranking_score = ndcg_score(ranking, size)
            # bt.logging.info(f"LLM Ranking score: {ranking_score}")
            # return ranking_score
            return ranking
        except Exception as e:
            bt.logging.error(f"Error while parsing LLM result: {e}, retrying...")
            if retries > 0:
                return self.llm_keyword_ranking_evaluation(
                    query_string, docs, retries - 1
                )
            else:
                bt.logging.error(
                    f"Failed to parse LLM result after retrying. Returning [0]."
                )
            return [0]

    def llm_author_index_data_evaluation(self, docs, retries=3):
        if docs is None or len(docs) == 0:
            return [0]
        try:
            newline = "\n"
            prompt_docs = "\n\n".join(
                [
                    f"ItemId: {i}\nTime: {doc['created_at'].split('T')[0]}\nText: {doc['text'][:1000].replace(newline, '  ')}"
                    for i, doc in enumerate(docs)
                ]
            )

            bt.logging.debug(
                f"[CST] Querying LLM of author index data with docs:\n" + prompt_docs
            )
            output = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """Below are the metrics and definations: 
outdated: Time-sensitive information that is no longer current or relevant.
insightless: Superficial content lacking depth and comprehensive insights.
somewhat insightful: Offers partial insight but lacks depth and comprehensive coverage.
Insightful: Comprehensive, insightful content suitable for informed decision-making.""",
                    },
                    {
                        "role": "system",
                        "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
                    },
                    {
                        "role": "system",
                        "content": """
Example 1:
ItemId: 0
Time: "2023-11-25" 
Text: Also driving the charm is Blast's unique design: Depositors start earning yields on the transferred ether alongside BLAST points. "Blast natively participates in ETH staking, and the staking yield is passed back to the L2's users and dapps," the team said in a post Tuesday. 'We've redesigned the L2 from the ground up so that if you have 1 ETH in your wallet on Blast, over time, it grows to 1.04, 1.08, 1.12 ETH automatically."
As such, Blast is invite-only as of Tuesday, requiring a code from invited users to gain access. Besides, the BLAST points can be redeemed starting in May.Blast raised over $20 million in a round led by Paradigm and Standard Crypto and is headed by pseudonymous figurehead @PacmanBlur, one of the co-founders of NFT marketplace Blur.
@PacmanBlur said in a separate post that Blast was an extension of the Blur ecosystem, letting Blur users earn yields on idle assets while improving the technical aspects required to offer sophisticated NFT products to users.
BLUR prices rose 12%% in the past 24 hours following the release of Blast


Output:
item_id: 0
choice: insightful
reason: It is contains insightful information about the Blast project.

Example 2:
ItemId: 1
Time: "2024-03-19"
Text: $SLERF to the moon!
$BOME $SOL $MUMU $BONK $BOPE $WIF $NAP 🥳

Output:
item_id: 1
choice: insightless
reason: It does not contain much meaningful information, just sentiment about some tickers.
""",
                    },
                    {
                        "role": "user",
                        "content": f"You will be given a list of documents with id and you have to rate them based on its information and insightfulness. The documents are as follows:\n"
                        + prompt_docs,
                    },
                    {
                        "role": "user",
                        "content": f"Use the metric choices [outdated, insightless, somewhat insightful, insightful] to evaluate the text.",
                    },
                    {
                        "role": "user",
                        "content": "Must answer in JSON format of a list of choices with item ids for all the given items: "
                        + "{'results': [{'item_id': the item id of choice, e.g. 0, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }, {'item_id': 1, 'reason': explanation, 'choice': answer } , ... ] } ",
                    },
                ],
                temperature=0,
            )
            bt.logging.debug(
                f"[CST] LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
            )
        except Exception as e:
            bt.logging.error(f"Error while querying LLM: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            return 0

        try:
            result = json.loads(output.choices[0].message.content)
            bt.logging.debug(f"LLM result: {result}")
            ranking = parse_llm_result_for_author_index(result)
            bt.logging.info(f"LLM ranking: {ranking}")
            if len(ranking) != len(docs):
                raise ValueError(
                    f"Length of ranking {len(ranking)} does not match input docs length {len(docs)}"
                )
            return ranking
        except Exception as e:
            bt.logging.error(f"Error while parsing LLM result: {e}, retrying...")
            if retries > 0:
                return self.llm_author_index_data_evaluation(docs, retries - 1)
            else:
                bt.logging.error(
                    f"Failed to parse LLM result after retrying. Returning [0]."
                )
            return [0]


    def llm_author_index_data_evaluation_optimize(self, docs, retries=3):
        if docs is None or len(docs) == 0:
            return [0]
        try:
            newline = "\n"
            prompt_docs = "\n\n".join(
                [
                    f"ItemId: {i}\nTime: {doc['created_at'].split('T')[0]}\nText: {doc['text'][:1000].replace(newline, '  ')}"
                    for i, doc in enumerate(docs)
                ]
            )
            bt.logging.info(f"[CST] llm_author_index_data_evaluation.prompt_docs: {prompt_docs}")
            bt.logging.debug(
                f"[CST] Querying LLM of author index data with docs:\n" + prompt_docs
            )
            output = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """Below are the metrics and definations: 
outdated: Time-sensitive information that is no longer current or relevant.
insightless: Superficial content lacking depth and comprehensive insights.
somewhat insightful: Offers partial insight but lacks depth and comprehensive coverage.
Insightful: Comprehensive, insightful content suitable for informed decision-making.""",
                    },
                    {
                        "role": "system",
                        "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
                    },
                    {
                        "role": "system",
                        "content": """
Example 1:
ItemId: 0
Time: "2023-11-25" 
Text: Also driving the charm is Blast's unique design: Depositors start earning yields on the transferred ether alongside BLAST points. "Blast natively participates in ETH staking, and the staking yield is passed back to the L2's users and dapps," the team said in a post Tuesday. 'We've redesigned the L2 from the ground up so that if you have 1 ETH in your wallet on Blast, over time, it grows to 1.04, 1.08, 1.12 ETH automatically."
As such, Blast is invite-only as of Tuesday, requiring a code from invited users to gain access. Besides, the BLAST points can be redeemed starting in May.Blast raised over $20 million in a round led by Paradigm and Standard Crypto and is headed by pseudonymous figurehead @PacmanBlur, one of the co-founders of NFT marketplace Blur.
@PacmanBlur said in a separate post that Blast was an extension of the Blur ecosystem, letting Blur users earn yields on idle assets while improving the technical aspects required to offer sophisticated NFT products to users.
BLUR prices rose 12%% in the past 24 hours following the release of Blast


Output:
item_id: 0
choice: insightful
reason: It is contains insightful information about the Blast project.

Example 2:
ItemId: 1
Time: "2024-03-19"
Text: $SLERF to the moon!
$BOME $SOL $MUMU $BONK $BOPE $WIF $NAP 🥳

Output:
item_id: 1
choice: insightless
reason: It does not contain much meaningful information, just sentiment about some tickers.
""",
                    },
                    {
                        "role": "user",
                        "content": f"You will be given a list of documents with id and you have to rate them based on its information and insightfulness. The documents are as follows:\n"
                        + prompt_docs,
                    },
                    {
                        "role": "user",
                        "content": f"Use the metric choices [outdated, insightless, somewhat insightful, insightful] to evaluate the text.",
                    },
                    {
                        "role": "user",
                        "content": "Must answer in JSON format of a list of choices with item ids for all the given items: "
                        + "{'results': [{'item_id': the item id of choice, e.g. 0, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }, {'item_id': 1, 'reason': explanation, 'choice': answer } , ... ] } ",
                    },
                ],
                temperature=0,
            )
            bt.logging.debug(f"[CST] LLM response: {output.choices[0].message.content}")
            bt.logging.debug(
                f"[CST] LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
            )
        except Exception as e:
            bt.logging.error(f"Error while querying LLM: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            return 0

        try:
            result = json.loads(output.choices[0].message.content)
            result = result['results']
            print(f"LLM result: {result}")
            for i, doc in enumerate(docs):
                data = result[i]
                print(f"LLM data: {data}")
                doc['choice'] = data['choice']
                doc['reason'] = data['reason']
                print(f"[CST] final doc: {doc}")
            return docs
        except Exception as e:
            bt.logging.error(f"Error while parsing LLM result: {e}, retrying...")
            if retries > 0:
                return self.llm_author_index_data_evaluation(docs, retries - 1)
            else:
                bt.logging.error(
                    f"Failed to parse LLM result after retrying. Returning [0]."
                )
            return [0]


def get_datetime(time_str: str):
    return datetime.fromisoformat(time_str.rstrip("Z"))
