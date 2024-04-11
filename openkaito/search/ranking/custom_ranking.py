import itertools
import math
from datetime import datetime, date, timezone

import bittensor as bt
import nltk

from openkaito.search.ranking import AbstractRankingModel

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class HeuristicRankingModelV2(AbstractRankingModel):
    def __init__(self, length_weight=0.4, age_weight=0.6):
        super().__init__()
        self.length_weight = length_weight
        self.age_weight = age_weight

    def rank(self, query, documents):
        bt.logging.info(f"[RANK]  start")
        ranked_docs = sorted(
            documents,
            key=lambda doc: self.compute_score(query, doc),
            reverse=True,
        )
        return ranked_docs

    def compute_score(self, query, doc):
        if datetime.fromisoformat(doc["created_at"].rstrip("Z")).date() < date(
                2024, 1, 1):
            bt.logging.info(f"OUT_DATE:::{0.01}")
            return 0.01
        result = self.get_amend(doc)
        bt.logging.info(f"[compute_score]:{result}")
        if (result is not None) and (
                (result['flattened_words'] < 21) or (result['sentences'] < 2)):
            bt.logging.info(
                f"OPTIMIZE:::{0.01 + (result['flattened_words'] / 10000)}")
            return 0.01 + (result['flattened_words'] / 10000)
        else:
            now = datetime.now(timezone.utc)
            text_length = len(doc["text"])
            age = (
                    now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))
            ).total_seconds()
            bt.logging.info(
                f"DEFAULT:{self.length_weight * self.text_length_score(text_length) + self.age_weight * self.age_score(age)}")
            return self.length_weight * self.text_length_score(
                text_length
            ) + self.age_weight * self.age_score(age)

    def text_length_score(self, text_length):
        return math.log(text_length + 1) / 10

    def age_score(self, age):
        return 60 * 60 / (age + 1)

    def filter_useful_words(self, words):
        pos_tags = nltk.pos_tag(words)
        useful_words = []
        for pos_tag in pos_tags:
            word = pos_tag[0]
            tag = pos_tag[1]
            if tag[:2] in ['NN', 'JJ', 'VB', 'RB', 'PR']:
                useful_words.append(word)
        return useful_words

    def get_sentence_score(self, text):
        sentences = nltk.sent_tokenize(text)
        return len(sentences)

    def get_useful_words_score(self, text):
        sentences = nltk.sent_tokenize(text)
        words = [nltk.word_tokenize(sent) for sent in sentences]
        useful_words = [self.filter_useful_words(word) for word in words]
        flattened_useful_words = list(
            itertools.chain.from_iterable(useful_words))
        flattened_words = list(itertools.chain.from_iterable(words))
        return {"flattened_words": len(flattened_words),
                "flattened_useful_words": len(flattened_useful_words),
                "sentences": len(sentences)}

    def get_clean_doc(self, doc):
        newline = "\n"
        bt.logging.info(f"DOC: {doc}")
        bt.logging.info(f"Text: {doc['text'][:1000].replace(newline, '  ')}")
        return doc['text'][:1000].replace(newline, '  ')

    def get_amend(self, doc):
        try:
            clean_text = self.get_clean_doc(doc)
            return self.get_useful_words_score(clean_text)
        except Exception as e:
            bt.logging.info(f"Exception:{e}")
            return None


class CustomRankingModel(AbstractRankingModel):
    def __init__(self, length_weight=0.77, age_weight=0.23):
        super().__init__()
        self.length_weight = length_weight
        self.age_weight = age_weight

    def rank(self, query, documents):
        now = datetime.now(timezone.utc)
        ages = [(
                        now - datetime.fromisoformat(
                    doc["created_at"].rstrip("Z"))
                ).total_seconds() for doc in documents]
        max_age = 0 if len(ages) == 0 else max(ages)

        ranked_docs = sorted(
            documents,
            key=lambda doc: self.compute_score(query, doc, max_age, now),
            reverse=True,
        )
        return ranked_docs

    def compute_score(self, query, doc, max_age, now):
        age = (
                now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))
        ).total_seconds()

        llm_score = self.llm_score(doc)
        age_score = self.age_score(age, max_age)
        length_score = self.length_score(doc)

        return self.length_weight * length_score + self.age_weight * age_score

    def llm_score(self, doc):
        choice_mapping = {
            "outdated": 0,
            "insightless": 0,
            "somewhat insightful": 0.5,
            "insightful": 1,
        }

        choice = doc.get('choice')
        if choice is None:
            return 0
        return choice_mapping[choice];

    def length_score(self, doc):
        if len(doc['text']) > 200:
            return 1
        elif len(doc['text']) > 150:
            return 0.75
        elif len(doc['text']) > 100:
            return 0.5
        elif len(doc['text']) > 75:
            return 0.25
        else:
            return 0

    def age_score(self, age, max_age):
        return 1 - (age / (max_age + 1))