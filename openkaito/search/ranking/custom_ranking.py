import csv
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
        print(f"[RANKING]  starting.:::::::::::::::::::::::::::::::::::::::::")
        now = datetime.now(timezone.utc)
        ages = [(now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))).total_seconds() for doc in documents]
        max_age = 1 if len(ages) == 0 else max(ages)
        ranked_docs = sorted(
            documents,
            key=lambda doc: self.compute_score(query, doc, max_age, now),
            reverse=True,
        )
        return ranked_docs

    def compute_score(self, query, doc, max_age, now):
        age = (now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))).total_seconds()

        length_score = self.get_length_score(doc)
        print(f"length_score::::{length_score}")
        age_score = self.age_score(age, max_age)
        print(f"age_score::::{age_score}")
        author_score = self.get_author_score_of(doc)
        print(f"author_score::::{author_score}")
        return self.length_weight * length_score * author_score + self.age_weight * age_score

    def get_author_score_of(self, doc):
        author_scores = self.load_author_scores()
        founds = [author_score for author_score in author_scores if author_score['username'] == doc['username']]
        return 0 if len(founds) == 0 else founds[0]['score']

    def load_author_scores(self):
        with open('author_score.csv', 'r') as file:
            reader = csv.DictReader(file)
            data_dict = [row for row in reader]
        return data_dict

    def get_length_score(self, doc):
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

    # def get_length_score(self, doc):
    #     result = self.get_amend(doc)
    #     print(f"[compute_score_word]:{result}")
    #
    #     if result is None:
    #         return 0.1
    #
    #     if result['flattened_words'] >= 50:
    #         return 1
    #     if result['flattened_words'] >= 40:
    #         return 0.75
    #     if result['flattened_words'] >= 30:
    #         return 0.5
    #     if result['flattened_words'] >= 20:
    #         return 0.25
    #     else:
    #         return 0

    # def compute_score1(self, query, doc):
    #     now = datetime.now(timezone.utc)
    #     age = (now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))).total_seconds()
    #     if datetime.fromisoformat(doc["created_at"].rstrip("Z")).date() < date(2024, 1, 20):
    #         print(f"OUT_DATE:::{0.01 + (1 / age)}")
    #         return 0.01 + (1 / age)
    #
    #     result = self.get_amend(doc)
    #     print(f"[compute_score_word]:{result}")
    #
    #     if result is None:
    #         bt.logging.info(f"[NONE_DATA]:")
    #         return 0.1
    #
    #     if result['flattened_words'] >= 60:
    #         print(f"[compute_score 0.5]:::: age :{age} : re::: {0.5 + (1 / age)}")
    #         return 0.5 + (1 / age)
    #     if result['flattened_words'] >= 50:
    #         print(f"[compute_score 0.4]:::: age :{age} : re::: {0.4 + (1 / age)}")
    #         return 0.4 + (1 / age)
    #     if result['flattened_words'] >= 40:
    #         print(f"[compute_score 0.3]:::: age :{age} : re::: {0.3 + (1 / age)}")
    #         return 0.3 + (1 / age)
    #     if (result['flattened_words'] >= 30):
    #         print(f"[compute_score 0.25]:::: age :{age} : re::: {0.25 + (1 / age)}")
    #         return 0.25 + (1 / age)
    #     else:
    #         print(f"[compute_score 0.1]:::: age :{age} : re::: {0.1 + (1 / age)}")
    #         return 0.1 + (1 / age)

    # note: co nhunng cau 1 tu nhung ko co y nghia
    # chu y nhung co 1 cau nhung so tu dai

    def text_length_score(self, text_length):
        return math.log(text_length + 1) / 10

    def age_score(self, age, max_age):
        return 1 - (age / (max_age + 1))

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
        print(f"DOC::::::::::::: {doc} ::::::::::::::::::")
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
        ages = [(now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))).total_seconds() for doc in documents]
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

        age_score = self.age_score(age, max_age)
        length_score = self.length_score(doc)

        return self.length_weight * length_score + self.age_weight * age_score

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
