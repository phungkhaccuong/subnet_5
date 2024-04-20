import csv
import itertools
import math
from datetime import datetime, date, timezone

import bittensor as bt
import nltk

from openkaito.search.ranking import AbstractRankingModel

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class OptimizeRankingModel(AbstractRankingModel):
    def __init__(self, length_weight=0.4, age_weight=0.6):
        super().__init__()
        self.length_weight = length_weight
        self.age_weight = age_weight

    def rank(self, query, documents):
        print(f"[RANKING]  starting.........")
        now = datetime.now(timezone.utc)
        ages = [(now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))).total_seconds() for doc in documents]
        max_age = 1 if len(ages) == 0 else max(ages)

        check = self.check_to_switch(documents)
        if (check):
            print("HEEEEEEEEEEEEEEEEEEEEEEE")
            ranked_docs = sorted(
                documents,
                key=lambda doc: self.compute_score(query, doc, max_age, now),
                reverse=True,
            )
            return ranked_docs
        else:
            print("HIIIIIIIIIIIIIIIIIIIIIIII")
            ranked_docs = sorted(
                documents,
                key=lambda doc: self.compute_score_v1(query, doc, max_age, now),
                reverse=True,
            )
            return ranked_docs

    def check_to_switch(self, documents):
        unique_names = list({item["username"] for item in documents})
        scores = self.load_author_scores()
        scores_of_unique_names = [float(score["score"]) for score in scores if score["username"] in unique_names]
        print(f"LIST_SCORE::::{scores_of_unique_names}")
        return any(score >= 0.3 for score in scores_of_unique_names)

    def get_author_score_of(self, doc):
        author_scores = self.load_author_scores()
        founds = [author_score for author_score in author_scores if author_score['username'] == doc['username']]
        return 0 if len(founds) == 0 else float(founds[0]['score'])


    def compute_score(self, query, doc, max_age, now):
        #print(f"[DOC]::::::::::::::::{doc}:::::::::::::::::::::::::::::::::::::::::::::")
        age = (now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))).total_seconds()

        length_score = self.get_length_score(doc)
        print(f"length_score::::{length_score}")
        age_score = self.age_score(age, max_age)
        print(f"age_score::::{age_score}")
        author_score = self.get_author_score_of(doc)
        print(f"author_score::::{author_score}")
        print(f"RESULT::::{self.length_weight * length_score * author_score + self.age_weight * age_score}")
        return self.length_weight * length_score * author_score + self.age_weight * age_score

    def compute_score_v1(self, query, doc, max_age, now):
        #print(f"[DOC]::::::::::::::::{doc}:::::::::::::::::::::::::::::::::::::::::::::")
        age = (now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))).total_seconds()
        length_score = self.get_length_score_v1(doc)
        print(f"length_score::::{length_score}")
        age_score = self.age_score(age, max_age)
        print(f"age_score::::{age_score}")
        print(f"RESULT::::{self.length_weight * length_score + self.age_weight * age_score}")
        return self.length_weight * length_score + self.age_weight * age_score

    def get_author_score_of(self, doc):
        author_scores = self.load_author_scores()
        founds = [author_score for author_score in author_scores if author_score['username'] == doc['username']]
        return 0 if len(founds) == 0 else float(founds[0]['score'])

    def load_author_scores(self):
        with open('author_score.csv', 'r') as file:
            reader = csv.DictReader(file)
            data_dict = [row for row in reader]
        return data_dict

    def get_length_score_v1(self, doc):
        result = self.classify_doc(doc)
        print(f"len_doc_original:::{len(doc['text'])}")
        print(f"result:::::{result}")
        if '?' in doc['text']:
            return 0.1

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
    #     result = self.classify_doc(doc)
    #     print(f"len_doc_original:::{len(doc['text'])}")
    #     print(f"result:::::{result}")
    #     if '?' in doc['text']:
    #         return 0.1
    #
    #     if result is None:
    #         return 0.1
    #
    #     if result['flattened_words'] >= 60:
    #         return 1
    #     if result['flattened_words'] >= 50:
    #         return 0.75
    #     if result['flattened_words'] >= 40:
    #         return 0.5
    #     if result['flattened_words'] >= 30:
    #         return 0.25
    #     if result['flattened_words'] >= 20:
    #         return 0.15
    #     else:
    #         return 0

    def get_length_score(self, doc):
        result = self.classify_doc(doc)
        print(f"[compute_score_word]:{result}")

        if '?' in doc['text']:
            return 0.1

        if result is None:
            return 0.1

        if result['sentences'] >= 5:
            return 1
        if result['sentences'] >= 4:
            return 0.75
        if result['sentences'] >= 3:
            return 0.5
        if result['sentences'] >= 2:
            return 0.25
        else:
            return 0

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
        bt.logging.info(f"Text: {doc['text'][:1000].replace(newline, '  ')}")
        return doc['text'][:1000].replace(newline, '  ')

    def classify_doc(self, doc):
        try:
            clean_text = self.get_clean_doc(doc)
            return self.get_useful_words_score(clean_text)
        except Exception as e:
            bt.logging.info(f"Exception:{e}")
            return None


class OptimizeRankingModelV1(AbstractRankingModel):
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
