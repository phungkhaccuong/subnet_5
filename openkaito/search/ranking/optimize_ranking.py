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
        print(f"[RANKING]  starting..............................................")
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
        choice = self.get_choice_score(doc)
        age_score = self.age_score(age, max_age)
        author_score = self.get_author_score(doc)

        return 0.2 * length_score + 0.2 * choice + 0.2 * author_score + 0.2 * age_score

    def get_choice_score(self, doc):
        try:
            choice = doc['choice']
            if choice == 'insightful':
                return 0.62
            elif choice == 'somewhat insightful':
                return 0.3358
            elif choice == 'insightless':
                return 0.0876
            else:
                return 0.04395
        except Exception as e:
            return 0.0876

    def get_author_score(self, doc):
        author_scores = self.load_author_scores()
        founds = [author_score for author_score in author_scores if author_score['username'] == doc['username']]
        return 0 if len(founds) == 0 else float(founds[0]['score'])

    def load_author_scores(self):
        with open('author_score.csv', 'r') as file:
            reader = csv.DictReader(file)
            data_dict = [row for row in reader]
        return data_dict

    def get_length_score(self, doc):
        len_doc = len(doc['text'])
        if len_doc >= 244:
            return 0.5744
        elif (len_doc >= 150) and (len_doc < 244):
            return 0.3667
        elif (len_doc >= 139) and (len_doc < 150):
            return 0.264
        elif (len_doc >= 110) and (len_doc < 139):
            return 0.1936
        elif (len_doc >= 85) and (len_doc < 110):
            return 0.134
        elif (len_doc >= 65) and (len_doc < 85):
            return 0.072
        elif (len_doc >= 48) and (len_doc < 65):
            return 0.043
        elif (len_doc >= 30) and (len_doc < 48):
            return 0.023
        else:
            return 0.007

    def age_score(self, age, max_age):
        return 1 - (age / (max_age + 1))


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
