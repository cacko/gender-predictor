import nltk as _nltk
import random as _random
import pickle as _pickle
import collections as _collections
from pathlib import Path


class GenderPredictor():
    def __init__(self):
        counts = _collections.Counter()
        self.feature_set = []
        data_path = Path(__file__).parent / "names.pickle"
        data = _pickle.loads(data_path.read_bytes())
        for name_results in data:
            name, male_counts, female_counts = name_results

            if male_counts == female_counts:
                continue

            features = self._name_features(name)
            gender = 'M' if male_counts > female_counts else 'F'
            counts.update([gender])

            m_prob = male_counts / sum([male_counts, female_counts])
            m_prob = 0.01 if m_prob == 0 else 0.99 if m_prob == 1 else m_prob

            features['m_prob'] = m_prob
            features['f_prob'] = 1 - m_prob
            self.feature_set.append((features, gender))
        self.train_and_test()

    def classify(self, name):
        return(self.classifier.classify(self._name_features(name.upper())))

    def train_and_test(self, percent_to_train=0.80):
        _random.shuffle(self.feature_set)
        partition = int(len(self.feature_set) * percent_to_train)
        train = self.feature_set[:partition]
        self.classifier = _nltk.NaiveBayesClassifier.train(train)

    def _name_features(self, name):
        return({
            'last_is_vowel': (name[-1] in 'AEIOUY'),
            'last_letter': name[-1],
            'last_three': name[-3:],
            'last_two': name[-2:]})
