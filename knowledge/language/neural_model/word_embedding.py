__author__ = 'Sun'

import numpy as np

from knowledge.language.core.word import Word

class WordEmbedding(object):

    def __init__(self, word_list, window_size):

        padding_word = Word.padding_word()

        self._word_pos = dict()

        self._word_pos[padding_word.content] = 0

        pos = 1
        for word in word_list:
            if word.content != padding_word.content:
                self._word_pos[word] = pos
                pos += 1

        self._window_size = window_size
        self._embeddings = None


    def create_random_embeddings(self, feature_num):

        self._embeddings = np.random.random((len(self._word_pos), feature_num))


    def get_word_feature(self, word):

        pos = self._word_pos[word]
        return self._embeddings[pos]

    def get_sentence_features(self, sentence):

        word_windows = sentence.word_windows(self._window_size)

        window_features = []

        for window in word_windows:
            window_features.append(
                np.hstack([self.get_word_feature(word) for word in window ])
            )

        return np.vstack(window_features)


