__author__ = 'Sun'

import numpy as np
import theano.tensor as T
from knowledge.language.core.word import Word

class WordEmbeddingLayer(object):

    def __init__(self, word_list, window_size, feature_num):

        padding_word = Word.padding_word()

        self._word_pos = dict()

        self._word_pos[padding_word.content] = 0

        pos = 1
        for word in word_list:
            if word.content != padding_word.content:
                self._word_pos[word] = pos
                pos += 1

        self._window_size = window_size

        self._embeddings = T.shared(np.random.random((len(self._word_pos), feature_num)))
        self._feature_dim = feature_num


    def get_output_size(self):

        return self._window_size * self._feature_dim


    def get_word_feature(self, word):

        pos = self._word_pos[word]
        return self._embeddings[pos]

    def get_sentence_features(self, sentence):

        word_windows = sentence.word_windows(self._window_size)

        window_features = []

        for window in word_windows:
            window_features.append(
                T.horizontal_stack([self.get_word_feature(word) for word in window])
            )

        self.output = T.vertical_stack(window_features)

    def get_words

    def update(self, sentence, delta):

        word_windows = sentence.word_windows(self._window_size)
        update = []
        for window_pos, window in enumerate(word_windows):
            cur_delta = delta[window_pos].reshape([self._window_size, self._feature_dim])

            for word_pos, word in window:
                update.append(T.inc_subtensor(self.get_word_feature(word), cur_delta[word_pos]))

        return update


