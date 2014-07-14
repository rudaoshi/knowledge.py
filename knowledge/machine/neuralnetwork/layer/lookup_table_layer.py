__author__ = 'Sun'

import numpy as np
import theano.tensor as T
from knowledge.language.core.word import Word

class LookupTableLayer(object):

    def __init__(self, table_size, window_size, feature_num):

        self._table_size = table_size
        self._feature_dim = feature_num
        self._window_size = window_size

        self._embeddings = T.shared(np.random.random((self._table_size, feature_num)))

    @property
    def embeddings(self):
        return self._embeddings

    def get_output_size(self):

        return self._window_size * self._feature_dim

    def output(self, item_idx):

        return  T.horizontal_stack([self._embeddings[idx] for idx in item_idx])

    def params(self):

        return [self._embeddings]




