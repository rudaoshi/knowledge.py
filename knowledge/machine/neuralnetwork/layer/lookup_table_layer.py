__author__ = 'Sun'

import theano
import numpy as np
import theano.tensor as T


class LookupTableLayer(object):

    def __init__(self, input, table_size, window_size, feature_num):

        self._table_size = table_size
        self._feature_num = feature_num
        self._window_size = window_size

        self._embeddings = theano.shared(np.random.random((self._table_size, feature_num)))


        self.output, self.update = theano.map(fn = lambda vec: self._embeddings[vec].flatten(), sequences = input, name='x_scan')
        #self.output = T.horizontal_stack([self._embeddings[idx] for idx in input] )

    @property
    def embeddings(self):
        return self._embeddings

    def get_output_size(self):

        return self._window_size * self._feature_num

    def params(self):

        return [self._embeddings]




