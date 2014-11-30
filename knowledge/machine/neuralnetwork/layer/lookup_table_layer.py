__author__ = 'Sun'

import theano
import numpy as np
import theano.tensor as T

import os

class LookupTableLayer(object):

    def __init__(self, embeddings = None, table_size = None, feature_num = None):

        self._table_size = table_size
        self._feature_num = feature_num

        if embeddings is not None :

            self._embeddings = theano.shared(embeddings.astype(theano.config.floatX),
                                            name = 'embeddings',
                                            borrow=True)
        elif  self._table_size is not None and self._feature_num is not None:
            self._embeddings = theano.shared(np.random.random((self._table_size, feature_num)).astype(theano.config.floatX),
                                            name = 'embeddings',
                                            borrow=True)

        else:
            self._embeddings = None


    def output(self, inputs, tensor_output = False):

        if inputs.ndim == 1:
            return self._embeddings[inputs]

        else:
            if not tensor_output:
                return self._embeddings[inputs].reshape((inputs.shape[0], inputs.shape[1] * self._feature_num))
            else:
                return self._embeddings[inputs]  # dimension = (input.shape[0], inputs.shape[1], self._feature_num)
        #self.output, self.update = theano.map(fn = lambda vec: self._embeddings[vec].flatten(), sequences = inputs, name='x_scan')
        #self.output = T.horizontal_stack([self._embeddings[idx] for idx in input] )

    @property
    def embeddings(self):
        return self._embeddings


    def params(self):
        return [self._embeddings]

    def __getstate__(self):

        state = dict()
        state['name'] = "lookup-table"
        state['embeddings'] = self._embeddings.get_value()

        return state

    def __setstate__(self, state):

        assert state['name'] == "lookup-table"

        self._table_size , self._feature_num = state['embeddings'].shape
        self._embeddings = theano.shared(state['embeddings'].astype(theano.config.floatX),
                name = 'embeddings',
                borrow=True)
