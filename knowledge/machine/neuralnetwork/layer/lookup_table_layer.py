__author__ = 'Sun'

import theano
import numpy as np
import theano.tensor as T

import os
from knowledge.machine.neuralnetwork.layer.layer import Layer

class LookupTableLayer(Layer):

    def __init__(self, table_size = None, feature_num = None, embeddings = None, ):

        if table_size is not None and feature_num is not None:

            self._table_size = table_size
            self._feature_num = feature_num

            if embeddings is None:
                embeddings = np.random.random((self._table_size, feature_num))
            else:
                assert self._table_size == embeddings.shape[0] and \
                    self._feature_num == embeddings.shape[1], \
                    "the size info is not match the size of given embedding"

        elif table_size is None and feature_num is None:

            if embeddings is None:
                raise Exception("neither the size info nor the embedding is given!")
            else:
                self._table_size , self._feature_num = embeddings.shape

        else:

            raise Exception("the size info must both be  provided or both not")

        self._embeddings = theano.shared(embeddings.astype(theano.config.floatX),
                                            borrow=True)


    def output(self, inputs, **kwargs): #, tensor_output = False):

        return self._embeddings[inputs]
        # if inputs.ndim == 1:
        #     return self._embeddings[inputs]
        #
        # else:
        #     if not tensor_output:
        #         return self._embeddings[inputs].reshape((inputs.shape[0], inputs.shape[1] * self._feature_num))
        #     else:
        #         return self._embeddings[inputs]  # dimension = (input.shape[0], inputs.shape[1], self._feature_num)
        #self.output, self.update = theano.map(fn = lambda vec: self._embeddings[vec].flatten(), sequences = inputs, name='x_scan')
        #self.output = T.horizontal_stack([self._embeddings[idx] for idx in input] )

    @property
    def embeddings(self):
        return self._embeddings


    def params(self):
        return [self._embeddings]

    def get_parameter_size(self):
        return self._feature_num * self._table_size

    def get_parameter(self):
        return self._embeddings.get_value(borrow=True).reshape((-1,))

    def set_parameter(self, parameter_vec):
        return self._embeddings.set_value(
            parameter_vec.reshape((self._table_size, self._feature_num))
        )

    def __getstate__(self):

        state = dict()
        state['type'] = "lookup-table"
        state['embeddings'] = self._embeddings.get_value()

        return state

    def __setstate__(self, state):

        assert state['type'] == "lookup-table"

        self._table_size , self._feature_num = state['embeddings'].shape
        self._embeddings = theano.shared(state['embeddings'].astype(theano.config.floatX),
                borrow=True)
