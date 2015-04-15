__author__ = 'Sun'

import theano
import numpy as np
import theano.tensor as T

import os
from knowledge.machine.neuralnetwork.layer.layer import Layer

class MultiLookupTableLayer(Layer):

    def __init__(self, embeddings_sizes = None, embeddings_list = None, ):

        if embeddings_list is None and embeddings_sizes is None:
            raise Exception("neither size nor embeddings is provided")
        elif embeddings_list is None:
            embeddings_list = []
            for size in embeddings_sizes:
                embeddings_list.append(np.random.random(size))

        elif embeddings_sizes is None:
            embeddings_sizes = []
            for embeddings in embeddings_list:
                embeddings_sizes.append(embeddings.shape)

        else:
            assert len(embeddings_sizes) == len(embeddings_list), \
                "The element num of sizes and embeddings_list is not equal"

            for idx, size in enumerate(embeddings_sizes):
                assert embeddings_list[idx].shape == size, \
                    "The size info is not equal to the shape of embeddings @%d"%idx

        self.embeddings_sizes = embeddings_sizes
        self.embeddings_list = []

        for embeddings in embeddings_list:
            self.embeddings_list.append(
                theano.shared(embeddings.astype(theano.config.floatX),
                borrow=True)
            )


    def output(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        return self._embeddings[inputs]

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
