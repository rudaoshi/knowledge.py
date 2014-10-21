__author__ = 'Sun'

import theano
import numpy as np
import theano.tensor as T


class LookupTableLayer(object):

    def __init__(self, table_size, feature_num):

        self._table_size = table_size
        self._feature_num = feature_num

        #self._embeddings = theano.shared(np.random.random((self._table_size, feature_num)))
        self._embeddings = theano.shared(np.random.random((self._table_size, feature_num)).astype(theano.config.floatX))

    def output(self, inputs, tensor_output = False):

        if not tensor_output:
            return self._embeddings[inputs].reshape((inputs.shape[0], -1))
        else:
            return self._embeddings[inputs]
        #self.output, self.update = theano.map(fn = lambda vec: self._embeddings[vec].flatten(), sequences = inputs, name='x_scan')
        #self.output = T.horizontal_stack([self._embeddings[idx] for idx in input] )

    @property
    def embeddings(self):
        return self._embeddings


    def params(self):

        return [self._embeddings]



class MultiLookupTableLayer(object):

    def __init__(self, inputs_lst,table_size_lst, feature_num_lst):

        self.lookup_size = len(inputs_lst)
        self._table_size_lst = table_size_lst
        self._feature_num_lst = feature_num_lst

        self._embeddings_lst = [theano.shared(np.random.random((self._table_size_lst[i], feature_num_lst[i]))) for i in xrange(self.lookup_size)]

        self.output = [self._embeddings_lst[i][inputs_lst[i]].reshape((inputs_lst[i].shape[0], -1)) for i in xrange(self.lookup_size)]

    @property
    def embeddings(self):
        return self._embeddings_lst

    def get_output_size(self):

        return self._feature_num_lst

    def params(self):

        return [self._embeddings_lst]




