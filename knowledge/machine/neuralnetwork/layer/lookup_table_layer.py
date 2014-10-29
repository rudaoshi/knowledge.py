__author__ = 'Sun'

import theano
import numpy as np
import theano.tensor as T

import os

from knowledge.machine.neuralnetwork.layer.base_module import BaseModule

class LookupTableLayer(BaseModule):

    def __init__(self, table_size, feature_num, name, load = False, model_folder=None):
        super(LookupTableLayer,self).__init__(name)

        self._table_size = table_size
        self._feature_num = feature_num
        if not load:
            self._embeddings = theano.shared(np.random.random((self._table_size, feature_num)).astype(theano.config.floatX),
                    name = 'embeddings',
                    borrow=True)
        else:
            assert isinstance(model_folder,str)
            self.load(model_folder)

    def output(self, inputs, tensor_output = False):

        if not tensor_output:
            return self._embeddings[inputs].reshape((inputs.shape[0], inputs.shape[1] * self._feature_num))
        else:
            return self._embeddings[inputs]
        #self.output, self.update = theano.map(fn = lambda vec: self._embeddings[vec].flatten(), sequences = inputs, name='x_scan')
        #self.output = T.horizontal_stack([self._embeddings[idx] for idx in input] )

    @property
    def embeddings(self):
        return self._embeddings


    def params(self):
        return [self._embeddings]


    def load(self,model_folder):
        super(LookupTableLayer,self).load(model_folder)
        d = self.params_lst[0]
        self._embeddings = theano.shared(d,
                name = 'lookuptable',
                borrow=True)

class MultiLookupTableLayer(object):

    def __init__(self, inputs_lst,table_size_lst, feature_num_lst):

        self.lookup_size = len(inputs_lst)
        self._table_size_lst = table_size_lst
        self._feature_num_lst = feature_num_lst

        self._embeddings_lst = [theano.shared(
            np.random.random((self._table_size_lst[i], feature_num_lst[i])),dtype=theano.config.floatX)
                                for i in xrange(self.lookup_size)]

        self.output = [self._embeddings_lst[i][inputs_lst[i]].reshape((inputs_lst[i].shape[0], -1)) for i in xrange(self.lookup_size)]

    @property
    def embeddings(self):
        return self._embeddings_lst

    def get_output_size(self):

        return self._feature_num_lst

    def params(self):

        return [self._embeddings_lst]




