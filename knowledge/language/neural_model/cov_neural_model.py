__author__ = 'huang'

import theano
import theano.tensor as T
import numpy
import time
import sys
import os

from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import MultiLookupTableLayer
from knowledge.util.theano_util import shared_dataset

class SrlNeuralLanguageModelCore(object):


    def __init__(self, word_ids, word_num, window_size, w2vec_feature_num,word_pos_feature_num,verb_pos_feature_num,
                 hidden_layer_size, n_outs, L1_reg = 0.00, L2_reg = 0.0001,
                 numpy_rng = None, theano_rng=None, ):
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.w2vec_feature_num = w2vec_feature_num
        self.word_pos_feature_num = word_pos_feature_num
        self.verb_pos_feature_num = verb_pos_feature_num
        self.feature_num = self.w2vec_feature_num + self.word_pos_feature_num + self.verb_pos_feature_num


        self.lookup_table_layer = MultiLookupTableLayer(inputs = word_ids, table_size = word_num,
                                                   window_size = window_size, feature_num = feature_num)

        self.hidden_layer = HiddenLayer(rng=numpy_rng, input=self.lookup_table_layer.output,
                                       n_in = self.lookup_table_layer.get_output_size(),
                                       n_out = hidden_layer_size,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.output_layer = LogisticRegression(
                                        input=self.hidden_layer.output,
                                        n_in=hidden_layer_size,
                                        n_out=n_outs)


        self.errors = self.output_layer.errors





