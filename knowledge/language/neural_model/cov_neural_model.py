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


    def __init__(self, inputs, word_num, window_size, w2vec_feature_num,word_pos_feature_num,verb_pos_feature_num,
                 hidden_layer_size, n_outs, L1_reg = 0.00, L2_reg = 0.0001,
                 numpy_rng = None, theano_rng=None, ):
        # inputs shpae: (batch_size,max_sentence_length+3,max_sentence_length)
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.w2vec_feature_num = w2vec_feature_num
        self.POS_feature_num = POS_feature_num
        self.wordpos_feature_num = wordpos_feature_num
        self.verbpos_feature_num = verbpos_feature_num

        self.conv_hidden_feature_num = conv_hidden_feature_num


        # we have 4 lookup tables here:
        # 1,word vector
        #   output shape: (batch size,1,max_sentence_length * word_vector_length)
        # 2,POS tag vector
        #   output shape: (batch size,1,max_sentence_length * POS_vector_length)
        # 3,verb position vector
        #   output shape: (batch size,1,max_sentence_length * verbpos_vector_length)
        # 4,word position vector
        #   output shape: (batch size,max_sentence_length,max_sentence_length * wordpos_vector_length)
        self.wordvect = LookupTableLayer(inputs = inputs[:,:,0], table_size = word_num,
                                                   window_size = window_size, feature_num = w2vec_feature_num)
        self.POSvect = LookupTableLayer(inputs = inputs[:,:,1], table_size = word_num,
                                                   window_size = window_size, feature_num = POS_feature_num)
        self.verbpos_vect = LookupTableLayer(inputs = inputs[:,:,2], table_size = word_num,
                                                   window_size = window_size, feature_num = verbpos_feature_num)
        self.wordpos_vect = LookupTableLayer(inputs = inputs[:,:,3:], table_size = word_num,
                                                   window_size = window_size, feature_num = wordpos_feature_num)

        self.conv_word = None
        self.conv_POS = None
        self.conv_verbpos = None
        self.conv_wordpos = None

        # conv_out shape: (batch_size,max_sentence_length,max_sentence_length,conv_hidden_feature_num)
        # the first max_sentence_length means each element of it is one prediction for that word
        # the second max_sentence_length means each element of it is one output of conv
        self.conv_word.out.dimshuffle(0,'x',1,2)
        self.conv_POS.out.dimshuffle(0,'x',1,2)
        self.conv_verbpos.out.dimshuffle(0,'x',1,2)
        self.conv_out = self.conv_word.out + self.conv_POS + self.conv_verbpos + self.conv_wordpos

        # max_out shape: (batch_size,max_sentence_length,conv_hidden_feature_num)
        self.max_out = None

        self.hidden_layer = HiddenLayer(rng=numpy_rng, input=self.conv_out,
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





