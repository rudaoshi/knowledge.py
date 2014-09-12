__author__ = 'huang'

from theano.tensor.signal import downsample
import theano.tensor as T
import theano
import numpy
import time
import sys
import os

from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression
from knowledge.machine.neuralnetwork.layer.conv_layer import SrlConvLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.util.theano_util import shared_dataset

class SrlNeuralLanguageModelCore(object):


    def __init__(self,rng,inputs, word_num, window_size, word_feature_num,word_pos_feature_num,verb_pos_feature_num,
                 hidden_layer_size, n_outs, L1_reg = 0.00, L2_reg = 0.0001,
                 numpy_rng = None, theano_rng=None, ):
        # inputs shpae: (batch_size,max_sentence_length+3,max_sentence_length)
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.word_num = word_num
        self.POS_num = POS_num
        self.verbpos_num = verbpos_num
        self.wordpos_num = wordpos_num

        self.word_feature_num = word_feature_num
        self.POS_feature_num = POS_feature_num
        self.wordpos_feature_num = wordpos_feature_num
        self.verbpos_feature_num = verbpos_feature_num

        self.conv_window = conv_window
        self.conv_hidden_feature_num = conv_hidden_feature_num

        self.hidden_layer_size = hidden_layer_size

        # we have 4 lookup tables here:
        # 1,word vector
        #   output shape: (batch size,1,max_sentence_length * word_vector_length)
        # 2,POS tag vector
        #   output shape: (batch size,1,max_sentence_length * POS_vector_length)
        # 3,verb position vector
        #   output shape: (batch size,1,max_sentence_length * verbpos_vector_length)
        # 4,word position vector
        #   output shape: (batch size,max_sentence_length,max_sentence_length * wordpos_vector_length)
        self.wordvect = LookupTableLayer(inputs = inputs[:,0:1,:], table_size = word_num,
                                                   window_size = window_size, feature_num = word_feature_num,
                                                   reshp = (inputs.shape[0],1,1,inputs.shape[2]*word_feature_num))
        self.POSvect = LookupTableLayer(inputs = inputs[:,1:2,:], table_size = word_num,
                                                   window_size = window_size, feature_num = POS_feature_num,
                                                   reshp = (inputs.shape[0],1,1,inputs.shape[2]*POS_feature_num))
        self.verbpos_vect = LookupTableLayer(inputs = inputs[:,2:3,:], table_size = word_num,
                                                   window_size = window_size, feature_num = verbpos_feature_num,
                                                   reshp = (inputs.shape[0],1,1,inputs.shape[2]*verbpos_feature_num))
        self.wordpos_vect = LookupTableLayer(inputs = inputs[:,3:,:], table_size = word_num,
                                                   window_size = window_size, feature_num = wordpos_feature_num,
                                                   reshp = (inputs.shape[0],inputs.shape[1],1,inputs.shape[2]*wordpos_feature_num))

        #name,rng,inputs, hiden_size,window_size, feature_num_lst,feature_map_size=None,init_W=None,init_b=None):
        # conv_word.out.shape = (batch_size,1,conv_hidden_feature_num,max_sentence_length)
        # conv_POS.out.shape = (batch_size,1,conv_hidden_feature_num,max_sentence_length)
        # conv_verbpos.out.shape = (batch_size,1,conv_hidden_feature_num,max_sentence_length)
        # conv_wordpos.out.shape = (batch_size,max_sentence_length,conv_hidden_feature_num,max_sentence_length)
        # note. all output above have been seted 'dimshuffle'
        self.conv_word = SrlConvLayer('conv_word',rng,self.wordvect.output,\
                self.conv_hidden_feature_num,1,self.conv_window,self.word_feature_num)
        self.conv_POS = SrlConvLayer('conv_POS',rng,self.POSvect.output,\
                self.conv_hidden_feature_num,1,self.conv_window,self.POS_feature_num)
        self.conv_verbpos = SrlConvLayer('conv_verbpos',rng,self.verbpos_vect.output,\
                self.conv_hidden_feature_num,1,self.conv_window,self.verbpos_feature_num)
        self.conv_wordpos = SrlConvLayer('conv_wordpos',rng,self.wordpos_vect.output,\
                self.conv_hidden_feature_num,max_sentence_length,self.conv_window,self.wordpos_feature_num)


        # the first max_sentence_length means each element of it is one prediction for that word
        # the second max_sentence_length means each element of it is one output of conv
        # conv_out shape: (batch_size,max_sentence_length,conv_hidden_feature_num,max_sentence_length)
        self.conv_out = self.conv_word.out + self.conv_POS + self.conv_verbpos + self.conv_wordpos
        self.conv_out = self.conv_out.dimshuffle(1,0,2,3,4).reshape(inputs.shape[0],inputs.shape[1]-3,conv_hidden_feature_num,-1)

        # max_out shape: (batch_size,max_sentence_length,conv_hidden_feature_num)
        self.max_out = T.max(self.conv_out,axis=3).reshape((self.conv_out.shape[0],))


        # hidden layer
        self.hidden_layer = HiddenLayer(rng=rng, input=self.max_out,
                                       n_in = self.conv_hidden_feature_num,
                                       n_out = hidden_layer_size,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer




