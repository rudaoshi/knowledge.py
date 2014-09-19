import theano
import theano.tensor as T
import numpy as np
import time
import sys
import os

from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression
from knowledge.machine.neuralnetwork.layer.conv_layer import SrlConvLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.util.theano_util import shared_dataset

def test_foo():
    mini_batch_size = 1000
    word_num = 10000
    sent_size = 30
    word_feature_num = 20
    wordpos_feature_num = 20
    window_size = 5

    conv_hidden_feature_num = 3
    conv_window = 7

    rng = np.random.RandomState(1234)
    inputs = T.itensor3('inputs')

    # inputs shpae: (batch_size,max_sentence_length+3,max_sentence_length)
    wordvect = LookupTableLayer(inputs = inputs[:,0:1,:], table_size = word_num,
            window_size = window_size, feature_num = word_feature_num,
            reshp = (inputs.shape[0],1,1,inputs.shape[2]*word_feature_num))


    wordpos_vect = LookupTableLayer(inputs = inputs[:,3:,:], table_size = word_num,
           window_size = window_size, feature_num = wordpos_feature_num,
           reshp = (inputs.shape[0],inputs.shape[1]-3,1,inputs.shape[2]*wordpos_feature_num))


    dinputs = np.random.randint(low=0,high=19,size=(mini_batch_size,sent_size+3,sent_size)).astype(np.int32)
    foo1a = theano.function(inputs=[inputs],outputs=wordvect.output)
    outputs1a = foo1a(dinputs)
    print 'test lookup, outputs1a shape',outputs1a.shape
    foo1b = theano.function(inputs=[inputs],outputs=wordpos_vect.output)
    outputs1b = foo1b(dinputs)
    print 'test lookup, outputs1b shape',outputs1b.shape

    conv_word = SrlConvLayer('conv_word',rng,wordvect.output,\
            conv_hidden_feature_num,1,conv_window,word_feature_num)
    foo2a = theano.function(inputs=[inputs],outputs = conv_word.out)
    outputs2a = foo2a(dinputs)
    print 'test conv1, outputs2a shape',outputs2a.shape
    conv_wordpos = SrlConvLayer('conv_wordpos',rng,wordpos_vect.output,\
            conv_hidden_feature_num,sent_size,conv_window,wordpos_feature_num)
    foo2b = theano.function(inputs=[inputs],outputs = conv_wordpos.output)
    outputs2b = foo2b(dinputs)
    print 'test conv1, outputs2b shape',outputs2b.shape

    conv_out = conv_word.out  + conv_wordpos.output
    foo3a = theano.function(inputs=[inputs],outputs = conv_out)
    outputs3a = foo3a(dinputs)
    print 'test conv, outputs3a shape',outputs3a.shape
    print 'test_foo done'


