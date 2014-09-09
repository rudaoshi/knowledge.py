import theano
import theano.tensor as T
import numpy as np
import time
import sys
import os

from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression
from knowledge.machine.neuralnetwork.layer.srl_cov_layer import SrlConvLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.util.theano_util import shared_dataset

def test_foo():
    mini_batch_size = 1000
    word_num = 10000
    sent_size = 10
    word_feature_num = 20
    window_size = 5

    conv_hidden_feature_num = 3
    conv_window = 5

    rng = np.random.RandomState(1234)
    inputs = T.itensor3('inputs')

    wordvect = LookupTableLayer(inputs = inputs[:,:,0:1], table_size = word_num,
            window_size = window_size, feature_num = word_feature_num,
            reshp = (inputs.shape[0],inputs.shape[1],1,inputs.shape[2]*word_feature_num))

    conv_word = SrlConvLayer('conv_word',rng,wordvect.output,\
            conv_hidden_feature_num,conv_window,word_feature_num)

    print 'test_foo done'


