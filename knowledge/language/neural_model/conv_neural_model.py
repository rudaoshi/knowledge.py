__author__ = 'huang'

import theano.tensor as T
import theano
import numpy
import time
import sys
import os

from knowledge.language.neural_model.sentence_level_log_likelihood_layer import SentenceLevelLogLikelihoodLayer
from knowledge.language.neural_model.problem.srl_problem import SrlProblem
from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression
from knowledge.machine.neuralnetwork.layer.conv_layer import SrlConvLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.util.theano_util import shared_dataset

class SrlNeuralLanguageModelCore(object):


    def __init__(self,rng,inputs,sent_length, y,masks, word_num, window_size, word_feature_num,word_pos_feature_num,verb_pos_feature_num,
                 hidden_layer_size, n_outs, L1_reg = 0.00, L2_reg = 0.0001,
                 numpy_rng = None, theano_rng=None, ):
        # inputs shpae: (batch_size,max_term_per_sent+3,max_sentence_length)
        # ,where  max_sentence_length = max_sentence_length + window_size - 1
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.max_sentence_length = max_sentence_length
        self.window_size = window_size
        self.max_term_per_sent = self.max_sentence_length - window_size + 1
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
        self.wordvec = LookupTableLayer(inputs = inputs[:,0:1,:], table_size = word_num,
                                                   window_size = window_size, feature_num = word_feature_num,
                                                   reshp = (inputs.shape[0],1,1,inputs.shape[2]*word_feature_num))
        self.POSvec = LookupTableLayer(inputs = inputs[:,1:2,:], table_size = word_num,
                                                   window_size = window_size, feature_num = POS_feature_num,
                                                   reshp = (inputs.shape[0],1,1,inputs.shape[2]*POS_feature_num))
        self.verbpos_vec = LookupTableLayer(inputs = inputs[:,2:3,:], table_size = word_num,
                                                   window_size = window_size, feature_num = verbpos_feature_num,
                                                   reshp = (inputs.shape[0],1,1,inputs.shape[2]*verbpos_feature_num))
        self.wordpos_vec = LookupTableLayer(inputs = inputs[:,3:,:], table_size = word_num,
                                                   window_size = window_size, feature_num = wordpos_feature_num,
                                                   reshp = (inputs.shape[0],inputs.shape[2],1,inputs.shape[2]*wordpos_feature_num))

        # conv_word.out.shape = (batch_size,1,conv_hidden_feature_num,max_sentence_length-conv_window+1)
        # conv_POS.out.shape = (batch_size,1,conv_hidden_feature_num,max_sentence_length-conv_window+1)
        # conv_verbpos.out.shape = (batch_size,1,conv_hidden_feature_num,max_sentence_length-conv_window+1)
        # conv_wordpos.out.shape = (batch_size,max_sentence_length,conv_hidden_feature_num,max_sentence_length-conv_window+1)
        # note. all output above have been seted 'dimshuffle'
        self.conv_word = SrlConvLayer('conv_word',rng,self.wordvec.output,\
                self.conv_hidden_feature_num,1,self.conv_window,self.word_feature_num)
        self.conv_POS = SrlConvLayer('conv_POS',rng,self.POSvec.output,\
                self.conv_hidden_feature_num,1,self.conv_window,self.POS_feature_num)
        self.conv_verbpos = SrlConvLayer('conv_verbpos',rng,self.verbpos_vec.output,\
                self.conv_hidden_feature_num,1,self.conv_window,self.verbpos_feature_num)
        self.conv_wordpos = SrlConvLayer('conv_wordpos',rng,self.wordpos_vec.output,\
                self.conv_hidden_feature_num,max_sentence_length,self.conv_window,self.wordpos_feature_num)


        # the first max_sentence_length means each element of it is one prediction for that word
        # the second max_sentence_length means each element of it is one output of conv
        # conv_out shape: (batch_size,max_sentence_length,conv_hidden_feature_num,max_sentence_length-conv_window+1)
        self.conv_out = self.conv_word.output + self.conv_POS.output + self.conv_verbpos.output + self.conv_wordpos.output
        self.conv_out = self.conv_out.dimshuffle(1,0,2,3,4).reshape(inputs.shape[0],inputs.shape[2],conv_hidden_feature_num,-1)

        # max_out shape: (batch_size,max_term_per_sent,conv_hidden_feature_num)
        self.max_out = T.max(self.conv_out,axis=3).reshape((self.conv_out.shape[0],))


        # hidden layer
        # hidden_layer output shape: (batch_size, max_term_per_sent, hidden_layer_size)
        self.hidden_layer = HiddenLayer(rng=rng, input=self.max_out,
                                       n_in = self.conv_hidden_feature_num,
                                       n_out = hidden_layer_size,
                                       activation=T.tanh)

        # TODO make a scan here
        likelihood = theano.shared(0.0)
        # TODO we use poitwise likelihood here
        results, updates = theano.scan(lambda din,mask,pre_like :
                pre_like + SentenceLevelLogLikelihoodLayer(din,
                    hidden_layer_size,
                    SrlProblem.get_class_num()).negative_log_likelihood_pointwise(mask),
                sequences=[self.hidden_layer.output,self.masks],
                outputs_info=likelihood)
        self.likelihood = results[-1]


        # The logistic regression layer gets as input the hidden units
        # of the hidden layer


class SrlNeuralLanguageModel(object):

    def __init__(self):
        self.input = T.imatrix('input') # the data is a minibatch
        self.label = T.imatrix('label') # label's shape (mini_batch size, max_term_per_sent)
        self.masks = T.ivector('masks') # masks which used in error and likelihood calculation

        self.core = SrlNeuralLanguageModelCore(self.input)

        self.params = self.core.wordvec.params() \
                + self.core.POSvec.params() \
                + self.core.wordpos_vec.params() \
                + self.core.verbpos_vec.params() \
                + self.core.conv_word.params() \
                + self.core.conv_POS.params() \
                + self.core.conv_wordpos.params() \
                + self.core.conv_verbpos.params() \
                + self.core.hidden_layer.params

        self.L2_sqr = (self.core.wordvec.embeddings ** 2).sum() \
                    + (self.core.POSvec.embeddings ** 2).sum() \
                    + (self.core.wordpos_vec.embeddings ** 2).sum() \
                    + (self.core.verbpos_vec.embeddings ** 2).sum() \
                    + (self.core.conv_word.W ** 2).sum() \
                    + (self.core.conv_POS.W ** 2).sum() \
                    + (self.core.conv_wordpos.W ** 2).sum() \
                    + (self.core.conv_verbpos.W ** 2).sum() \
                    + (self.core.hidden_layer.W ** 2).sum()

        self.negative_log_likelihood = self.core.output_layer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.core.output_layer.errors

        self.cost = self.negative_log_likelihood(self.label,self.masks) \
                     + self.L2_reg * self.L2_sqr


    def fit_batch(self,x,y,masks,iter_num=1000,learning_rate=0.1):
        self.gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)

        updates = []

        for param, gparam in zip(self.params, self.gparams):
            updates.append((param, param - learning_rate * gparam))

        borrow = True
        train_set_X = T.cast(theano.shared(numpy.asarray(X,
                                dtype=theano.config.floatX),
                                 borrow=borrow), "int32")
        train_set_y = T.cast(theano.shared(numpy.asarray(y,
                                dtype=theano.config.floatX),
                                 borrow=borrow), "int32")
        train_set_mask = T.cast(theano.shared(numpy.asarray(masks,
                                dtype=theano.config.floatX),
                                 borrow=borrow), "int32")

        train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.cost,
                updates=updates,
                givens={
                    self.input: train_set_X,
                    self.label: train_set_y,
                    self.masks: train_set_mask
                })

        print '... training'

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        best_params = None
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        validation_frequency = 100

        while (epoch < n_epochs) and (not done_looping):

            print >> sys.stderr, "begin epoch ", epoch
            epoch = epoch + 1

            minibatch_avg_cost = train_model(x,y,masks)


    def valid(self,x,y):
        pass

