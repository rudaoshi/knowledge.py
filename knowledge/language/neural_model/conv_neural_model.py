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


    def __init__(self,rng,x,y,sent_length,masks,
            model_params):
        # x shpae: (batch_size,max_term_per_sent+3,max_sentence_length)
        # ,where  max_sentence_length = max_term_per_sent + window_size - 1
        self.L1_reg = model_params['L1_reg']
        self.L2_reg = model_params['L2_reg']

        self.x = x
        self.y = y
        self.sent_length = sent_length
        self.masks = masks

        self.max_sentence_length = model_params['max_sentence_length']
        self.window_size = model_params['window_size']
        self.max_term_per_sent = self.max_sentence_length - self.window_size + 1
        self.word_num = model_params['word_num']
        self.POS_num = model_params['POS_num']
        self.verbpos_num = model_params['verbpos_num']
        self.wordpos_num = model_params['wordpos_num']

        self.word_feature_num = model_params['word_feature_num']
        self.POS_feature_num = model_params['POS_feature_num']
        self.wordpos_feature_num = model_params['wordpos_feature_num']
        self.verbpos_feature_num = model_params['verbpos_feature_num']

        self.conv_window = model_params['conv_window']
        self.conv_hidden_feature_num = model_params['conv_hidden_feature_num']

        self.hidden_layer_size = model_params['hidden_layer_size']
        self.tags_num = model_params['tags_num']

        # we have 4 lookup tables here:
        # 1,word vector
        #   output shape: (batch size,1,max_sentence_length * word_feature_num)
        # 2,POS tag vector
        #   output shape: (batch size,1,max_sentence_length * POS_feature_num)
        # 3,verb position vector
        #   output shape: (batch size,1,max_sentence_length * verbpos_feature_num)
        # 4,word position vector
        #   output shape: (batch size,max_term_per_sent,1,max_sentence_length * wordpos_feature_num)
        self.wordvec = LookupTableLayer(inputs = x[:,0:1,:], table_size = self.word_num,
                window_size = self.max_sentence_length, feature_num = self.word_feature_num,
                reshp = (x.shape[0],1,1,x.shape[2] * self.word_feature_num))

        self.POSvec = LookupTableLayer(inputs = x[:,1:2,:], table_size = self.POS_num,
                window_size = self.max_sentence_length, feature_num = self.POS_feature_num,
                reshp = (x.shape[0],1,1,x.shape[2] * self.POS_feature_num))

        self.verbpos_vec = LookupTableLayer(inputs = x[:,2:3,:], table_size = self.verbpos_num,
                window_size = self.max_sentence_length, feature_num = self.verbpos_feature_num,
                reshp = (x.shape[0],1,1,x.shape[2] * self.verbpos_feature_num))

        self.wordpos_vec = LookupTableLayer(inputs = x[:,3:,:], table_size = self.wordpos_num,
                window_size = self.max_sentence_length, feature_num = self.wordpos_feature_num,
                reshp = (x.shape[0],self.max_term_per_sent,1,x.shape[2] * self.wordpos_feature_num))

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
                self.conv_hidden_feature_num,self.max_term_per_sent,self.conv_window,self.wordpos_feature_num)


        # the first max_sentence_length means each element of it is one prediction for that word
        # the second max_sentence_length means each element of it is one output of conv
        # conv_out shape: (batch_size,max_term_per_sent,conv_hidden_feature_num,max_term_per_sent)
        self.conv_out = self.conv_word.output + self.conv_POS.output + self.conv_verbpos.output + self.conv_wordpos.output
        self.conv_out = self.conv_out.dimshuffle(1,0,2,3,4).reshape((x.shape[0],self.max_term_per_sent,self.conv_hidden_feature_num,-1))

        # max_out shape: (batch_size,max_term_per_sent,conv_hidden_feature_num)
        self.max_out = T.max(self.conv_out,axis=3).reshape((self.conv_out.shape[0],self.max_term_per_sent,-1))


        # hidden layer
        # hidden layer perform one linear map and one nolinear transform
        # ,then in likelihood, it performs another linear map. This is 
        # what senna do (P.7, figure 1).
        # hidden_layer OUTPUT SHAPE: (batch_size, max_term_per_sent, hidden_layer_size)
        self.hidden_layer = HiddenLayer(rng=rng, input=self.max_out,
                n_in = self.conv_hidden_feature_num,
                n_out = self.hidden_layer_size,
                activation=T.tanh)

        self.sentce_loglikelihood = SentenceLevelLogLikelihoodLayer(self.hidden_layer_size,self.tags_num)

        # TODO we use poitwise likelihood here
        '''
        results, _updates = theano.scan(lambda din,mask:
                self.sentce_loglikelihood.negative_log_likelihood_pointwise(din,self.y,mask),
                sequences=[self.hidden_layer.output,self.masks])
                #outputs_info=like)
        #self._likelihood = results[-1]
        self._likelihood = results.sum()
        '''
        #self._likelihood = self.sentce_loglikelihood.negative_log_likelihood_pointwise(self.hidden_layer.output[0,:,:],self.y[0,:],self.masks[0,:])

        results, _updates = theano.scan(lambda x_i:
                self.sentce_loglikelihood.negative_log_likelihood_pointwise(self.hidden_layer.output[0,:,:],self.y[0,:],self.masks[0,:]),
                sequences=[T.arange(self.x.shape[0])])
                #outputs_info=like)
        #self._likelihood = results[-1]
        self._likelihood = results.sum()

    def likelihood(self):
        return self._likelihood

    def errors(self):
        #errors = theano.shared(0.0)
        results, updates = theano.scan(lambda din,mask:
                self.sentce_loglikelihood.errors(self.y,mask),
                sequences=[self.hidden_layer.output,self.masks])
                #outputs_info=errors)
        #self._errors = results[-1]
        self._errors = results.sum()
        return self._errors



class SrlNeuralLanguageModel(object):

    def __init__(self,rng,model_params):
        self.input = T.itensor3('input') # the data is a minibatch
        self.label = T.imatrix('label') # label's shape (mini_batch size, max_term_per_sent)
        self.sent_length= T.ivector('sent_length') # sent_length is the number of terms in each sentence
        self.masks = T.imatrix('masks') # masks which used in error and likelihood calculation

        self.core = SrlNeuralLanguageModelCore(rng,self.input,self.label,self.sent_length,self.masks,model_params)

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

        self.negative_log_likelihood = self.core.likelihood()
        # same holds for the function computing the number of errors
        #self.errors = self.core.errors()

        # we only use L2 regularization
        self.cost = self.negative_log_likelihood \
                + self.core.L2_reg * self.L2_sqr


        self.gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)

        self.updates = []

        learning_rate = 0.1
        for param, gparam in zip(self.params, self.gparams):
            self.updates.append((param, param - learning_rate * gparam))


        #self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.core.conv_word.output,on_unused_input='ignore')
        #self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.core.conv_POS.output,on_unused_input='ignore')
        #self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.core.conv_verbpos.output,on_unused_input='ignore')
        #self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.core.conv_wordpos.output,on_unused_input='ignore')
        #self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.core.conv_out,on_unused_input='ignore')
        #self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.core.max_out,on_unused_input='ignore')
        #self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.core.hidden_layer.output,on_unused_input='ignore')
        #self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.core.negative_log_likelihood,on_unused_input='ignore')
        self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.cost,on_unused_input='ignore')
        #self.train_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.cost,updates=self.updates)

    def test_foo(self,x,y,sent_length,masks,batch_iter_num=1,learning_rate=0.1):
        borrow = True
        '''
        train_set_X = T.cast(theano.shared(numpy.asarray(x,
            dtype=theano.config.floatX),
            borrow=borrow), "int32")
        train_set_y = T.cast(theano.shared(numpy.asarray(y,
            dtype=theano.config.floatX),
            borrow=borrow), "int32")
        train_set_masks = T.cast(theano.shared(numpy.asarray(masks,
            dtype=theano.config.floatX),
            borrow=borrow), "int32")
        '''

        print '... training'

        start_time = time.clock()
        epoch = 0
        minibatch_avg_cost = 0
        # begin to train this mini batch
        while (epoch < batch_iter_num):
            epoch = epoch + 1
            minibatch_avg_cost = self.train_model(x,y,masks)
        end_time = time.clock()
        return minibatch_avg_cost,end_time - start_time
        '''
        return train_model
        '''


    def fit_batch(self,x,y,sent_length,masks,batch_iter_num=1,learning_rate=0.1):
        '''
        borrow = True
        train_set_X = T.cast(theano.shared(numpy.asarray(x,
            dtype=theano.config.floatX),
            borrow=borrow), "int32")
        train_set_y = T.cast(theano.shared(numpy.asarray(y,
            dtype=theano.config.floatX),
            borrow=borrow), "int32")
        train_set_masks = T.cast(theano.shared(numpy.asarray(masks,
            dtype=theano.config.floatX),
            borrow=borrow), "int32")
        '''


        print '... training'

        start_time = time.clock()
        epoch = 0
        minibatch_avg_cost = 0
        # begin to train this mini batch
        while (epoch < batch_iter_num):
            epoch = epoch + 1
            minibatch_avg_cost = self.train_model(x,y,masks)
        end_time = time.clock()
        return minibatch_avg_cost,end_time - start_time


    def valid(self,x,y,sent_length,masks):
        borrow = True
        valid_set_X = T.cast(theano.shared(numpy.asarray(x,
            dtype=theano.config.floatX),
            borrow=borrow), "int32")
        valid_set_y = T.cast(theano.shared(numpy.asarray(y,
            dtype=theano.config.floatX),
            borrow=borrow), "int32")
        valid_set_masks = T.cast(theano.shared(numpy.asarray(masks,
            dtype=theano.config.floatX),
            borrow=borrow), "int32")

        start_time = time.clock()
        valid_model = theano.function(inputs=[self.input,self.label,self.masks], outputs=self.errors,
                givens={
                    self.input: valid_set_X,
                    self.label: valid_set_y,
                    self.masks: valid_set_masks
                    })
        end_time = time.clock()
        return valid_model(x,y,masks),end_time - start_time



