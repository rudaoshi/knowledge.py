__author__ = 'huang'

import time

import theano.tensor as T
import theano
from knowledge.language.neural_model.sentence_level_log_likelihood_layer import SentenceLevelLogLikelihoodLayer
from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.conv1d_layer import Conv1DLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression

from knowledge.language.problem.locdifftypes import LocDiffToWordTypes
import time

class SentenceLevelNeuralModelCore(object):


    def __init__(self,rng,x,y,z,**kwargs):
        # x shape: [mini-batch size, feature-dim].
        # In this problem [mini-batch feature-dim]



        self.x = x
        self.y = y


        self.word_num = kwargs['word_num']
        self.POS_type_num = kwargs['POS_type_num']
        self.SRL_type_num = kwargs['SRL_type_num']
        self.dist_to_verb_num = kwargs['dist_to_verb_num']
        self.dist_to_word_num = kwargs['dist_to_word_num']

        self.word_feature_dim = kwargs['word_feature_dim']
        self.POS_feature_dim = kwargs['POS_feature_dim']
        self.dist_to_verb_feature_dim = kwargs['dist_to_verb_feature_dim']
        self.dist_to_word_feature_dim = kwargs['dist_to_word_feature_dim']

        self.conv_window_size = kwargs['conv_window_size']
        self.conv_output_dim = kwargs['conv_output_dim']

        self.hidden_output_dim = kwargs['hidden_output_dim']


        batch_size = x.shape[0]
        #feature_num = x.shape[1]
#        assert (feature_num - 6) % 4 == 0, "Bad input parmeter X with wrong size {0}".format(x.shape)
        #sentence_length = (feature_num - 6)/4
        # [sentence.word_num()] + word_id_vec + pos_id_vec +
        # [word_idx, PosTags.POSTAG_ID_MAP[word.pos], loc_diff[word_idx]]

        #all_word_id_input = x[:, 0 : sentence_length ]
        #all_pos_id_input = x[:, sentence_length : 2*sentence_length ]
        #all_dist_to_verb_id_input = x[:, 2*sentence_length :3*sentence_length ]
        #all_dist_to_word_id_input = x[:, 3*sentence_length :4*sentence_length ]
        word_id_input = x[:, 0]
        verb_id_input = x[:, 1]
        word_pos_input = x[:, 2]
        verb_pos_input = x[:, 3]
        verb_loc_input = x[:, 4]
        word_loc_input = x[:, 5]
        dist_id_verb2word = x[:, 6]

        tree_input = z

        # we have 5 lookup tables here:
        # 1,word vector
        #   output shape: (batch size,sentence_len, word_feature_num)
        self.word_embedding_layer = LookupTableLayer(
            table_size = self.word_num,
            feature_num = self.word_feature_dim
        )
        wordvec = self.word_embedding_layer.output(
            inputs = word_id_input,
            tensor_output = True
        )
        #).reshape(batch_size,-1)
        self.wordvec = wordvec
        # 2,verb vector
        #   output shape: (batch size,sentence_len, POS_feature_num)
        self.verb_embedding_layer = LookupTableLayer(
            table_size = self.word_num,
            feature_num = self.word_feature_dim
        )
        verbvec = self.verb_embedding_layer.output(
            inputs = verb_id_input,
            tensor_output = True
        )
        #).reshape(batch_size,-1)
        # 3,word POS tag vector
        #   output shape: (batch size,sentence_len, POS_feature_num)
        self.word_pos_embedding_layer = LookupTableLayer(
            table_size = self.POS_type_num,
            feature_num = self.POS_feature_dim,
        )
        wordPOSvec = self.word_pos_embedding_layer.output(
            inputs = word_pos_input,
            tensor_output = True
        )
        #).reshape(batch_size,-1)
        # 4,verb POS tag vector
        #   output shape: (batch size,sentence_len, POS_feature_num)
        self.verb_pos_embedding_layer = LookupTableLayer(
            table_size = self.POS_type_num,
            feature_num = self.POS_feature_dim,
        )
        verbPOSvec = self.verb_pos_embedding_layer.output(
            inputs = verb_pos_input,
            tensor_output = True
        )
        #).reshape(batch_size,-1)
        # 5,distance tag vector
        #   output shape: (batch size,sentence_len, POS_feature_num)
        self.dist_embedding_layer = LookupTableLayer(
            table_size = len(LocDiffToWordTypes.DIFF_ID_MAP),
            feature_num = self.dist_to_verb_feature_dim,
        )
        distvec = self.dist_embedding_layer.output(
            inputs = dist_id_verb2word,
            tensor_output = True
        )
        #).reshape(batch_size,-1)
        # 6,features from tree
        # output shape: (batch size,1,sentence_len, word_feature_num)
        # here we have output shape: (1,sentence_len, word_feature_num)
        self.tree_embedding_layer = self.word_embedding_layer
        tree_raw = self.tree_embedding_layer.output(
            inputs = tree_input,
            tensor_output = True
        )
        # we use conv here instead of rnn, which makes it easy to train
        # extend (1,sentence_len,POS_feature_num) to (1,1,sentence_len,word_feature_num)
        tree_raw = tree_raw.dimshuffle(0,'x',1,2)
        self.tree_conv_layer = Conv1DLayer('conv',rng,1,self.conv_output_dim,self.word_feature_dim)
        # treecnov shape (batch size,conv_output_dim,sentence_length,1)
        # in here we have (1,conv_output_dim,sentence_length,1)
        treeconv = self.tree_conv_layer.output(tree_raw).reshape((tree_input.shape[0],self.conv_output_dim,-1))
        treevec = T.max(treeconv,axis=2).reshape((tree_input.shape[0],-1))
        treevec = treevec.repeat(batch_size,axis=0)
        self.treevec = treevec

        input_cat = T.concatenate(
            (
                wordvec,
                verbvec,
                wordPOSvec,
                verbPOSvec,
                distvec,
                treevec
            ),
            axis = 1
        )
        self.input_cat = input_cat

        input_cat_dim = self.word_feature_dim * 2 + \
                self.POS_feature_dim * 2 + \
                self.dist_to_verb_feature_dim + \
                self.conv_output_dim

        # hidden layer
        # hidden layer perform one linear map and one nolinear transform
        # ,then in likelihood, it performs another linear map. This is 
        # what senna do (P.7, figure 1).
        # hidden_layer OUTPUT SHAPE: (batch_size, max_term_per_sent, hidden_layer_size)
        self.hidden_layer = HiddenLayer(rng=rng, input=self.input_cat,
                n_in = input_cat_dim,
                n_out = self.hidden_output_dim,
                activation=T.tanh)


        self.output_layer = LogisticRegression(
                                        input=self.hidden_layer.output,
                                        n_in=self.hidden_output_dim,
                                        n_out=self.SRL_type_num)


        self.errors = self.output_layer.errors
        # TODO we use poitwise likelihood here
        # self.sentce_loglikelihood = SentenceLevelLogLikelihoodLayer(rng,self.hidden_layer.output,
        #         self.y,
        #         self.max_term_per_sent,
        #         self.hidden_layer_size,
        #         self.tags_num)
        #
        self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        # self._errors = self.sentce_loglikelihood.errors()

        self.params = self.word_embedding_layer.params() \
                + self.verb_embedding_layer.params() \
                + self.word_pos_embedding_layer.params() \
                + self.verb_pos_embedding_layer.params() \
                + self.dist_embedding_layer.params() \
                + self.tree_conv_layer.params() \
                + self.hidden_layer.params() \
                + self.output_layer.params()



    #
    # def errors(self):
    #     return self._errors

import numpy as np


class SentenceLevelNeuralModel(object):

    def __init__(self,rng, **kwargs):
        self.input = T.lmatrix('input') # the data is a minibatch (a sentence)
        self.label = T.lvector('label') # label's shape (mini_batch size)
        self.tree = T.lmatrix('tree')


        self.core = SentenceLevelNeuralModelCore(rng, self.input, self.label,self.tree, **kwargs)



    def fit(self, train_problem, valid_problem, **kwargs):


        self.L1_reg = kwargs['L1_reg']
        self.L2_reg = kwargs['L2_reg']


        self.L2_sqr = (self.core.word_embedding_layer.embeddings ** 2).sum() \
                + (self.core.verb_embedding_layer.embeddings ** 2).sum() \
                + (self.core.word_pos_embedding_layer.embeddings ** 2).sum() \
                + (self.core.verb_pos_embedding_layer.embeddings ** 2).sum() \
                + (self.core.dist_embedding_layer.embeddings ** 2).sum() \
                + (self.core.tree_conv_layer.W ** 2).sum() \
                + (self.core.hidden_layer.W ** 2).sum() \
                + (self.core.output_layer.W ** 2).sum()


        self.cost = self.core.negative_log_likelihood(self.label) \
                    + self.L2_reg * self.L2_sqr
        self.errors = self.core.errors(self.label)

        self.params = self.core.params



        self.gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)

        self.updates = []

        learning_rate = kwargs['learning_rate']
        for param, gparam in zip(self.params, self.gparams):
            self.updates.append((param, param - learning_rate * gparam))


        self.train_model = theano.function(
            inputs=[self.input,self.label,self.tree],
            outputs=[self.cost,self.core.treevec],
            updates=self.updates,
            on_unused_input='ignore')
        self.valid_model = theano.function(
            inputs=[self.input,self.label,self.tree],
            outputs=self.errors,
            on_unused_input='ignore')

        '''
        self.tmp = theano.function(inputs=[self.input,self.label],
                outputs=[self.core.input_cat],
                on_unused_input='ignore')
        '''

        n_epochs = kwargs["n_epochs"]
        info = kwargs["info"]


        ###############
        # TRAIN MODEL #
        ###############
        if info:
            print '... training'

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant


        best_params = None
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        validation_frequency = 1000

        total_minibatch = 0

        with open('res.log','w') as fw:
            while (epoch < n_epochs) and (not done_looping):
                epoch = epoch + 1

                minibatch = 0
                for X, y, z in train_problem.get_data_batch():

                    #print 'X shape',X.shape
                    #print 'y shape',y.shape
                    #print 'z shape',z.shape
                    start_time = time.clock()
                    minibatch_avg_cost,treevec = self.train_model(X,y,z)
                    #print 'treevec shape',treevec.shape
                    end_time = time.clock()
                    #vec = np.asarray(self.tmp(X,y))
                    #print X.shape,vec.shape
                    minibatch += 1
                    total_minibatch += 1
                    if info and minibatch % 100 == 0:
                        str = 'epoch {0}.{1}, cost = {2}, time = {3}'.format(epoch,minibatch,minibatch_avg_cost,end_time - start_time)
                        print str
                        fw.write(str + '\n')


                    if total_minibatch  % validation_frequency == 0:
                        # compute zero-one loss on validation set
                        validation_losses = []
                        test_num = 0
                        for valid_X, valid_y, valid_z in valid_problem.get_data_batch():
                            test_num += 1
                            validation_losses.append(self.valid_model(valid_X,valid_y,valid_z))

                            #if test_num >= 100:
                            #    break

                        this_validation_loss = np.mean(validation_losses)

                        if info:
                            str = 'minibatch {0}, validation error {1} %%'.format(total_minibatch, this_validation_loss * 100.)
                            print str
                            fw.write(str + '\n')

                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:
                            #improve patience if loss improvement is good enough
                            if this_validation_loss < best_validation_loss *  \
                                   improvement_threshold:
                                patience = max(patience, epoch * patience_increase)

                            best_validation_loss = this_validation_loss
                            best_iter = epoch

                        if patience <= epoch:
                            done_looping = True
                            break


        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i.') %
              (best_validation_loss * 100., epoch))

    def fit_batch(self,x,y,sent_length,masks,batch_iter_num=1,learning_rate=0.1):
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
        start_time = time.clock()
        minibatch_errors,pred = self.valid_model(x,y,masks)
        end_time = time.clock()
        return minibatch_errors,pred,end_time - start_time



