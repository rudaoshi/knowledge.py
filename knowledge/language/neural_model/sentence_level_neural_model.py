__author__ = 'Huang'

import time

import theano.tensor as T
import theano
from knowledge.language.neural_model.sentence_level_log_likelihood_layer import SentenceLevelLogLikelihoodLayer
from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.conv1d_layer import Conv1DLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression
from knowledge.machine.neuralnetwork.layer.base import BaseModel

from knowledge.language.problem.locdifftypes import LocDiffToWordTypes
from theano.tensor.signal import downsample

from sklearn.metrics import f1_score

import time
import cPickle

class SentenceLevelNeuralModelCore(object):


    def __init__(self,rng,**kwargs):
        # x shape: [mini-batch size, feature-dim].
        # In this problem [mini-batch feature-dim]



        self.x = T.lmatrix('x')
        self.z = T.lmatrix('z')


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

        self.hidden_output_dim_1 = kwargs['hidden_output_dim_1']
        self.hidden_output_dim_2 = kwargs['hidden_output_dim_2']


        batch_size = self.x.shape[0]
        #feature_num = x.shape[1]
        #assert (feature_num - 6) % 4 == 0, "Bad input parmeter X with wrong size {0}".format(x.shape)
        #sentence_length = (feature_num - 6)/4

        #all_word_id_input = x[:, 0 : sentence_length ]
        #all_pos_id_input = x[:, sentence_length : 2*sentence_length ]
        #all_dist_to_verb_id_input = x[:, 2*sentence_length :3*sentence_length ]
        #all_dist_to_word_id_input = x[:, 3*sentence_length :4*sentence_length ]
        word_id_input = self.x[:, 0]
        verb_id_input = self.x[:, 1]
        word_pos_input = self.x[:, 2]
        verb_pos_input = self.x[:, 3]
        verb_loc_input = self.x[:, 4]
        word_loc_input = self.x[:, 5]
        dist_id_verb2word = self.x[:, 6]

        tree_word_input = self.z[0,:].reshape((1,self.z.shape[1]))
        tree_POS_input = self.z[1,:].reshape((1,self.z.shape[1]))

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
        self.wordvec = wordvec
        # 2,verb vector
        #   output shape: (batch size,sentence_len, POS_feature_num)
        '''
        self.verb_embedding_layer = LookupTableLayer(
            table_size = self.word_num,
            feature_num = self.word_feature_dim
        )
        '''
        self.verb_embedding_layer = self.word_embedding_layer
        verbvec = self.verb_embedding_layer.output(
            inputs = verb_id_input,
            tensor_output = True
        )
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
        # 4,verb POS tag vector
        #   output shape: (batch size,sentence_len, POS_feature_num)
        '''
        self.verb_pos_embedding_layer = LookupTableLayer(
            table_size = self.POS_type_num,
            feature_num = self.POS_feature_dim,
        )
        '''
        self.verb_pos_embedding_layer = self.word_pos_embedding_layer
        verbPOSvec = self.verb_pos_embedding_layer.output(
            inputs = verb_pos_input,
            tensor_output = True
        )
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
        # 6,word features from tree
        # output shape: (batch size,1,sentence_len, word_feature_num)
        # here we have output shape: (1,sentence_len, word_feature_num)
        self.tree_word_embedding_layer = self.word_embedding_layer
        tree_word_raw = self.tree_word_embedding_layer.output(
            inputs = tree_word_input,
            tensor_output = True
        )
        # we use conv here instead of rnn, which makes it easy to train
        # extend (1,sentence_len,POS_feature_num) to (1,1,sentence_len,word_feature_num)
        tree_word_raw = tree_word_raw.dimshuffle(0,'x',1,2)
        self.tree_word_conv_layer = Conv1DLayer('conv',rng,1,self.conv_output_dim,self.word_feature_dim)
        # treecnov shape (batch size,conv_output_dim,sentence_length,1)
        # in here we have (1,conv_output_dim,sentence_length,1)
        treewordconv = self.tree_word_conv_layer.output(tree_word_raw).reshape((tree_word_input.shape[0],self.conv_output_dim,tree_word_input.shape[1]))
        tree_word_vec = T.max(treewordconv,axis=2).reshape((tree_word_input.shape[0],self.conv_output_dim))
        tree_word_vec = tree_word_vec.repeat(batch_size,axis=0)
        self.tree_word_vec = tree_word_vec
        # 7,word pos from tree
        # output shape: (batch size,1,sentence_len, word_feature_num)
        # here we have output shape: (1,sentence_len, word_feature_num)
        self.tree_POS_embedding_layer = self.word_pos_embedding_layer
        tree_POS_raw = self.tree_POS_embedding_layer.output(
            inputs = tree_POS_input,
            tensor_output = True
        )
        # we use conv here instead of rnn, which makes it easy to train
        # extend (1,sentence_len,POS_feature_num) to (1,1,sentence_len,word_feature_num)
        tree_POS_raw = tree_POS_raw.dimshuffle(0,'x',1,2)
        self.tree_POS_conv_layer = Conv1DLayer('conv',rng,1,self.conv_output_dim,self.POS_feature_dim)
        # treecnov shape (batch size,conv_output_dim,sentence_length,1)
        # in here we have (1,conv_output_dim,sentence_length,1)
        treeposconv = self.tree_POS_conv_layer.output(tree_POS_raw).reshape((tree_POS_input.shape[0],self.conv_output_dim,tree_POS_input.shape[1]))
        tree_POS_vec = T.max(treeposconv,axis=2).reshape((tree_POS_input.shape[0],self.conv_output_dim))
        tree_POS_vec = tree_POS_vec.repeat(batch_size,axis=0)
        self.tree_POS_vec = tree_POS_vec
        input_cat = T.concatenate(
            (
                wordvec,
                verbvec,
                wordPOSvec,
                verbPOSvec,
                distvec,
                tree_word_vec,
                #tree_POS_vec
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
        self.hidden_layer_1 = HiddenLayer(rng=rng, input=self.input_cat,
                n_in = input_cat_dim,
                n_out = self.hidden_output_dim_1,
                activation=T.tanh)

        '''
        self.hidden_layer_2 = HiddenLayer(rng=rng, input=self.hidden_layer_1.output,
                n_in = self.hidden_output_dim_1,
                n_out = self.hidden_output_dim_2,
                activation=T.tanh)
        '''


        self.output_layer = LogisticRegression(
                                        input=self.hidden_layer_1.output,
                                        n_in=self.hidden_output_dim_2,
                                        n_out=self.SRL_type_num)


        #self.errors = self.output_layer.errors
        # TODO we use poitwise likelihood here
        # self.sentce_loglikelihood = SentenceLevelLogLikelihoodLayer(rng,self.hidden_layer.output,
        #         self.y,
        #         self.max_term_per_sent,
        #         self.hidden_layer_size,
        #         self.tags_num)
        #
        #self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        # self._errors = self.sentce_loglikelihood.errors()

        self.params = self.word_embedding_layer.params() \
                + self.word_pos_embedding_layer.params() \
                + self.dist_embedding_layer.params() \
                + self.tree_word_conv_layer.params() \
                + self.hidden_layer_1.params() \
                + self.output_layer.params()

                #+ self.hidden_layer_2.params() \
                #+ self.tree_POS_conv_layer.params() \
                #+ self.verb_embedding_layer.params() \
                #+ self.verb_pos_embedding_layer.params() \


    def inputs(self):
        return [self.x,self.z]

import numpy as np


class SentenceLevelNeuralModel(BaseModel):

    def __init__(self,name,rng,load,dump,model_folder=None,init_model_name=None,**kwargs):
        super(SentenceLevelNeuralModel,self).__init__(name,model_folder)
        self.label = T.lvector('label') # label's shape (mini_batch size)
        self.lr = T.scalar('lr') # learning rate
        self.load = load
        self.dump = dump

        if self.load:
            self.core = self.load_core(init_model_name)
        else:
            self.core = SentenceLevelNeuralModelCore(rng,**kwargs)



    def fit(self, train_problem, valid_problem, **kwargs):


        self.L1_reg = kwargs['L1_reg']
        self.L2_reg = kwargs['L2_reg']


        self.L2_sqr = (self.core.word_embedding_layer.embeddings ** 2).sum() \
                + (self.core.word_pos_embedding_layer.embeddings ** 2).sum() \
                + (self.core.dist_embedding_layer.embeddings ** 2).sum() \
                + (self.core.tree_word_conv_layer.W ** 2).sum() \
                + (self.core.hidden_layer_1.W ** 2).sum() \
                + (self.core.output_layer.W ** 2).sum()

                #+ (self.core.hidden_layer_2.W ** 2).sum() \
                #+ (self.core.tree_POS_conv_layer.W ** 2).sum() \
                #+ (self.core.verb_embedding_layer.embeddings ** 2).sum() \
                #+ (self.core.verb_pos_embedding_layer.embeddings ** 2).sum() \

        self.cost = self.core.output_layer.negative_log_likelihood(self.label) \
                    + self.L2_reg * self.L2_sqr
        self.errors = self.core.output_layer.errors(self.label)

        self.params = self.core.params



        self.gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)

        self.updates = []

        learning_rate = kwargs['learning_rate']
        min_learning_rate = kwargs['min_learning_rate']
        learning_rate_decay_ratio = kwargs['learning_rate_decay_ratio']
        for param, gparam in zip(self.params, self.gparams):
            #self.updates.append((param, param - learning_rate * gparam))
            self.updates.append((param, param - self.lr * gparam))


        self.train_model = theano.function(
            inputs=self.core.inputs()+[self.label,self.lr],
            outputs=self.cost,
            updates=self.updates,
        )
        self.valid_model = theano.function(
            inputs=self.core.inputs()+[self.label],
            outputs=[self.errors,self.core.output_layer.y_pred],
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
                    #minibatch_avg_cost,tree_word_vec = self.train_model(X,y,z)
                    minibatch_avg_cost= self.train_model(X,z,y,learning_rate)

                    #print 'tree_word_vec shape',tree_word_vec.shape
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
                        if self.dump:
                            print 'dumping...'
                            self.dump_core('%d-%d' % (epoch,minibatch),False)

                        # compute zero-one loss on validation set
                        validation_losses = []
                        validation_pred = []
                        validation_label = []
                        test_num = 0
                        for valid_X, valid_y, valid_z in valid_problem.get_data_batch():
                            test_num += 1
                            error,pred = self.valid_model(valid_X,valid_z,valid_y)
                            validation_losses.append(error)
                            validation_pred += pred.tolist()
                            validation_label += valid_y.tolist()

                            #if test_num >= 100:
                            #    break

                        this_validation_loss = np.mean(validation_losses)
                        f1 = f1_score(np.asarray(validation_label),np.asarray(validation_pred),average='weighted')

                        if info:
                            str = 'minibatch {0}, validation error {1} %%, f1 score {2} , learning rate {3}'.format(total_minibatch, this_validation_loss * 100.,f1,learning_rate)
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

                learning_rate *= learning_rate_decay_ratio
                if learning_rate <= min_learning_rate:
                    learning_rate = min_learning_rate

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



