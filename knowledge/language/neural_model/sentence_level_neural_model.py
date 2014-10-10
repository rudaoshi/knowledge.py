__author__ = 'huang'

import time

import theano.tensor as T
import theano
from knowledge.language.neural_model.sentence_level_log_likelihood_layer import SentenceLevelLogLikelihoodLayer
from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.conv1d_layer import Conv1DLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression

class SentenceLevelNeuralModelCore(object):


    def __init__(self,rng,x,y,**kwargs):
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
        feature_num = x.shape[1]
        assert (feature_num - 6) % 4 == 0, "Bad input parmeter X with wrong size"
        sentence_length = (x - 6)/4
        # [sentence.word_num()] + word_id_vec + pos_id_vec +
        # [word_idx, PosTags.POSTAG_ID_MAP[word.pos], loc_diff[word_idx]]

        all_word_id_input = x[:, 0 : sentence_length ]
        all_pos_id_input = x[:, sentence_length : 2*sentence_length ]
        all_dist_to_verb_id_input = x[:, 2*sentence_length :3*sentence_length ]
        all_dist_to_word_id_input = x[:, 3*sentence_length :4*sentence_length ]
        verb_id_input = x[:, -6]
        word_id_input = x[:, -5]
        verb_pos_input = x[:, -4]
        word_pos_input = x[:, -3]
        dist_id_verb2word = x[:, -2]
        dist_id_word2verb = x[:, -1]


        # we have 4 lookup tables here:
        # 1,word vector
        #   output shape: (batch size,sentence_len, word_feature_num)
        # 2,POS tag vector
        #   output shape: (batch size,sentence_len, POS_feature_num)
        self.word_embedding_layer = LookupTableLayer(
            table_size = self.word_num,
            feature_num = self.word_feature_dim
        )
        wordvec = self.word_embedding_layer.output(
            inputs = all_word_id_input,
            tensor_output = True
        )

        self.pos_embedding_layer = LookupTableLayer(
            table_size = self.POS_type_num,
            feature_num = self.POS_feature_dim,
        )
        POSvec = self.pos_embedding_layer.output(
            inputs = all_pos_id_input,
            tensor_output = True
        )

        self.dist_to_verb_embedding_layer = LookupTableLayer(
            table_size = self.dist_to_verb_num,
            feature_num = self.dist_to_verb_feature_dim,
        )
        dist_to_verb_vec = self.dist_to_verb_embedding_layer.output(
            inputs = all_dist_to_verb_id_input,
            tensor_output = True
        )

        self.dist_to_word_embedding_layer = LookupTableLayer(
            table_size = self.dist_to_word_num,
            feature_num = self.dist_to_word_feature_dim,
        )
        dist_to_word_vec = self.dist_to_word_embedding_layer.output(
            inputs = all_dist_to_word_id_input,
            tensor_output = True
        )

        self.word_conv_layer = Conv1DLayer(
            'conv_word',
            rng,
            1,
            self.conv_output_dim,
            self.conv_window_size)
        #output dim = (batchsize, conv_out_dim, sentence_len, word_embedding_dim)
        word_conv_out = self.word_conv_layer.output(wordvec.dimshuffle(0,'x',1,2))
        #reorganize data (batchsize, sentence_len, conv_out_dim * word_embedding_dim)
        word_conv_out = word_conv_out.dimshuffle(0,2,1,3).reshape(
            (
                word_conv_out.shape[0],
                word_conv_out.shape[1],
                -1
            )
        )

        self.pos_conv_layer = Conv1DLayer(
            'conv_POS',
            rng,
            1,
            self.conv_output_dim,
            self.conv_window_size)

        #output dim = (batchsize, conv_out_dim, sentence_len, pos_embedding_dim)
        pos_conv_out = self.pos_conv_layer.output(POSvec.dimshuffle(0,'x',1,2))
        #reorganize data (batchsize, sentence_len, conv_out_dim * word_embedding_dim)
        pos_conv_out = pos_conv_out.dimshuffle(0,2,1,3).reshape(
            (
                pos_conv_out.shape[0],
                pos_conv_out.shape[1],
                -1
            )
        )

        self.dist_to_verb_conv_layer = Conv1DLayer(
            'conv_dist_to_verb',
            rng,
            1,
            self.conv_output_dim,
            self.conv_window_size)

        #output dim = (batchsize, conv_out_dim, sentence_len, pos_embedding_dim)
        dist_to_verb_conv_out = self.dist_to_verb_conv_layer.output(dist_to_verb_vec.dimshuffle(0,'x',1,2))
        #reorganize data (batchsize, sentence_len, conv_out_dim * word_embedding_dim)
        dist_to_verb_conv_out = dist_to_verb_conv_out.dimshuffle(0,2,1,3).reshape(
            (
                dist_to_verb_conv_out.shape[0],
                dist_to_verb_conv_out.shape[1],
                -1
            )
        )

        self.dist_to_word_conv_layer = Conv1DLayer(
            'conv_dist_to_verb',
            rng,
            1,
            self.conv_output_dim,
            self.conv_window_size)

        #output dim = (batchsize, conv_out_dim, sentence_len, pos_embedding_dim)
        dist_to_word_conv_out = self.dist_to_word_conv_layer.output(dist_to_word_vec.dimshuffle(0,'x',1,2))
        #reorganize data (batchsize, sentence_len, conv_out_dim * word_embedding_dim)
        dist_to_word_conv_out = dist_to_word_conv_out.dimshuffle(0,2,1,3).reshape(
            (
                dist_to_word_conv_out.shape[0],
                dist_to_word_conv_out.shape[1],
                -1
            )
        )

        total_conv_output = T.concatenate(
            (
                word_conv_out,
                pos_conv_out,
                dist_to_verb_conv_out,
                dist_to_word_conv_out
            ),
            axis = 2
        )

        # max_out shape: (batch_size,max_term_per_sent,conv_hidden_feature_num)
        conv_max_feature = T.max(total_conv_output, axis=1)


        # hidden layer
        # hidden layer perform one linear map and one nolinear transform
        # ,then in likelihood, it performs another linear map. This is 
        # what senna do (P.7, figure 1).
        # hidden_layer OUTPUT SHAPE: (batch_size, max_term_per_sent, hidden_layer_size)
        self.hidden_layer = HiddenLayer(rng=rng, input=conv_max_feature,
                n_in = self.conv_output_dim,
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
                + self.pos_embedding_layer.params() \
                + self.word_conv_layer.params() \
                + self.pos_conv_layer.params() \
                + self.hidden_layer.params() \
                + self.output_layer.params()



    #
    # def errors(self):
    #     return self._errors

import numpy as np


class SentenceLevelNeuralModel(object):

    def __init__(self,rng, **kwargs):
        self.input = T.itensor4('input') # the data is a minibatch (a sentence)
        self.label = T.ivector('label') # label's shape (mini_batch size, max_term_per_sent)


        self.core = SentenceLevelNeuralModelCore(rng, self.input, self.label, **kwargs)



    def fit(self, train_problem, valid_problem, **kwargs):


        self.L1_reg = kwargs['L1_reg']
        self.L2_reg = kwargs['L2_reg']


        self.L2_sqr = (self.core.word_embedding_layer.embeddings ** 2).sum() \
                + (self.core.pos_embedding_layer.embeddings ** 2).sum() \
                + (self.core.word_conv_layer.W ** 2).sum() \
                + (self.core.pos_conv_layer.W ** 2).sum() \
                + (self.core.hidden_layer.W ** 2).sum()


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
            inputs=[self.input,self.label],
            outputs=self.cost,
            updates=self.updates,
            on_unused_input='ignore')
        self.valid_model = theano.function(
            inputs=[self.input,self.label],
            outputs=self.errors,
            on_unused_input='ignore')


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

        validation_frequency = 100

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1

            minibatch = 0
            for X, y in train_problem.get_data_batch():

                minibatch_avg_cost = self.train_model(X,y)
                minibatch += 1
                if info:
                    print 'epoch {0}.{1}, cost = {2}'.format(epoch,minibatch,minibatch_avg_cost)

            if epoch  % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = []
                for valid_X, valid_y in valid_problem.get_data_batch():

                    validation_losses.append(self.valid_model(valid_X,valid_y))

                this_validation_loss = np.mean(validation_losses)

                if info:
                    print 'epoch {0}, validation error {1} %%'.format(epoch, this_validation_loss * 100.)

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



