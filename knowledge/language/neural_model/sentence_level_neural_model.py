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


    def __init__(self,rng,x,y,model_params):
        # x shape: [mini-batch size, feature-dim].
        # In this problem [mini-batch feature-dim]



        self.x = x
        self.y = y



        self.max_sentence_length = model_params['max_sentence_length']
        self.window_size = model_params['window_size']
        self.max_term_per_sent = self.max_sentence_length - self.window_size + 1
        self.word_num = model_params['word_num']
        self.POS_num = model_params['POS_num']

        self.word_feature_num = model_params['word_feature_num']
        self.POS_feature_num = model_params['POS_feature_num']

        self.conv_window = model_params['conv_window']
        self.conv_hidden_feature_num = model_params['conv_hidden_feature_num']

        self.hidden_layer_size = model_params['hidden_layer_size']
        self.srl_type_num = model_params['srl_type_num']

        batch_size = x.shape[0]
        sentence_length = int(x[0][0])
        # [sentence.word_num()] + word_id_vec + pos_id_vec +
        # [word_idx, PosTags.POSTAG_ID_MAP[word.pos], loc_diff[word_idx]]

        all_word_id_input = x[:, 1: sentence_length + 1]
        all_pos_id_input = x[:, sentence_length + 1: 2*sentence_length + 1]
        word_id_input = x[:, -3]
        pos_id_input = x[:, -2]
        loc_diff_input = x[:, -1]

        # we have 4 lookup tables here:
        # 1,word vector
        #   output shape: (batch size,sentence_len * word_feature_num)
        # 2,POS tag vector
        #   output shape: (batch size,sentence_len * POS_feature_num)
        self.word_embedding_layer = LookupTableLayer(table_size = self.word_num,
                feature_num = self.word_feature_num)
        self.wordvec = self.word_embedding_layer.output(inputs = all_word_id_input) + x[:,-3:]

        self.pos_embedding_layer = LookupTableLayer( table_size = self.POS_num,
                feature_num = self.POS_feature_num,)
        self.POSvec = self.pos_embedding_layer.output(inputs = all_pos_id_input) + x[:,-3:]

        conv_feature_num = 100
        # conv_word.out.shape = (batch_size,conv_feature_num, 1, feature_num - conv_window+1)
        # conv_POS.out.shape = (batch_size,conv_feature_num, 1, feature_num - conv_window+1)
        # conv_verbpos.out.shape = (batch_size,1,conv_hidden_feature_num,max_sentence_length-conv_window+1)
        # conv_wordpos.out.shape = (batch_size,max_sentence_length,conv_hidden_feature_num,max_sentence_length-conv_window+1)
        # note. all output above have been seted 'dimshuffle'
        self.conv_word = Conv1DLayer('conv_word',rng, self.wordvec.output.dimshuffle(0,'x','x',1),
                1, conv_feature_num, self.conv_window)

        self.conv_POS = Conv1DLayer('conv_POS',rng, self.POSvec.output.dimshuffle(0,'x','x',1),
                1, conv_feature_num, self.conv_window)


        # the first max_sentence_length means each element of it is one prediction for that word
        # the second max_sentence_length means each element of it is one output of conv
        # conv_out shape: (batch_size,max_term_per_sent,conv_hidden_feature_num,max_term_per_sent)
        self.conv_out = self.conv_word.output + self.conv_POS.output
        self.conv_out = self.conv_out.reshape((batch_size, conv_feature_num * self.conv_out.shape[-1]))


        # max_out shape: (batch_size,max_term_per_sent,conv_hidden_feature_num)
        self.conv_max_feature = T.max(self.conv_out, axis=0)


        # hidden layer
        # hidden layer perform one linear map and one nolinear transform
        # ,then in likelihood, it performs another linear map. This is 
        # what senna do (P.7, figure 1).
        # hidden_layer OUTPUT SHAPE: (batch_size, max_term_per_sent, hidden_layer_size)
        self.hidden_layer = HiddenLayer(rng=rng, input=self.conv_max_feature,
                n_in = self.conv_hidden_feature_num,
                n_out = self.hidden_layer_size,
                activation=T.tanh)


        self.output_layer = LogisticRegression(
                                        input=self.hidden_layer.output,
                                        n_in=self.hidden_layer_size,
                                        n_out=self.srl_type_num)

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



    #
    # def errors(self):
    #     return self._errors



class SentenceLevelNeuralModel(object):

    def __init__(self,rng, model_params):
        self.input = T.itensor4('input') # the data is a minibatch (a sentence)
        self.label = T.ivector('label') # label's shape (mini_batch size, max_term_per_sent)

        self.L1_reg = model_params['L1_reg']
        self.L2_reg = model_params['L2_reg']

        self.core = SentenceLevelNeuralModelCore(rng, self.input, self.label, model_params)

        self.params = self.core.wordvec.params() \
                + self.core.POSvec.params() \
                + self.core.conv_word.params() \
                + self.core.conv_POS.params() \
                + self.core.hidden_layer.params

        self.L2_sqr = (self.core.wordvec.embeddings ** 2).sum() \
                + (self.core.POSvec.embeddings ** 2).sum() \
                + (self.core.conv_word.W ** 2).sum() \
                + (self.core.conv_POS.W ** 2).sum() \
                + (self.core.hidden_layer.W ** 2).sum()


        self.cost = self.core.negative_log_likelihood(self.label) \
                    + self.L2_reg * self.L2_sqr
        self.errors = self.core.errors(self.label)



        self.gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)

        self.updates = []

        learning_rate = model_params['learning_rate']
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



