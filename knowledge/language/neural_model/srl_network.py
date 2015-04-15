#coding=utf-8

import os
import time
import itertools
import theano.tensor as T
import theano
import numpy

from knowledge.machine.neuralnetwork.layer.perception import PerceptionLayer
from knowledge.machine.neuralnetwork.layer.conv1d_maxpool_layer import Conv1DMaxPoolLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.machine.neuralnetwork.layer.path_transition_layer import PathTransitionLayer
from knowledge.machine.cost.cost_factory import create_cost
from knowledge.machine.optimization.gradient_optimizable import GradientOptimizable

numpy.set_printoptions(threshold='nan')
class SRLNetowrkArchitecture(object):

    def __init__(self):

        self.word_feature_dim = None
        self.pos_feature_dim = None
        self.dist_feature_dim = None

        self.conv_window_height = None
        self.conv_output_dim = None

        self.hidden_layer_output_dims = None


class SRLNetwork(GradientOptimizable):


    def __init__(self, problem_character = None,
                 nn_architecture = None, trans_mat_prior = None):
        # x shape: [mini-batch size, feature-dim].
        # In this problem [mini-batch feature-dim]

        if ( problem_character is None or nn_architecture is None):

            raise Exception("both problem and architecture must be provided")

        word_num = problem_character['word_num']
        POS_type_num = problem_character['POS_type_num']

        dist_to_verb_num = problem_character['dist_to_verb_num']
        dist_to_word_num = problem_character['dist_to_word_num']

        # 1,word vector
        #   output shape: (batch size,sentence_len, word_feature_num)
        self.word_embedding_layer = LookupTableLayer(
            table_size = word_num,
            feature_num = nn_architecture.word_feature_dim
        )

        # 3,word POS tag vector
        #   output shape: (batch size,sentence_len, POS_feature_num)
        self.pos_embedding_layer = LookupTableLayer(
            table_size = POS_type_num,
            feature_num = nn_architecture.pos_feature_dim,
        )

#            self.loc_embedding_layer = LookupTableLayer(
#                table_size = loc_type_num,
#                feature_num = nn_architecture.dist_feature_dim,
#            )


        # 5,distance tag vector
        #   output shape: (batch size,sentence_len, POS_feature_num)
        self.locdiff_word_embedding_layer = LookupTableLayer(
            table_size = dist_to_word_num,
            feature_num = nn_architecture.dist_feature_dim,
        )

        self.locdiff_verb_embedding_layer = LookupTableLayer(
            table_size = dist_to_verb_num,
            feature_num = nn_architecture.dist_feature_dim,
        )

        conv_input_dim = nn_architecture.word_feature_dim * 3 + \
            nn_architecture.pos_feature_dim * 3 + \
            nn_architecture.dist_feature_dim * 4



        conv_shape = (nn_architecture.conv_output_dim,
                           1,
                           nn_architecture.conv_window_height,
                           conv_input_dim)
        self.conv_layer = Conv1DMaxPoolLayer(
            activator_type="sigmoid",
            tensor_shape = conv_shape)


        self.embedding_conv_layers = [self.word_embedding_layer,
            self.pos_embedding_layer,
            self.locdiff_word_embedding_layer,
            self.locdiff_verb_embedding_layer,
            self.conv_layer]

        input_dim = nn_architecture.conv_output_dim
        self.perception_layers = []
        for idx, output_dim in enumerate(nn_architecture.hidden_layer_output_dims):

            hidden_layer = PerceptionLayer(
                input_dim = input_dim,
                output_dim = output_dim,
                activator_type = "sigmoid")

            self.perception_layers.append(hidden_layer)
            input_dim = output_dim

        self.cost = create_cost("cross_entropy")

        self.__make_behaviors()
            # self.output_layer = PathTransitionLayer('output',
            #                             class_num=SRL_type_num,
            #                             trans_mat_prior= trans_mat_prior)
#            self.output_layer = SoftMaxLayer(n_in= nn_architecture.hidden_layer_output_dims[-1],
#                    n_out = SRL_type_num,)



    def __make_behaviors(self):

        X = theano.tensor.matrix()
        y = theano.tensor.matrix()

        # X.sentence_word_id = [] #当前句子的全局word id 列表
        # X.sentence_pos_id = [] #当前句子的全局词性 id 列表
        # 
        # #每个<word, verb> pair 一条记录
        # X.cur_word_id = []  # 当前word 的词id
        # X.cur_verb_id = []  # 当前verb 的词id
        # X.cur_word_pos_id = []  # 当前word的词性 id
        # X.cur_verb_pos_id = []  # 当前verb的词性 id
        # X.cur_word_loc_id = []  # 当前word的位置 id   # NOT IN USE
        # X.cur_verb_loc_id = []  # 当前verb的位置 id   # NOT IN USE
        # X.cur_word2verb_dist_id = []  # 当前word 到 当前verb的位置距离 id
        # X.cur_verb2word_dist_id = []  # 当前verb 到 当前word的位置距离 id
        # X.other_word2verb_dist_id = []  # 其他word 到当前verb的位置距离 id  # NOT IN USE
        # X.other_word2word_dist_id = []  # 其他word 到当前word的位置距离 id  # NOT IN USE


        wordvec = self.word_embedding_layer.output(
            inputs = X.cur_word_id #word_id_input
        )

        verbvec = self.word_embedding_layer.output(
            inputs = X.cur_word_id #verb_id_input
        )

        wordPOSvec = self.pos_embedding_layer.output(
            inputs = X.cur_word_pos_id #word_pos_input
        )

        verbPOSvec = self.pos_embedding_layer.output(
            inputs = X.cur_verb_pos_id #verb_pos_input
        )

#        wordlocvec = self.loc_embedding_layer.output(
#            inputs = word_loc_input,
#        )

#        verblocvec = self.loc_embedding_layer.output(
#            inputs = verb_loc_input,
#        )

        locdiff_word2verb_vec = self.locdiff_verb_embedding_layer.output(
            inputs = X.cur_word2verb_dist_id
        )

        locdiff_verb2word_vec = self.locdiff_word_embedding_layer.output(
            inputs = X.cur_verb2word_dist_id
        )

        sentence_word_vec = self.word_embedding_layer.output(
            inputs = X.sentence_word_id,
        )

        sentence_pos_vec = self.pos_embedding_layer.output(
            inputs = X.sentence_pos_id,
        )

        other_loc2word_vec = self.locdiff_word_embedding_layer.output(
           inputs = X.other_word2word_dist_id
        )

        other_loc2verb_vec = self.locdiff_verb_embedding_layer.output(
           inputs = X.other_word2verb_dist_id
        )

        batch_size = len(X.cur_word_id)

        conv_input_feature = T.concatenate(

            (
                wordvec.dimshuffle(0,"x", "x",1).repeat(X.sentence_len, axis=2),
                verbvec.dimshuffle(0,"x", "x",1).repeat(X.sentence_len, axis=2),
                wordPOSvec.dimshuffle(0,"x", "x",1).repeat(X.sentence_len, axis=2),
                verbPOSvec.dimshuffle(0,"x", "x",1).repeat(X.sentence_len, axis=2),
                locdiff_word2verb_vec.dimshuffle(0,"x", "x",1).repeat(X.sentence_len, axis=2),
                locdiff_verb2word_vec.dimshuffle(0,"x", "x",1).repeat(X.sentence_len, axis=2),
                sentence_word_vec.dimshuffle("x", "x", 0, 1).repeat(batch_size, axis=0),
                sentence_pos_vec.dimshuffle("x", "x", 0, 1).repeat(batch_size, axis=0),
                other_loc2word_vec.dimshuffle(0, "x", 1, 2),
                other_loc2verb_vec.dimshuffle(0, "x", 1, 2),
            ),
            axis=3
        )

        conv_out = self.conv_layer.output(conv_input_feature).reshape((batch_size, -1))


        hidden_input_feature = conv_out
        for hidden_layer in self.perception_layers:
            hidden_input_feature = hidden_layer.output(hidden_input_feature)

        self.__output_expr = hidden_input_feature
        self.__object_expr =  self.cost(self.__output_expr, y)
        self.__object_func = theano.function([X,y], outputs=self.__object_expr)

        params = self.params()

        grad = T.grad(self.__object_expr, params)

        gradient_vec = []
        for param in grad:
            gradient_vec.append(param.reshape((-1,)))

        self.__gradient_expr = theano.tensor.concatenate(gradient_vec)
        self.__gradient_func = theano.function([X,y], outputs=self.__gradient_expr)


    def get_parameter(self):

        all_layes = self.embedding_conv_layers + self.perception_layers
        param_vec = [layer.get_parameter() for layer in all_layes]

        return numpy.concatenate(param_vec)

    def set_parameter(self, param_vec):

        all_layes = self.embedding_conv_layers + self.perception_layers
        parameter_size_vec = [layer.get_parameter_size() for layer in all_layes]

        start_idx = [0] + list(numpy.cumsum(parameter_size_vec))

        for idx, layer in enumerate(all_layes):

            layer.set_parameter(param_vec[start_idx[idx]:start_idx[idx] + parameter_size_vec[idx]])


    def object(self, X, y = None):

        return self.__object_func(X, y)

    def gradient(self, X, y = None):

        return self.__gradient_func(X, y)

    def params(self):

        all_layes = self.embedding_conv_layers + self.perception_layers

        return list(itertools.chain.from_iterable([layer.params() for layer in all_layes]))



    def __getstate__(self):

        state = dict()
        state['name'] = "srl-machine"
        state['word_embedding_layer'] = self.word_embedding_layer.__getstate__()
        state['word_conv_layer'] = self.word_conv_layer.__getstate__()
        state['pos_embedding_layer'] = self.pos_embedding_layer.__getstate__()
        state['pos_conv_layer'] = self.pos_conv_layer.__getstate__()
        state['loc_embedding_layer'] = self.loc_embedding_layer.__getstate__()
        state['locdiff_word_embedding_layer'] = self.locdiff_word_embedding_layer.__getstate__()
        state['locdiff_word_conv_layer'] = self.locdiff_word_conv_layer.__getstate__()
        state['locdiff_verb_embedding_layer'] = self.locdiff_verb_embedding_layer.__getstate__()
        state['locdiff_verb_conv_layer'] = self.locdiff_verb_conv_layer.__getstate__()

        for idx, hidden_layer in enumerate(self.perception_layers):
            state['hidden_layer_' + str(idx)] = hidden_layer.__getstate__()

        state['output_layer'] = self.output_layer.__getstate__()

        return state

    def __setstate__(self, state):

        assert state['name'] == "srl-machine"

        self.word_embedding_layer = LookupTableLayer()
        self.word_embedding_layer.__setstate__(state["word_embedding_layer"])

        self.pos_embedding_layer = LookupTableLayer()
        self.pos_embedding_layer.__setstate__(state["pos_embedding_layer"])

        self.loc_embedding_layer = LookupTableLayer()
        self.loc_embedding_layer.__setstate__(state["loc_embedding_layer"])

        self.locdiff_word_embedding_layer = LookupTableLayer()
        self.locdiff_word_embedding_layer.__setstate__(state["locdiff_word_embedding_layer"])

        self.locdiff_verb_embedding_layer = LookupTableLayer()
        self.locdiff_verb_embedding_layer.__setstate__(state["locdiff_verb_embedding_layer"])

        self.word_conv_layer = Conv1DMaxPoolLayer()
        self.word_conv_layer.__setstate__(state["word_conv_layer"])

        self.pos_conv_layer = Conv1DMaxPoolLayer()
        self.pos_conv_layer.__setstate__(state["pos_conv_layer"])

        self.locdiff_word_conv_layer = Conv1DMaxPoolLayer()
        self.locdiff_word_conv_layer.__setstate__(state["locdiff_word_conv_layer"])

        self.locdiff_verb_conv_layer = Conv1DMaxPoolLayer()
        self.locdiff_verb_conv_layer.__setstate__(state["locdiff_verb_conv_layer"])



def get_train_func(srl_nn, learning_rate = 0.001, l1_reg = 0, l2_reg = 0 ):

    X = T.matrix("X")
    y = T.ivector("y")

    params = srl_nn.params()

    param_l1_reg = sum([abs(param).sum() for param in params ])
    param_l2_reg = sum([(param **2).sum() for param in params ])

    regularized_cost = srl_nn.cost(X,y) + l1_reg * param_l1_reg + l2_reg * param_l2_reg


    gparams = []
    for param in params:
        gparams.append(T.grad(regularized_cost, param))


    updates = []

    for param, gparam in zip(params, gparams):
        updates.append((param, param - learning_rate * gparam))

    train_func = theano.function(
            inputs=[X,y],
            outputs=regularized_cost,
            updates=updates)

    return train_func


def get_test_func(srl_nn):

    X = T.matrix("X")
    y = T.ivector("y")

    error = srl_nn.error(X,y )

    test_func = theano.function(
            inputs=[X,y],
            outputs=error)

    return test_func

def get_pred_func(srl_nn):

    X = T.matrix("X")

    pred = srl_nn.predict(X)

    test_func = theano.function(
            inputs=[X],
            outputs=pred)

    return test_func

#from theano.compile.ops import as_op

#
# @as_op(itypes=[theano.tensor.lvector],
#        otypes=[theano.tensor.lvector])
# def numpy_unique(a):
#     return numpy.unique(a)
#
# def get_pred_stat_func(srl_nn):
#
#     X = T.matrix("X")
#
#     pred = numpy_unique(srl_nn.predict(X))
#
#     all_same = T.eq(T.prod(pred.shape), T.as_tensor_variable(1))
#
#     test_func = theano.function(
#             inputs=[X],
#             outputs=all_same)
#
#     return test_func

import numpy as np


class NeuralModelHyperParameter(object):

    def __init__(self):

        self.n_epochs = None
        self.learning_rate = None
        self.learning_rate_decay_ratio = None
        self.learning_rate_lowerbound = None
        self.l1_reg = None
        self.l2_reg = None

from knowledge.language.evaluation.srl_evaluate import eval_srl

def train_srl_neural_model(train_problem, valid_problem,
                           nn_architecture,  hyper_param,
                           model_path=None, model_tag=None):


    problem_character = train_problem.get_problem_property()
    trans_mat_prior = train_problem.get_trans_mat_prior()

    srl_nn = SRLNetwork(problem_character, nn_architecture, trans_mat_prior)

    if model_path != None:
        srl_nn.load_model(model_path, model_tag)

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

    validation_frequency = 10000

    total_minibatch = 0

    train_func = get_train_func(srl_nn, hyper_param.learning_rate, hyper_param.l1_reg, hyper_param.l2_reg)

    valid_func = get_test_func(srl_nn)

    #stat_func = get_pred_stat_func(srl_nn)
    pred_func = get_pred_func(srl_nn)

    while (epoch < hyper_param.n_epochs) and (not done_looping):
        epoch = epoch + 1

        minibatch = 0
        for X, y in train_problem.get_data_batch():

            if X[0][0] < 3:
                continue

            start_time = time.clock()


            minibatch_avg_cost= train_func(X.astype("float32"), y.astype('int32'))

            end_time = time.clock()

            minibatch += 1
            total_minibatch += 1
            if minibatch % 100 == 0:

                debug_info = 'epoch {0}.{1}, cost = {2}, time = {3}'.format(epoch,minibatch,minibatch_avg_cost,end_time - start_time)
                print debug_info
            '''
                numpy.savetxt(str(minibatch) +  ".X.txt",
                              numpy.asarray(srl_nn.hidden_output(T.shared(X)).eval()))
                numpy.savetxt(str(minibatch) +  ".y.txt",
                              y)
            '''


            if total_minibatch  % validation_frequency == 0:

#                srl_nn.dump_model('./models/',str(total_minibatch/validation_frequency))

                # compute zero-one loss on validation set
                validation_losses = 0
                sample_num = 0
                validation_pred = []
                validation_label = []
                test_num = 0
                all_same = 0

                same_rate = 0
                test_label_file_path = "test_label_" + str(total_minibatch/validation_frequency) + ".txt"
                pred_label_file_path = "pred_label_" + str(total_minibatch/validation_frequency) + ".txt"

                test_label_file = open(test_label_file_path, "w")
                pred_label_file = open(pred_label_file_path, "w")
                start_time = time.clock()
                for sentence in valid_problem.sentences():
                    test_labels = []
                    pred_labels = []
                    for srl_x, srl_y in valid_problem.get_dataset_for_sentence(sentence):
                        test_labels.append(srl_y)
                        pred_labels.append(pred_func(srl_x.astype("float32")))

                    test_label_str = valid_problem.pretty_srl_label(sentence, test_labels)
                    pred_label_str = valid_problem.pretty_srl_label(sentence, pred_labels)

                    test_label_file.write(test_label_str)
                    pred_label_file.write(pred_label_str)

                test_label_file.close()
                pred_label_file.close()


                valid_result = eval_srl(test_label_file_path, pred_label_file_path)
                valid_info = 'minibatch {0}, validation info {1}% '.format(
                    total_minibatch, valid_result)
                print valid_info


                # # if we got the best validation score until now
                # if validation_losses < best_validation_loss:
                #     #improve patience if loss improvement is good enough
                #     if validation_losses < best_validation_loss *  \
                #            improvement_threshold:
                #         patience = max(patience, epoch * patience_increase)
                #
                #     best_validation_loss = validation_losses
                #     best_iter = epoch
                #
                # if patience <= epoch:
                #     done_looping = True
                #     break

        hyper_param.learning_rate *= hyper_param.learning_rate_decay_ratio
        if hyper_param.learning_rate <= hyper_param.learning_rate_lowerbound:
            hyper_param.learning_rate = hyper_param.learning_rate_lowerbound

    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i.') %
          (best_validation_loss * 100., epoch))




