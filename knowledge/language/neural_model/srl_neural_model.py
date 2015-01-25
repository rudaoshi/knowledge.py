__author__ = 'Huang'

import os, sys
import time

import theano.tensor as T
import theano

from knowledge.machine.neuralnetwork.layer.perception import PerceptionLayer
from knowledge.machine.neuralnetwork.layer.conv1d_layer import Conv1DLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.machine.neuralnetwork.layer.softmax import SoftMaxLayer
from knowledge.machine.neuralnetwork.layer.path_transition_layer import PathTransitionLayer
from knowledge.util.conlleval import append_prop_text, conlleval
from knowledge.util.mis import tmpfile, cleantmp

import numpy
numpy.set_printoptions(threshold='nan')
class SRLNetowrkArchitecture(object):

    def __init__(self):

        self.word_feature_dim = None
        self.pos_feature_dim = None
        self.dist_feature_dim = None

        self.conv_window_height = None
        self.conv_output_dim = None

        self.hidden_layer_output_dims = None


class SRLNeuralLanguageModel(object):


    def __init__(self, problem_character = None,
                 nn_architecture = None, trans_mat_prior = None):
        # x shape: [mini-batch size, feature-dim].
        # In this problem [mini-batch feature-dim]

        if ( problem_character is not None and nn_architecture is not None):

            word_num = problem_character['word_num']
            POS_type_num = problem_character['POS_type_num']
            SRL_type_num = problem_character['SRL_type_num']
#            loc_type_num = problem_character['loc_type_num']

            dist_to_verb_num = problem_character['dist_to_verb_num']
            dist_to_word_num = problem_character['dist_to_word_num']

            # 1,word vector
            #   output shape: (batch size,sentence_len, word_feature_num)
            self.word_embedding_layer = LookupTableLayer('word_embedding',
                table_size = word_num,
                feature_num = nn_architecture.word_feature_dim
            )

            # 3,word POS tag vector
            #   output shape: (batch size,sentence_len, POS_feature_num)
            self.pos_embedding_layer = LookupTableLayer('pos_embedding',
                table_size = POS_type_num,
                feature_num = nn_architecture.pos_feature_dim,
            )

#            self.loc_embedding_layer = LookupTableLayer(
#                table_size = loc_type_num,
#                feature_num = nn_architecture.dist_feature_dim,
#            )


            # 5,distance tag vector
            #   output shape: (batch size,sentence_len, POS_feature_num)
            self.locdiff_word_embedding_layer = LookupTableLayer('locdiff_word',
                table_size = dist_to_word_num,
                feature_num = nn_architecture.dist_feature_dim,
            )

            self.locdiff_verb_embedding_layer = LookupTableLayer('locdiff_verb',
                table_size = dist_to_verb_num,
                feature_num = nn_architecture.dist_feature_dim,
            )

            word_conv_shape = (nn_architecture.conv_output_dim, 1, nn_architecture.conv_window_height, nn_architecture.word_feature_dim)
            self.word_conv_layer = Conv1DLayer('word_conv', tensor_shape = word_conv_shape)
            pos_conv_shape = (nn_architecture.conv_output_dim, 1, nn_architecture.conv_window_height, nn_architecture.pos_feature_dim)
            self.pos_conv_layer = Conv1DLayer('pos_conv', tensor_shape = pos_conv_shape)

            locdiff_word_conv_shape = (nn_architecture.conv_output_dim, 1, nn_architecture.conv_window_height, nn_architecture.dist_feature_dim)
            self.locdiff_word_conv_layer = Conv1DLayer('locdiff_word_conv', tensor_shape = locdiff_word_conv_shape)
            locdiff_verb_conv_shape = (nn_architecture.conv_output_dim, 1, nn_architecture.conv_window_height, nn_architecture.dist_feature_dim)
            self.locdiff_verb_conv_layer = Conv1DLayer('locdiff_verb_conv', tensor_shape = locdiff_verb_conv_shape)

            # add max pool here

            input_dim = nn_architecture.word_feature_dim * 2 + \
                nn_architecture.pos_feature_dim * 2 + \
                nn_architecture.dist_feature_dim * 2 + \
                nn_architecture.conv_output_dim * 4

            self.hidden_layers = []
            for idx, output_dim in enumerate(nn_architecture.hidden_layer_output_dims):

                hidden_layer = PerceptionLayer('hidden_%d' % (idx),
                    n_in = input_dim,
                    n_out = output_dim,
                    activation=T.tanh)

                self.hidden_layers.append(hidden_layer)
                input_dim = output_dim

#            last_hidden_layer = SoftMaxLayer(n_in= nn_architecture.hidden_layer_output_dims[-1],
#                    n_out = SRL_type_num,)
            last_hidden_layer = PerceptionLayer('last_hidden',
                     n_in = nn_architecture.hidden_layer_output_dims[-1],
                     n_out = SRL_type_num,
                     activation=T.nnet.sigmoid)
            self.hidden_layers.append(last_hidden_layer)

            self.output_layer = PathTransitionLayer('output',
                                        class_num=SRL_type_num,
                                        trans_mat_prior= trans_mat_prior)
#            self.output_layer = SoftMaxLayer(n_in= nn_architecture.hidden_layer_output_dims[-1],
#                    n_out = SRL_type_num,)

    def dump_model(self, model_file_folder, tag=None):
        assert isinstance(model_file_folder, str)
        assert tag == None or isinstance(tag, str)
        for idx, x in enumerate(self.params()):
            if tag != None:
                model_filename = os.path.join(model_file_folder,"%s_%s_param" % (tag, x.name))
            else:
                model_filename = os.path.join(model_file_folder,"%s_param" % (x.name))
            value = x.get_value(borrow = True)
            print type(value), value.shape
            # numpy.savetxt(model_filename, value)
            numpy.save(model_filename, value)

    def load_model(self, model_file_folder, tag=None):
        assert isinstance(model_file_folder, str)
        assert tag == None or isinstance(tag, str)
        for idx, x in enumerate(self.params()):
            if tag != None:
                model_filename = os.path.join(model_file_folder,"%s_%s_param.npy" % (tag, x.name))
            else:
                model_filename = os.path.join(model_file_folder,"%s_param.npy" % (x.name))
            npdata = numpy.load(model_filename)
            x.set_value(npdata, borrow=True)

    def hidden_output(self, X):



        # X.append(
        #          [sentence_len, wd.id, verb.id,
        #           PosTags.POSTAG_ID_MAP[sentence.get_word_property(verb_loc).pos],
        #           PosTags.POSTAG_ID_MAP[sentence.get_word_property(word_loc).pos],
        #           LocTypes.get_loc_id(word_loc),
        #           LocTypes.get_loc_id(verb_loc),
        #           LocDiffToWordTypes.get_locdiff_id(verb_loc - word_loc),
        #           LocDiffToVerbTypes.get_locdiff_id(word_loc - verb_loc )
        #          ] +  loc_to_word + loc_to_verb
        #          )


        sentence_len = X[0,0].astype("int32")
        word_id_input = X[:, 1].astype("int32")
        verb_id_input = X[:, 2].astype("int32")
        word_pos_input = X[:, 3].astype("int32")
        verb_pos_input = X[:, 4].astype("int32")
        word_loc_input = X[:, 5].astype("int32")
        verb_loc_input = X[:, 6].astype("int32")
        locdiff_verb2word = X[:,7].astype("int32")
        locdiff_word2verb = X[:, 8].astype("int32")

        sentence_wordid_input = X[0:sentence_len, 1].astype("int32")  # sentence means globally same for this batch
        sentence_pos_input = X[0:sentence_len, 3].astype("int32")
        other_loc2word_input = X[:, 9:9+sentence_len].astype("int32") # other means specific for this sample
        other_loc2verb_input = X[:, 9+sentence_len: 9+2*sentence_len].astype("int32")


        wordvec = self.word_embedding_layer.output(
            inputs = word_id_input
        )

        verbvec = self.word_embedding_layer.output(
            inputs = verb_id_input
        )

        wordPOSvec = self.pos_embedding_layer.output(
            inputs = word_pos_input
        )

        verbPOSvec = self.pos_embedding_layer.output(
            inputs = verb_pos_input
        )

#        wordlocvec = self.loc_embedding_layer.output(
#            inputs = word_loc_input,
#        )

#        verblocvec = self.loc_embedding_layer.output(
#            inputs = verb_loc_input,
#        )


        locdiff_verb2word_vec = self.locdiff_word_embedding_layer.output(
            inputs = locdiff_verb2word
        )

        locdiff_word2verb_vec = self.locdiff_verb_embedding_layer.output(
            inputs = locdiff_word2verb
        )

        sentence_word_vec = self.word_embedding_layer.output(
            inputs = sentence_wordid_input,
        )

        sentence_pos_vec = self.pos_embedding_layer.output(
            inputs = sentence_pos_input,
        )

        other_loc2word_vec = self.locdiff_word_embedding_layer.output(
            inputs = other_loc2word_input,
            tensor_output=True
        )

        other_loc2verb_vec = self.locdiff_verb_embedding_layer.output(
            inputs = other_loc2verb_input,
            tensor_output=True
        )

        batch_size = X.shape[0].astype('int32')

        # conv input size = [mini-batch size, number of input feature maps, image height, image width].
        # conv output size = [  mini-batch size, number of output feature maps,
        #                       image height - tensor_height + 1, image width - tensor_width + 1].

        # for sentence level input, mini_batch_size = 1, input_feature_map num = 1,
        #                            height = sentence_len, width = feature_dim
        sentence_word_conv = self.word_conv_layer.output(sentence_word_vec.dimshuffle("x","x",0,1))
        sentence_word_conv_max = T.max(sentence_word_conv, axis=2)
        sentence_word_conv_feature = sentence_word_conv_max.reshape((1, sentence_word_conv_max.shape[1] * sentence_word_conv_max.shape[2])).repeat(batch_size,axis=0)

        sentence_pos_conv = self.pos_conv_layer.output(sentence_pos_vec.dimshuffle("x","x",0,1))
        sentence_pos_conv_max = T.max(sentence_pos_conv, axis=2)
        sentence_pos_conv_feature = sentence_pos_conv_max.reshape((1, sentence_pos_conv_max.shape[1] * sentence_pos_conv_max.shape[2])).repeat(batch_size,axis=0)



        # for other level input, minibatch size = batch size, input_feature_map num = 1
        #                        height = sentence_len, width = feature_dim
        other_loc2word_cov = self.locdiff_word_conv_layer.output(other_loc2word_vec.dimshuffle(0,"x", 1, 2))
        other_loc2word_conv_max =  T.max(other_loc2word_cov, axis=2)
        other_loc2word_conv_feature = other_loc2word_conv_max.reshape((batch_size, other_loc2word_conv_max.shape[1] * other_loc2word_conv_max.shape[2]))

        other_loc2verb_cov = self.locdiff_word_conv_layer.output(other_loc2verb_vec.dimshuffle(0,"x", 1, 2))
        other_loc2verb_conv_max =  T.max(other_loc2verb_cov, axis=2)
        other_loc2verb_conv_feature = other_loc2verb_conv_max.reshape((batch_size,  other_loc2verb_conv_max.shape[1] * other_loc2verb_conv_max.shape[2]))

        hidden_input_feature = T.concatenate(
            (
                wordvec,
                verbvec,
                wordPOSvec,
                verbPOSvec,
#                wordlocvec,
#                verblocvec,
                locdiff_verb2word_vec,
                locdiff_word2verb_vec,
                sentence_word_conv_feature,
                sentence_pos_conv_feature,
                other_loc2word_conv_feature,
                other_loc2verb_conv_feature
            ),
            axis = 1
        )

        for hidden_layer in self.hidden_layers:
            hidden_input_feature = hidden_layer.output(hidden_input_feature)


        return hidden_input_feature

    def output(self, X):

        hidden_output = self.hidden_output(X)
        return self.output_layer.output(hidden_output)

    def cost(self, X, y):
        hidden_output = self.hidden_output(X)
        return self.output_layer.cost(hidden_output, y)

    def error(self, X,y ):
        hidden_output = self.hidden_output(X)
        return self.output_layer.error(hidden_output, y)

    def predict(self, X):

        hidden_output = self.hidden_output(X)
        return self.output_layer.predict(hidden_output)

    def params(self):

        params =  self.word_embedding_layer.params() \
                + self.word_conv_layer.params() \
                + self.pos_embedding_layer.params() \
                + self.pos_conv_layer.params() \
                + self.locdiff_word_embedding_layer.params() \
                + self.locdiff_word_conv_layer.params() \
                + self.locdiff_verb_embedding_layer.params() \
                + self.locdiff_verb_conv_layer.params()
#                + self.loc_embedding_layer.params() \

        for hidden_layer in self.hidden_layers:
            params.extend(hidden_layer.params())

        params.extend(self.output_layer.params())

        return params


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

        for idx, hidden_layer in enumerate(self.hidden_layers):
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

        self.word_conv_layer = Conv1DLayer()
        self.word_conv_layer.__setstate__(state["word_conv_layer"])

        self.pos_conv_layer = Conv1DLayer()
        self.pos_conv_layer.__setstate__(state["pos_conv_layer"])

        self.locdiff_word_conv_layer = Conv1DLayer()
        self.locdiff_word_conv_layer.__setstate__(state["locdiff_word_conv_layer"])

        self.locdiff_verb_conv_layer = Conv1DLayer()
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

from theano.compile.ops import as_op


@as_op(itypes=[theano.tensor.lvector],
       otypes=[theano.tensor.lvector])
def numpy_unique(a):
    return numpy.unique(a)

def get_pred_stat_func(srl_nn):

    X = T.matrix("X")

    pred = numpy_unique(srl_nn.predict(X))

    all_same = T.eq(T.prod(pred.shape), T.as_tensor_variable(1))

    test_func = theano.function(
            inputs=[X],
            outputs=all_same)

    return test_func

import numpy as np


class NeuralModelHyperParameter(object):

    def __init__(self):

        self.n_epochs = None
        self.learning_rate = None
        self.learning_rate_decay_ratio = None
        self.learning_rate_lowerbound = None
        self.l1_reg = None
        self.l2_reg = None



def train_srl_neural_model(train_problem, valid_problem, nn_architecture,  hyper_param, model_path=None, model_tag=None):


    problem_character = train_problem.get_problem_property()
    trans_mat_prior = train_problem.get_trans_mat_prior()

    srl_nn = SRLNeuralLanguageModel(problem_character, nn_architecture, trans_mat_prior)

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

    validation_frequency = 20000

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
                f_golden = tmpfile('golden')
                f_pred = tmpfile('pred')

                srl_nn.dump_model('./models/',str(total_minibatch/validation_frequency))

                # compute zero-one loss on validation set
                validation_losses = 0
                sample_num = 0
                validation_pred = []
                validation_label = []
                test_num = 0
                all_same = 0

                same_rate = 0

                start_time = time.clock()
                for valid_X, valid_y, in valid_problem.get_data_batch():
                    test_num += 1

                    error = valid_func(valid_X.astype("float32") ,valid_y.astype('int32'))
                    valid_pred = pred_func(valid_X.astype("float32"))
                    append_prop_text(f_golden, f_pred, valid_X, valid_y, valid_pred)
                    #same_predict = stat_func(valid_X.astype("float32"))
                    #all_same += same_predict
                    validation_losses += error * valid_X.shape[0]
                    sample_num += valid_X.shape[0]

#                    validation_pred += pred.tolist()
#                    validation_label += valid_y.tolist()

                    #if test_num >= 100:
                    #    break
                end_time = time.clock()
                debug_info = 'valid {0}.{1}, time = {2}'.format(epoch,minibatch,end_time - start_time)
                print debug_info

                if sample_num > 0:
                    validation_losses /= sample_num
                    same_rate = all_same/float(test_num)
#                this_validation_loss = np.mean(validation_losses)
#                f1 = f1_score(np.asarray(validation_label),np.asarray(validation_pred),average='weighted')

                conlleval(f_golden, f_pred)
                valid_info = 'minibatch {0}, validation error {1}% with {2}% same predicts '.format(
                    total_minibatch, validation_losses * 100, same_rate * 100)
                print valid_info


                # if we got the best validation score until now
                if validation_losses < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if validation_losses < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, epoch * patience_increase)

                    best_validation_loss = validation_losses
                    best_iter = epoch

                if patience <= epoch:
                    done_looping = True
                    break

        hyper_param.learning_rate *= hyper_param.learning_rate_decay_ratio
        if hyper_param.learning_rate <= hyper_param.learning_rate_lowerbound:
            hyper_param.learning_rate = hyper_param.learning_rate_lowerbound

    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i.') %
          (best_validation_loss * 100., epoch))




