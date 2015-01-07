__author__ = ['Sun','Huang']

__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time
import warnings
import numpy as np
warnings.simplefilter("ignore", DeprecationWarning)

import theano
import theano.tensor as T
from knowledge.machine.neuralnetwork.random import get_numpy_rng


class BatchPathTransitionLayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, class_num):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """


        rng = get_numpy_rng()
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
#        self.W = theano.shared(value=np.asarray(rng.uniform(low=-2.0, high=2.0, size=(n_in, n_out)),
#                                                 dtype=theano.config.floatX),
#                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
#        self.b = theano.shared(value=np.asarray(rng.uniform(low=-2.0, high=2.0, size=(n_out,)),
#                                                 dtype=theano.config.floatX),
#                               name='b', borrow=True)

        # trasition matrix of class tags
        # A_{i,j} means the transition prob from class i to class j
        # A_{0, i} means the prob of start with class i
#        self.tag_trans_matrix = theano.shared(value = np.asarray(rng.uniform(low=-2.0, high=2.0, size=(class_num + 1 ,class_num )),
#                                                 dtype=theano.config.floatX),
#                                              name='tag_trans', borrow = True)
        self.tag_trans_matrix = theano.shared(value = np.zeros((class_num + 1 ,class_num ), dtype=theano.config.floatX),
                                              name='tag_trans', borrow = True)

    def cost(self, X, y):

        sentence_len = X[0,0].astype("int32")

        # X and y are {sentence_len} group of paths

        # pointwise_score shape (batch_size,max_term_per_sent,n_out)
        pointwise_score = X #T.dot(X, self.W) + self.b

        # y_pred_pointwise shape (batch_size,max_term_per_sent)
        # y_pred_pointwise = T.argmax(pointwise_score, axis=2)

#        self.results,_update = theano.scan(lambda score,y,mask: score[T.arange(max_term_per_sent),y] * mask,
#                       sequences=[self.pointwise_score,self.Y,self.masks])

        #TODO: compute total score of all path (eq, 12, NLP from Scratch)


        tag_num = self.tag_trans_matrix.shape[1].astype('int32')
        # comput logadd via eq, 14, NLP from scratch
        # \delta_t(k) is the logadd of all path of terms 1:t that end with class k
        # so, \delta_0(k) = logadd(A(0,k))
        # we are calculating delta_t(k) for t=1:T+1 and k = 1:TagNum+1

        trans_mat = self.tag_trans_matrix # T.nnet.softmax(self.tag_trans_matrix)
        def calculate_delta(i, delta_tm1, score_mat, sentence_len, tag_num, trans_mat_):

            if i % sentence_len == 0:
                return trans_mat[0,:] + score_mat[i,:]
            else:
                sum_mat = delta_tm1.dimshuffle('x',0).T.repeat(tag_num,axis=1) + trans_mat_[1:,:]

                max_sum_mat = T.mean(sum_mat)

                delta = score_mat[i,:] + max_sum_mat + T.log(
                    T.sum(T.exp(sum_mat - max_sum_mat), axis=0)
                )

                return delta

        result, updates = theano.scan(fn = calculate_delta,
                                sequences = T.arange(0, pointwise_score.shape[0]),
                                outputs_info= [
                                    dict(
                                        initial = np.zeros((tag_num,)),
                                        taps=[-1]
                                    )],
                                non_sequences=[pointwise_score, sentence_len, tag_num, trans_mat]
        )
        # theano.printing.Print('Trans')(
        delta = result[range(0,pointwise_score.shape[0],step=sentence_len)]

        def calculate_score_given_path(i, score_tm1, score_mat, trans_mat_, y):
            if i % sentence_len == 0:
                return trans_mat[0,:] + score_mat[i,:]
            else:
                return score_tm1 + score_mat[i, y[i]] + trans_mat_[y[i-1]+1 ,y[i]]


        results, updates = theano.scan(fn = calculate_score_given_path,
                                     sequences = T.arange(0,pointwise_score.shape[0]),
                                     outputs_info = [
                                         dict(
                                             initial = 0,
                                             taps=[-1]
                                         ),
                                        ],
                                     non_sequences=[pointwise_score, trans_mat, y])

        #result = theano.printing.Print('select_path_score_result')(result)


        selected_path_score = results[range(0,pointwise_score.shape[0],step=sentence_len)]

        max_delta = T.mean(delta, axis=1)

        logadd = max_delta + T.log(T.sum(T.exp(delta - max_delta),axis=1))

        return T.mean(logadd - selected_path_score)

    def params(self):
        # parameters of the model
        #self.params = [self.W, self.b, self.tag_trans_matrix]
        return []#[ self.tag_trans_matrix]


    def predict(self, X):
        pointwise_score = X # T.dot(X, self.W) + self.b

        sentence_len = pointwise_score.shape[0]
        tag_num = self.tag_trans_matrix.shape[1].astype("int32")
        # Viterbi algorithm
        # \delta_t(k) is the max  of all path of terms 1:t that end with class k
        # so, \delta_0(k) = A(0,k)
        # we are calculating delta_t(k) for t=1:T+1 and k = 1:TagNum+1
        # m_t is the class that achieve max obj of the path end at i-th word
        trans_mat = self.tag_trans_matrix # T.nnet.softmax(self.tag_trans_matrix)
        def viterbi_algo(current_word_scores, path_score_tm1, tag_num, trans_mat):

            all_prossible_path_score = path_score_tm1.dimshuffle('x',0).T.repeat(tag_num,axis=1) + trans_mat

            # previous tag that got largest score for tag k
            path_score, track_back = T.max_and_argmax(all_prossible_path_score, axis=0)
            path_score = path_score + current_word_scores
            return [path_score, track_back]

        path_score_1 = trans_mat[0, :] + pointwise_score[0,:]
        ([path_score, track_back], updates) = theano.scan(fn = viterbi_algo,
                                sequences = pointwise_score[1:,:],
                                outputs_info= [
                                    dict(
                                        initial = path_score_1,
                                        taps =[-1]
                                    ),
                                    None],
                                non_sequences=[tag_num, trans_mat[1:,:]]
        )
        track_back = track_back

        def back_track_algo(back_track, later_tag):
            return back_track[later_tag]

        last_answer = path_score[-1].argmax()
        (answer, updates) = theano.scan(fn = back_track_algo,
                                sequences = track_back[::-1,:],
                                outputs_info= [
                                    dict(
                                        initial = last_answer,
                                        taps =[-1]
                                    )],

        )

        answer = T.concatenate([answer[::-1], last_answer.dimshuffle('x')])
        return answer

    def error(self, X, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        y_pred = self.predict(X)
        # check if y has same dimension of y_pred
        assert  y.ndim == y_pred.ndim


        return T.mean(T.neq(y_pred, y))




