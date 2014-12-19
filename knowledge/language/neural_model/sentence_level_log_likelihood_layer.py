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


class SentenceLevelLogLikelihoodLayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out):
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
        self.tag_trans_matrix = theano.shared(value = np.zeros((n_out + 1 ,n_out ), dtype=theano.config.floatX),
                                              name='tag_trans', borrow = True)

    def cost(self, X, y):

        # pointwise_score shape (batch_size,max_term_per_sent,n_out)
        pointwise_score = X #T.dot(X, self.W) + self.b
        # y_pred_pointwise shape (batch_size,max_term_per_sent)
        # y_pred_pointwise = T.argmax(pointwise_score, axis=2)

#        self.results,_update = theano.scan(lambda score,y,mask: score[T.arange(max_term_per_sent),y] * mask,
#                       sequences=[self.pointwise_score,self.Y,self.masks])

        #TODO: compute total score of all path (eq, 12, NLP from Scratch)

        sentence_len = pointwise_score.shape[0]
        tag_num = self.tag_trans_matrix.shape[1].astype('int32')
        # comput logadd via eq, 14, NLP from scratch
        # \delta_t(k) is the logadd of all path of terms 1:t that end with class k
        # so, \delta_0(k) = logadd(A(0,k))
        # we are calculating delta_t(k) for t=1:T+1 and k = 1:TagNum+1

        trans_mat = T.nnet.softmax(self.tag_trans_matrix)
        def calculate_delta(s, delta_tm1, tag_num, trans_mat):

            sum_mat = delta_tm1.dimshuffle('x',0).T.repeat(tag_num,axis=1) + trans_mat

            max_sum_mat = T.max(sum_mat)

            delta = s + max_sum_mat + T.log(
                T.sum(T.exp(sum_mat - max_sum_mat), axis=0)
            )

            return delta

        result, updates = theano.scan(fn = calculate_delta,
                                sequences = pointwise_score[1:,:],
                                outputs_info= [
                                    dict(
                                        initial = trans_mat[0, :] + pointwise_score[0,:],
                                        taps=[-1]
                                    )],
                                non_sequences=[tag_num, trans_mat[1:,:]]
        )
        # theano.printing.Print('Trans')(
        delta = result[-1]

        def calculate_score_given_path(i, select_score, score_mat, trans_mat, y):
            return select_score + score_mat[i, y[i]] + trans_mat[y[i-1]+1,y[i]]


        result, update = theano.scan(fn = calculate_score_given_path,
                                     sequences = T.arange(1,sentence_len),
                                     outputs_info = [
                                         dict(
                                             initial = trans_mat[0, y[0]] + pointwise_score[0, y[0]],
                                             taps=[-1]
                                         ),
                                        ],
                                     non_sequences=[pointwise_score, trans_mat, y])

        selected_path_score = result[-1]

        max_delta = T.max(delta)

        logadd = max_delta + T.log(T.sum(T.exp(delta - max_delta),axis=0))

        return logadd - selected_path_score

    def params(self):
        # parameters of the model
        #self.params = [self.W, self.b, self.tag_trans_matrix]
        return [] #[ self.tag_trans_matrix]


    def predict(self, X):
        pointwise_score = X # T.dot(X, self.W) + self.b

        sentence_len = pointwise_score.shape[0]
        tag_num = self.tag_trans_matrix.shape[1].astype('int32')
        # Viterbi algorithm
        # \delta_t(k) is the max  of all path of terms 1:t that end with class k
        # so, \delta_0(k) = A(0,k)
        # we are calculating delta_t(k) for t=1:T+1 and k = 1:TagNum+1
        # m_t is the class that achieve max obj of the path end at i-th word
        trans_mat = T.nnet.softmax(self.tag_trans_matrix)
        def viterbi_algo(current_word_scores, delta_tm1, tag_num, trans_mat):

            delta = current_word_scores + T.max(T.sum(delta_tm1.dimshuffle('x',0).T.repeat(tag_num,axis=1) + trans_mat,axis=0 ), axis=0)
            return delta

        delta1 = trans_mat[0, :] + pointwise_score[0,:]
        (delta, updates) = theano.scan(fn = viterbi_algo,
                                sequences = pointwise_score[1:,:],
                                outputs_info= [
                                    dict(
                                        initial = delta1,
                                        taps =[-1]
                                    )],
                                non_sequences=[tag_num, trans_mat[1:,:]]
        )

        return T.argmax(T.concatenate([delta1.dimshuffle('x',0),delta]), axis= 1)

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

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(y_pred, y))
        else:
            raise NotImplementedError()

