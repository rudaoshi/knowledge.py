"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import os
import sys
import time
import numpy

import theano
import theano.tensor as T

from knowledge.machine.neuralnetwork.random import get_numpy_rng

class PerceptionLayer(object):
    def __init__(self, W=None, b=None, n_in = None, n_out = None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        if W is None and n_in is not None and n_out is not None :
            rng = get_numpy_rng()
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None and n_in is not None and n_out is not None :
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.activation = activation

    def output(self, X):

        lin_output = T.dot(X, self.W) + self.b
        return (lin_output if self.activation is None
                       else self.activation(lin_output))

    def params(self):
        # parameters of the model
        return [self.W, self.b]

    def __getstate__(self):

        state = dict()
        state['name'] = "perception"
        state['W'] = self.W.get_value()
        state['b'] = self.b.get_value()

        if self.activation is None :
            state['activation'] = "linear"
        elif self.activation == T.tanh:
            state['activation'] = "tanh"
        elif self.activation == T.nnet.sigmoid:
            state['activation'] = "sigmoid"

        return state

    def __setstate__(self, state):

        self.W = theano.shared(value=state['W'].astype(theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=state['b'].astype(theano.config.floatX),
                                name='b', borrow=True)

        if state['activation'] == "linear":
            self.activation = None
        elif state['activation'] == "tanh":
            self.activation = T.tanh
        elif state['activation'] == "sigmoid":
            self.activation = T.nnet.sigmoid
        else:
            raise Exception("Unknown activation type")
