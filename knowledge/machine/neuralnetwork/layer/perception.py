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
from knowledge.machine.neuralnetwork.layer.layer import Layer
from knowledge.machine.neuralnetwork.activation.activation_factory import get_activation

class PerceptionLayer(Layer):
    def __init__(self, activator_type="linear",
                 input_dim = None, output_dim = None,
                 W=None, b=None,
                 ):

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

        assert activator_type is not None, "Activation must be provided"
        self.activator_type = activator_type
        self.activator = get_activation(self.activator_type)

        if input_dim is not None and output_dim is not None:

            if W is None:
                rng = get_numpy_rng()
                W = numpy.asarray(rng.uniform(
                        low=-numpy.sqrt(6. / (input_dim + output_dim)),
                        high=numpy.sqrt(6. / (input_dim + output_dim)),
                        size=(input_dim, output_dim)), dtype=theano.config.floatX)
                if self.activator == theano.tensor.nnet.sigmoid:
                    W *= 4
            else:
                assert input_dim == W.shape[0] and input_dim == W.shape[1]

            if b is None:
                b = numpy.zeros((output_dim,), dtype=theano.config.floatX)
            else:
                assert output_dim == b.shape[0]

            self.W = theano.shared(value=W, borrow=True)
            self.b = theano.shared(value=b, borrow=True)
            self.input_dim_, self.output_dim_ = W.shape
        elif W is not None and b is not None:
            self.W = theano.shared(value=W, borrow=True)
            self.b = theano.shared(value=b, borrow=True)
            self.input_dim_, self.output_dim_ = W.shape

        else:
            raise Exception("Perception Layer needs parameter "
                            "in pair of (W,b) or (n_in, n_out) besides activation")


    def input_dim(self):
        return self.input_dim_


    def output_dim(self):
        return self.output_dim_

    def output(self, X, **kwargs):

        return self.activator(T.dot(X, self.W) + self.b)

    def params(self):
        # parameters of the model
        return [self.W, self.b]

    def get_parameter_size(self):


        return self.input_dim_* self.output_dim_ + self.output_dim_

    def get_parameter(self):
        param_vec = []
        param_vec.append(self.W.get_value(borrow=True).reshape((-1,)))
        param_vec.append(self.b.get_value(borrow=True).reshape((-1,)))

        return numpy.concatenate(param_vec)

    def set_parameter(self, parameter_vec):
        W_size = self.input_dim_ * self.output_dim_
        W_shape = (self.input_dim_ , self.output_dim_)
        self.W.set_value(parameter_vec[0 :  W_size].reshape(W_shape),
                              borrow = True)

        b_size = self.output_dim_
        b_shape = (self.output_dim_,)
        self.b.set_value(parameter_vec[W_size : W_size + b_size].reshape(b_shape),
                              borrow = True)


    def __getstate__(self):

        state = dict()
        state['type'] = "perception"
        state['W'] = self.W.get_value()
        state['b'] = self.b.get_value()
        state['input_dim'] = self.input_dim
        state['output_dim'] = self.output_dim
        state['activator_type'] = self.activator_type

        return state

    def __setstate__(self, state):

        assert state['type'] == "perception", "The layer type is not match"

        self.W = theano.shared(value=state['W'].astype(theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=state['b'].astype(theano.config.floatX),
                                name='b', borrow=True)

        self.input_dim = state['input_dim']
        self.output_dim = state['output_dim']
        self.activator_type = state['activator_type']
        self.activator = get_activation(self.activator_type)
