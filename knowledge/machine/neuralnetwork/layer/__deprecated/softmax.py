"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'


import numpy

import theano
import theano.tensor as T

from knowledge.machine.neuralnetwork.random import get_numpy_rng
from knowledge.machine.neuralnetwork.layer.layer import Layer
class SoftMaxLayer(Layer):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self,
                 input_dim = None, output_dim = None,
                 W = None, b = None):
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
#        param_valid = (W is not None and b is not None ) or (n_in is not None and n_out is not None)

#        assert param_valid, "The construction param is not valid"

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


        if W:
            self.W = theano.shared(value=W.astype(theano.config.floatX),
                                name='softmax_W_%s' % (self.name), borrow=True)
        elif (n_in is not None and n_out is not None) :
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            rng = get_numpy_rng()
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values,
                                    name='softmax_W_%s' % (self.name), borrow=True)

        if b:
            self.b = theano.shared(value=b.astype(theano.config.floatX),
                                name='softmax_b_%s' % (self.name), borrow=True)
        elif (n_in is not None and n_out is not None) :
            # initialize the baises b as a vector of n_out 0s
            self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='softmax_b_%s' % (self.name), borrow=True)


    def __getstate__(self):

        state = dict()
        state['type'] = "soft-max"
        state['W'] = self.W.get_value()
        state['b'] = self.b.get_value()

        return state

    def __setstate__(self, state):

        self.W = theano.shared(value=state['W'].astype(theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=state['b'].astype(theano.config.floatX),
                                name='b', borrow=True)

    def params(self):
        # parameters of the model
        return [self.W, self.b]

    def output(self, X):

        return T.nnet.softmax(T.dot(X, self.W) + self.b)

    def predict(self, X):
        p_y_given_x = self.output(X)

        return T.argmax(p_y_given_x, axis=1)


    def cost(self, X, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.

        # compute vector of class-membership probabilities in symbolic form
        p_y_given_x = self.output(X)

        # compute prediction as class whose probability is maximal in
        # symbolic form
#        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

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


    # def predict(self, class_id):
    #
    #     prob = self.p_y_given_x[:, class_id]
    #
    #     return prob


