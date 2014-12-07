__author__ = ['Sun','Huang']

__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy as np

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
        self.W = theano.shared(value=np.asarray(rng.uniform(low=-2.0, high=2.0, size=(n_in, n_out)),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.asarray(rng.uniform(low=-2.0, high=2.0, size=(n_out,)),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # trasition matrix of class tags
        # A_{i,j} means the transition prob from class i to class j
        # A_{0, i} means the prob of start with class i
        self.tag_trans_matrix = theano.shared(value = np.zeros((n_out + 1 ,n_out ), dtype=theano.config.floatX),
                                              name='tag_trans', borrow = True)

    def cost(self, X, y):

        # pointwise_score shape (batch_size,max_term_per_sent,n_out)
        pointwise_score = T.dot(X, self.W) + self.b
        # y_pred_pointwise shape (batch_size,max_term_per_sent)
        # y_pred_pointwise = T.argmax(pointwise_score, axis=2)

#        self.results,_update = theano.scan(lambda score,y,mask: score[T.arange(max_term_per_sent),y] * mask,
#                       sequences=[self.pointwise_score,self.Y,self.masks])

        #TODO: compute total score of all path (eq, 12, NLP from Scratch)

        sentence_len = pointwise_score.shape[0]
        tag_num = self.tag_trans_matrix.shape[1]
        # comput logadd via eq, 14, NLP from scratch
        # \delta_t(k) is the logadd of all path of terms 1:t that end with class k
        # so, \delta_0(k) = logadd(A(0,k))
        # we are calculating delta_t(k) for t=1:T+1 and k = 1:TagNum+1

        def calculate_delta(s, delta_tm1, tag_num, trans_mat):
            delta = s + T.log(T.sum(T.exp(delta_tm1.dimshuffle('x',0).repeat(tag_num,axis=0) + trans_mat), axis=0))
            return delta

        result, updates = theano.scan(fn = calculate_delta,
                                sequences = pointwise_score,
                                outputs_info= [
                                    dict(
                                        initial = T.log(self.tag_trans_matrix[0, :]),
                                        taps=[-1]
                                    )],
                                non_sequences=[tag_num, self.tag_trans_matrix[1:,:]]
        )
        delta = result[-1]

        def calculate_score_given_path(i, select_score, score_mat, trans_mat, y):
            return select_score + score_mat[i, y[i]] + trans_mat[y[i-1]+1,y[i]]


        result, update = theano.scan(fn = calculate_score_given_path,
                                     sequences = T.arange(1,sentence_len),
                                     outputs_info = [
                                         dict(
                                             initial = self.tag_trans_matrix[0, y[0]] + pointwise_score[0, y[0]],
                                             taps=[-1]
                                         ),
                                        ],
                                     non_sequences=[pointwise_score, self.tag_trans_matrix, y])

        selected_path_score = result[-1]

        return selected_path_score - T.sum(T.exp(delta),axis=0)

    def params(self):
        # parameters of the model
        #self.params = [self.W, self.b, self.tag_trans_matrix]
        return [self.W, self.b, self.tag_trans_matrix]


    def predict(self, X):
        pointwise_score = T.dot(X, self.W) + self.b

        sentence_len = pointwise_score.shape[0]
        tag_num = self.tag_trans_matrix.shape[1]
        # Viterbi algorithm
        # \delta_t(k) is the max  of all path of terms 1:t that end with class k
        # so, \delta_0(k) = A(0,k)
        # we are calculating delta_t(k) for t=1:T+1 and k = 1:TagNum+1
        # m_t is the class that achieve max obj of the path end at i-th word

        def viterbi_algo(current_word_scores, delta_tm1, tag_num, trans_mat):

            delta = current_word_scores + T.max(T.sum(delta_tm1.dimshuffle('x',0).repeat(tag_num,axis=0) + trans_mat,axis=0 ), axis=0)
            return delta
        
        (delta, updates) = theano.scan(fn = viterbi_algo,
                                sequences = pointwise_score,
                                outputs_info= [
                                    dict(
                                        initial = T.log(self.tag_trans_matrix[0, :]),
                                        taps =[-1]
                                    )],
                                non_sequences=[tag_num, self.tag_trans_matrix[1:,:]]
        )

        return T.argmax(delta, axis= 1)

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



def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()
