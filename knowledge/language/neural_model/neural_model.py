__author__ = 'Sun'

import theano
import theano.tensor as T
import numpy
import time
import sys
import os

from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.util.theano_util import shared_dataset


class NeuralLanguageModelCore(object):


    def __init__(self, word_ids, word_num, window_size, feature_num,
                 hidden_layer_size, n_outs, L1_reg = 0.00, L2_reg = 0.0001,
                 numpy_rng = None, theano_rng=None, ):

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg


        self.lookup_table_layer = LookupTableLayer(input = word_ids, table_size = word_num,
                                                   window_size = window_size, feature_num = feature_num)

        self.hidden_layer = HiddenLayer(rng=numpy_rng, input=self.lookup_table_layer.output,
                                       n_in = self.lookup_table_layer.get_output_size(),
                                       n_out = hidden_layer_size,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.output_layer = LogisticRegression(
                                        input=self.hidden_layer.output,
                                        n_in=hidden_layer_size,
                                        n_out=n_outs)



class NeuralLanguageModel(object):



    def __init__(self, word_num, window_size, feature_num,
                 hidden_layer_size, n_outs, L1_reg = 0.00, L2_reg = 0.0001,
                 numpy_rng = None, theano_rng=None, ):

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg


        self.index = T.lscalar()
        self.input = T.imatrix('input')  # the data is presented as rasterized images
        self.label = T.ivector('label')

        self.core = NeuralLanguageModelCore(self.input, word_num, window_size, feature_num,
                 hidden_layer_size, n_outs, L1_reg, L2_reg,
                 numpy_rng, theano_rng)

        self.params = self.core.lookup_table_layer.params() + self.core.hidden_layer.params + self.core.output_layer.params

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.core.lookup_table_layer.embeddings).sum() \
                + abs(self.core.hidden_layer.W).sum() \
                + abs(self.core.output_layer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.core.lookup_table_layer.embeddings ** 2).sum() \
                    + (self.core.hidden_layer.W ** 2).sum() \
                    + (self.core.output_layer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.core.output_layer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.core.output_layer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of

        self.cost = self.negative_log_likelihood(self.label) \
                     + self.L1_reg * self.L1 \
                     + self.L2_reg * self.L2_sqr




    def fit(self, X, y, valid_X, valid_y,  batch_size = 1000, n_epochs = 10000, learning_rate = 0.01,):

        self.gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates = []
        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        for param, gparam in zip(self.params, self.gparams):
            updates.append((param, param - learning_rate * gparam))

        borrow = True
        train_set_X = T.cast(theano.shared(numpy.asarray(X,
                                dtype=theano.config.floatX),
                                 borrow=borrow), "int32")
        train_set_y = T.cast(theano.shared(numpy.asarray(y,
                                dtype=theano.config.floatX),
                                 borrow=borrow), "int32")

        valid_set_X = T.cast(theano.shared(numpy.asarray(valid_X,
                                dtype=theano.config.floatX),
                                 borrow=borrow), "int32")
        valid_set_y = T.cast(theano.shared(numpy.asarray(valid_y,
                                dtype=theano.config.floatX),
                                 borrow=borrow), "int32")

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[self.index], outputs=self.cost,
                updates=updates,
                givens={
                    self.input: train_set_X[self.index * batch_size:(self.index + 1) * batch_size],
                    self.label: train_set_y[self.index * batch_size:(self.index + 1) * batch_size]
                })
        n_train_batches = X.shape[0] / batch_size

        validate_model = theano.function(inputs=[self.index],
            outputs=self.cost,
            givens={
                self.input: valid_set_X[self.index * batch_size:(self.index + 1) * batch_size],
                self.label: valid_set_y[self.index * batch_size:(self.index + 1) * batch_size]
            })
        n_valid_batches = valid_X.shape[0] / batch_size


        ###############
        # TRAIN MODEL #
        ###############
        print '... training'

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
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
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if True:#(iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                         (epoch, minibatch_index + 1, n_train_batches,
                          this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                    if patience <= iter:
                        done_looping = True
                        break

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

