

import os
import sys
import time
import numpy
import itertools
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from knowledge.machine.neuralnetwork.layer.perception import PerceptionLayer
from knowledge.machine.optimization.gradient_optimizable import BatchStocasticGradientOptimizable

from knowledge.machine.neuralnetwork.layer.layer_factory import create_layer
from knowledge.machine.cost.cost_factory import create_cost
from knowledge.util.data_process import chunks

class MultiLayerPerception(BatchStocasticGradientOptimizable):

    def __init__(self, layer_params, cost_param):

        self.layers = []

        for idx, param in enumerate(layer_params):
            layer = create_layer(param)
            self.layers.append(layer)

            if idx > 0:
                assert layer.input_dim() == self.layers[idx-1].output_dim(), \
                    "The layer chain is broken at %d-th layer"%idx

        self.cost = create_cost(cost_param)

    def prepare_learning(self, batch_size):

        self.learning_batch_size = batch_size

        X = theano.tensor.matrix("X")
        y = theano.tensor.vector("y")

        batch_id = theano.tensor.iscalar('i')

        self.chunk_X = theano.shared(numpy.zeros((batch_size, self.layers[0].input_dim()), dtype = theano.config.floatX))
        self.chunk_y = theano.shared(numpy.zeros((batch_size, ), dtype = theano.config.floatX), )


        layer_out = X
        for layer in self.layers:
            layer_out = layer.output(layer_out)

        self.__predict_func = theano.function([batch_id],
                                             givens =[(X, self.chunk_X[batch_id*batch_size:(batch_id+1)*batch_size,:])],
                                             outputs=layer_out)

        self.__object_expr = self.cost.cost(layer_out, y)
        self.__object_func = theano.function([batch_id],
                                             givens =[(X, self.chunk_X[batch_id*batch_size:(batch_id+1)*batch_size,:]),
                                                 (y, self.chunk_y[batch_id*batch_size:(batch_id+1)*batch_size])],
                                             outputs=self.__object_expr)

        param = self.params()

        grad = T.grad(self.__object_expr, param)

        gradient_vec = []
        for gW, gb in chunks(grad, 2):
            gradient_vec.append(gW.reshape((-1,)))
            gradient_vec.append(gb.reshape((-1,)))

        self.__gradient_expr = theano.tensor.concatenate(gradient_vec)
        self.__gradient_func = theano.function([batch_id],
                                             givens =[(X, self.chunk_X[batch_id*batch_size:(batch_id+1)*batch_size,:]),
                                                 (y, self.chunk_y[batch_id*batch_size:(batch_id+1)*batch_size])],
                                             outputs=self.__gradient_expr)


    def update_chunk(self, X, y):

        self.learning_batch_num = (X.shape[0]+ self.learning_batch_size - 1)/self.learning_batch_size

        self.chunk_X.set_value(X)
        self.chunk_y.set_value(y)

    def get_batch_num(self):

        return self.learning_batch_num


    def params(self):

        return list(itertools.chain.from_iterable([layer.params() for layer in self.layers]))


    def predict(self, X):  # output the click prob

        for layer in self.layers:
            X = layer.output(X)

        return X


    def __getstate__(self):

        state = {"layer_params": [layer.__getstate__() for layer in self.layers],
                 "cost": self.cost.__getstate__()}

        return state

    def __setstate__(self, state):

        self.layers = []
        for idx, param in enumerate(state["layer_params"]):
            layer = create_layer(param)
            self.layers.append(layer)

        self.cost = create_cost(state["cost"])


    def get_parameter(self):

        param_vec = []

        for layer in self.layers:
            W, b = layer.params()

            param_vec.append(W.get_value(borrow=True).reshape((-1,)))
            param_vec.append(b.get_value(borrow=True).reshape((-1,)))

        return numpy.concatenate(param_vec)

    def set_parameter(self, param_vec):

        start_idx = 0

        for layer in self.layers:

            W_size = layer.input_dim()*layer.output_dim()
            W_shape = (layer.input_dim(), layer.output_dim())

            layer.W.set_value(param_vec[start_idx : start_idx + W_size].reshape(W_shape),
                              borrow = True)
            start_idx += W_size

            b_size = layer.output_dim()
            b_shape = (layer.output_dim(),)

            layer.b.set_value(param_vec[start_idx : start_idx + b_size].reshape(b_shape),
                              borrow = True)
            start_idx += b_size


    def predict(self, batch_id):
        return self.__predict_func(batch_id)

    def object(self, batch_id):

        return self.__object_func(batch_id)

    def gradient(self, batch_id):

        return self.__gradient_func(batch_id)



def build_train_function(ctr_model, datasets, batch_size, learning_rate):
    '''Generates a function `train` that implements one step of
    finetuning, a function `validate` that computes the error on
    a batch from the validation set, and a function `test` that
    computes the error on a batch from the testing set

    :type datasets: list of pairs of theano.tensor.TensorType
    :param datasets: It is a list that contain all the datasets;
                     the has to contain three pairs, `train`,
                     `valid`, `test` in this order, where each pair
                     is formed of two Theano variables, one for the
                     datapoints, the other for the labels

    :type batch_size: int
    :param batch_size: size of a minibatch

    :type learning_rate: float
    :param learning_rate: learning rate used during finetune stage
    '''

    X = T.matrix('X')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels


    (train_set_x, train_set_y) = datasets

    index = T.lscalar('index')  # index to a [mini]batch

    # compute the gradients with respect to the model parameters

    cost = ctr_model.cost(X,y)

    params = ctr_model.params()
    gparams = T.grad(cost, params)

    # compute list of fine-tuning updates
    updates = []
    for param, gparam in zip(params, gparams):
        updates.append((param, param - gparam * learning_rate))

    train_fn = theano.function(inputs=[index],
          outputs=cost,
          updates=updates,
          givens={
            X: train_set_x[index * batch_size:
                                (index + 1) * batch_size],
            y: train_set_y[index * batch_size:
                                (index + 1) * batch_size]},
          name='train')
    return train_fn

def build_predict_function(ctr_model, datasets, batch_size):
    (test_set_x, test_set_y) = datasets

    index = T.lscalar('index')
    X = T.matrix('X')  # the data is presented as rasterized images

    sample_num = test_set_x.get_value(borrow=True).shape[0]
    if sample_num % batch_size == 0:
        n_test_batches = sample_num/batch_size
    else:
        n_test_batches = sample_num/batch_size + 1


    predict = ctr_model.predict(X)

    predict = theano.function([index], predict,
        givens={
               X: test_set_x[index * batch_size:
                                  (index + 1) * batch_size],
               },
        name='predict')

    return [predict(i) for i in xrange(n_test_batches)]


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

from sklearn.metrics import roc_auc_score
def test_dnn(dnn, data_set, chunk_size, batch_size):

    validation_prediction = []
    for test_batch_x, test_batch_y in data_set.sample_batches(batch_size=chunk_size):

        test_batch_y = test_batch_y.astype('int32')
        test_batch_y[test_batch_y < 0] = 0

        shared_test_set_x, shared_valid_set_y = shared_dataset((test_batch_x, test_batch_y))

        cur_prediction = build_predict_function(dnn,(shared_test_set_x, shared_valid_set_y), batch_size = batch_size)
        validation_prediction.extend(cur_prediction)

    validation_prediction = T.concatenate(validation_prediction).eval()
    validation_label = data_set.read_columns([valid_dataset.target_column_name]).values

    validation_label[validation_label < 0] = 0

    numpy.savetxt('label.dat', validation_label)
    numpy.savetxt('prediction.dat', validation_prediction)

    assert len(validation_prediction) == len(validation_label)

    return roc_auc_score(validation_label[:4500000], validation_prediction[:4500000]*100)

import cPickle
def train_dnn_ctr_model(
        train_dataset,
        valid_dataset,
        test_dataset,
        hiden_layer_sizes = None,
        finetune_lr=0.001, pretraining_epochs=15,
        pretrain_lr=0.001, training_epochs=1000,
        chunk_size = 10000,
        batch_size=1000):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    if not hiden_layer_sizes:
        hiden_layer_sizes = [1000, 1000, 1000]

    # compute number of minibatches for training, validation and testing

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    dnn = DNNCTRModel( n_ins= train_dataset.get_dimension(),
              hidden_layers_sizes= hiden_layer_sizes,
              n_outs=2)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    #print '... getting the pretraining functions'
    #pretraining_fns = sda.pretraining_functions(train_set_x=shared_train_set_x,
    #                                            batch_size=batch_size)

    # print '... pre-training the model'
    # start_time = time.clock()
    # ## Pre-train layer-wise
    # corruption_levels = [.1, .2, .3]
    # for i in xrange(sda.n_layers):
    #     # go through pretraining epochs
    #     for epoch in xrange(pretraining_epochs):
    #         # go through the training set
    #         c = []
    #         for batch_index in xrange(n_train_batches):
    #             c.append(pretraining_fns[i](index=batch_index,
    #                      corruption=corruption_levels[i],
    #                      lr=pretrain_lr))
    #         print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
    #         print numpy.mean(c)
    #
    # end_time = time.clock()
    #
    # print >> sys.stderr, ('The pretraining code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################


    # get the training, validation and testing function for the model


    best_params = None
    best_validation_auc = 0
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        minichunk_index = 0

        n_train_chunks = train_dataset.get_sample_num()
        n_train_chunks /= chunk_size

        patience = 10 * n_train_chunks  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = 50
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
        for train_batch_x, train_batch_y in train_dataset.sample_batches(batch_size = chunk_size):

            n_train_batches = train_batch_x.shape[0]
            n_train_batches /= batch_size

            train_batch_y = train_batch_y.astype('int32')
            train_batch_y[train_batch_y < 0] = 0
            shared_train_set_x, shared_train_set_y = shared_dataset((train_batch_x, train_batch_y))
            print '... getting the finetuning functions'
            train_fn = build_train_function(dnn,
                (shared_train_set_x, shared_train_set_y),
                batch_size=batch_size,
                learning_rate=finetune_lr)

            print '... finetunning the model'
            # early-stopping parameters

            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_fn(minibatch_index)
                print('epoch %i, chunk %i/%i, minibatch %i/%i, cost %f ' %
                      (epoch, minichunk_index+1, n_train_chunks, minibatch_index + 1, n_train_batches,
                       minibatch_avg_cost ))


            minichunk_index += 1


            model_index = (minichunk_index + 1) / validation_frequency

            model_file_name = "dnn_ctr" + str(model_index) + ".model"

            print >> sys.stderr, "Saving the Model to file " + model_file_name
            model_file = open(model_file_name, "w")

            cPickle.dump(dnn,model_file)

            model_file.close()

            print >> sys.stderr, "Reloading the Model"

            model_file = open(model_file_name,"r")

            dnn = cPickle.load(model_file)

            model_file.close()

            print >> sys.stderr, "Model reloaded"

            iter = (epoch - 1) * n_train_chunks + minichunk_index

            if (iter + 1) % validation_frequency == 0:




                valid_auc = test_dnn(dnn, valid_dataset, chunk_size, batch_size)

                print('epoch %i, mini_chunk %i/%i, validation auc %f %%' %
                      (epoch, minichunk_index+1, n_train_chunks,
                       valid_auc * 100.))

                # if we got the best validation score until now
                if valid_auc > best_validation_auc:

                    #improve patience if loss improvement is good enough
                    if (valid_auc > best_validation_auc *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_auc = valid_auc
                    best_iter = iter

                    # test it on the test set

#                    test_auc = test_dnn(dnn, test_dataset, chunk_size, batch_size)

#                    print('epoch %i, mini_chunk %i/%i, test auc %f %%' %
#                          (epoch, minichunk_index+1, n_train_chunks,
#                           test_auc * 100.))

                if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_auc * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

#from recsys.toolchain.mlbase.data.supervised_dataset import SupervisedDataSet
#from recsys.toolchain.mlbase.model.neuralnetwork.random import init_rng
if __name__ == '__main__':

    train_dataset = SupervisedDataSet(sys.argv[1])
    valid_dataset = SupervisedDataSet(sys.argv[2])
    test_dataset = SupervisedDataSet(sys.argv[3])

    init_rng()


    # for test
    train_dnn_ctr_model(
        train_dataset,
        valid_dataset,
        test_dataset,
        finetune_lr=0.1, pretraining_epochs=15,
        pretrain_lr=0.001, training_epochs=1000,
        hiden_layer_sizes = [1000, 1000, 10000, 1000, 1000],
        chunk_size=500000,
        batch_size=100000)
