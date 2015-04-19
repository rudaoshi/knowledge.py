__author__ = 'Sun'

import theano, numpy

class BatchGradientOptimizer(object):

    def __init__(self, batch_size = 5000, max_epoches = 100):

        self.batch_size = batch_size
        self.batch_num = None
        self.max_epoches = 100


        self.chunk_X = theano.shared(numpy.zeros((self.batch_size, 1), dtype = theano.config.floatX))
        self.chunk_y = theano.shared(numpy.zeros((self.batch_size,), dtype = "int32"), )

    def update_chunk(self, X, y):

        self.batch_num = (X.shape[0]+ self.batch_size - 1)/self.batch_size

        self.chunk_X.set_value(X.astype(theano.config.floatX))
        self.chunk_y.set_value(y.astype("int32"))

    def optimize(self, machine, param):

        X = theano.tensor.matrix("X")
        y = theano.tensor.ivector("y")

        batch_id = theano.tensor.iscalar('i')

        batch_X = self.chunk_X[batch_id*self.batch_size:(batch_id+1)*self.batch_size,:]
        batch_y = self.chunk_y[batch_id*self.batch_size:(batch_id+1)*self.batch_size]

        object_, gradient = machine.object_gradient(X, y)

        object_func = theano.function([batch_id],
                     givens =[(X, batch_X),
                                (y, batch_y)],
                     outputs=object_)

        gradient_func = theano.function([batch_id],
                     givens =[(X, batch_X),
                                (y, batch_y)],
                     outputs=gradient)

        def object_func_(i, p):
            machine.set_parameter(p)
            return object_func(i)

        def grad_func_(i, p):
            machine.set_parameter(p)
            return gradient_func(i)

        param = self.optimize_internal(object_func_,
                                       grad_func_,
                                       param)
        return param


    def optimize_internal(self, object_func, grad_func, param):
        return param

    def one_turn_finished(self):
        pass
