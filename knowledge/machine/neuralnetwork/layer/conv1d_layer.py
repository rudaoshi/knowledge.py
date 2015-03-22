__author__ = 'huang'

import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv
from knowledge.machine.neuralnetwork.random import get_numpy_rng

class Conv1DLayer(object):
    '''
    Layer that perform 1d convolution
    This implementation is an adapter of conv.2d for 1d input
    Note : A faster version for GPU is avaliable at https://groups.google.com/forum/#!topic/theano-users/JJHZmuUDdPE
    '''

    def __init__(self, name, init_W=None,init_b=None, tensor_shape = None):

        assert isinstance(name, str) and len(name) > 0
        self.name = name
        if init_W is not None and tensor_shape is not None:
            assert init_W.shape == tensor_shape, "init tensor size is not equal to the given tensor shape"

        # Input: a 4D tensor corresponding to a mini-batch of input images with shape:
        #        [mini-batch size, number of input feature maps, image height, image width].
        # Weight: a 4D tensor with shape :
        #        [number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width]
        rng = get_numpy_rng()
        if init_W is None and tensor_shape is None:
            self.W = None
        elif init_W is not None:
            self.W = theano.shared(init_W.astype(theano.config.floatX), name="conv1d_w_%s" % (self.name), borrow=True)
        elif init_W is None:
            (output_feature_map_num, input_feature_map_num, conv_window , filter_width) = tensor_shape
            w_bound = np.sqrt(input_feature_map_num * filter_width)
            init_W = rng.uniform(low=-1.0 / w_bound, high=1.0 / w_bound, size=tensor_shape)

            self.W = theano.shared(init_W.astype(theano.config.floatX), name="conv1d_w_%s" % (self.name), borrow=True)



        if init_b is None and tensor_shape is None:
            self.b = None
        elif init_b is not None:
            self.b = theano.shared(init_b.astype(theano.config.floatX), name="conv1d_b_%s" % (self.name), borrow=True)
        elif init_b is None:
            (output_feature_map_num, input_feature_map_num, conv_window , filter_width) = tensor_shape
            b_shape = (output_feature_map_num,)
            init_b = rng.uniform(low=-.5, high=.5, size=b_shape)

            self.b = theano.shared(init_b.astype(theano.config.floatX), name="conv1d_b_%s" % (self.name), borrow=True)


    def output(self, input):
        return conv.conv2d(input, self.W, border_mode='valid') + self.b.dimshuffle('x', 0, 'x', 'x')

    def params(self):
        return [self.W,self.b]


    def __getstate__(self):

        state = dict()
        state['name'] = "conv1d"
        state['W'] = self.W.get_value()
        state['b'] = self.b.get_value()

        return state

    def __setstate__(self, state):

        self.W = theano.shared(value=state['W'].astype(theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=state['b'].astype(theano.config.floatX),
                                name='b', borrow=True)

