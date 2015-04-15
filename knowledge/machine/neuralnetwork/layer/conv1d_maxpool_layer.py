__author__ = 'huang'

import theano
import numpy as np
from operator import mul
import theano.tensor as T
from theano.tensor.nnet import conv
from knowledge.machine.neuralnetwork.random import get_numpy_rng
from knowledge.machine.neuralnetwork.activation.activation_factory import get_activation
from knowledge.machine.neuralnetwork.layer.layer import Layer


class Conv1DMaxPoolLayer(Layer):
    '''
    Layer that perform 1d convolution
    This implementation is an adapter of conv.2d for 1d input
    Note : A faster version for GPU is avaliable at https://groups.google.com/forum/#!topic/theano-users/JJHZmuUDdPE
    '''

    def __init__(self,
                 activator_type="linear",
                 tensor_shape = None,
                 init_W=None,
                 init_b=None, ):

        assert activator_type is not None, "Activation must be provided"
        self.activator_type = activator_type
        self.activator = get_activation(self.activator_type)


        if init_W is not None and tensor_shape is not None:
            assert init_W.shape == tensor_shape, "init tensor size is not equal to the given tensor shape"


        # Input: a 4D tensor corresponding to a mini-batch of input images with shape:
        #        [mini-batch size, number of input feature maps, image height, image width].
        # Weight: a 4D tensor with shape :
        #        [number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width]
        rng = get_numpy_rng()
        if init_W is None and tensor_shape is None:
            raise Exception("neither W now tensor shape is provided.")
        elif init_W is not None:
            self.W = theano.shared(init_W.astype(theano.config.floatX), borrow=True)
            self.tensor_shape = init_W.shape
        elif init_W is None:
            self.tensor_shape = tensor_shape
            (output_feature_map_num, input_feature_map_num, conv_window , filter_width) = tensor_shape
            w_bound = np.sqrt(input_feature_map_num * filter_width)
            init_W = rng.uniform(low=-1.0 / w_bound, high=1.0 / w_bound, size=tensor_shape)

            self.W = theano.shared(init_W.astype(theano.config.floatX),  borrow=True)



        if init_b is None and tensor_shape is None:
            raise Exception("neither b now tensor shape is provided.")
        elif init_b is not None:
            self.b = theano.shared(init_b.astype(theano.config.floatX),  borrow=True)
        elif init_b is None:
            (output_feature_map_num, input_feature_map_num, conv_window , filter_width) = tensor_shape
            b_shape = (output_feature_map_num,)
            init_b = rng.uniform(low=-.5, high=.5, size=b_shape)

            self.b = theano.shared(init_b.astype(theano.config.floatX),  borrow=True)


    def output(self, input, **kwargs):
        conv_output = conv.conv2d(input, self.W, border_mode='valid') + self.b.dimshuffle('x', 0, 'x', 'x')
        activate_output = self.activator(conv_output)
        max_output =  T.max(activate_output, axis=2)
        return max_output

    def params(self):
        return [self.W,self.b]

    def get_parameter_size(self):

        output_feature_map_num = self.tensor_shape[0]
        return reduce(mul, self.tensor_shape,1) + output_feature_map_num

    def get_parameter(self):
        param_vec = []
        param_vec.append(self.W.get_value(borrow=True).reshape((-1,)))
        param_vec.append(self.b.get_value(borrow=True).reshape((-1,)))

        return np.concatenate(param_vec)

    def set_parameter(self, parameter_vec):
        W_size = reduce(mul, self.tensor_shape,1)
        W_shape = self.tensor_shape
        self.W.set_value(parameter_vec[0 :  W_size].reshape(W_shape),
                              borrow = True)

        b_size = self.tensor_shape[0]
        b_shape = (b_size,)
        self.b.set_value(parameter_vec[W_size : W_size + b_size].reshape(b_shape),
                              borrow = True)


    def __getstate__(self):

        state = dict()
        state['type'] = "Conv1DMaxPoolLayer"
        state['W'] = self.W.get_value()
        state['b'] = self.b.get_value()
        state['activator_type'] = self.activator_type

        return state

    def __setstate__(self, state):

        self.W = theano.shared(value=state['W'].astype(theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=state['b'].astype(theano.config.floatX),
                                name='b', borrow=True)

