__author__ = 'huang'

import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv

class Conv1DLayer(object):
    '''
    Layer that perform 1d convolution
    This implementation is an adapter of conv.2d for 1d input
    Note : A faster version for GPU is avaliable at https://groups.google.com/forum/#!topic/theano-users/JJHZmuUDdPE
    '''

    def __init__(self,name,rng, input_feature_map_num, output_feature_map_num, conv_window , filter_width,
                 init_W=None,init_b=None):

        # Input: a 4D tensor corresponding to a mini-batch of input images with shape:
        #        [mini-batch size, number of input feature maps, image height, image width].
        # Weight: a 4D tensor with shape :
        #        [number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width]

        w_shape = (output_feature_map_num, input_feature_map_num, conv_window , filter_width)
        w_bound = np.sqrt(input_feature_map_num * filter_width)

        if init_W == None:
            self.W = theano.shared(rng.uniform(low=-1.0 / w_bound, high=1.0 / w_bound, size=w_shape).astype(theano.config.floatX),
                               name ='cov_1d_layer_W_%s' %(name))
        else:
            self.W = theano.shared(init_W,name='cov_1d_layer_W_%s' %(name))

        b_shape = (output_feature_map_num,)
        if init_b == None:
            self.b = theano.shared(rng.uniform(low=-.5, high=.5, size=b_shape).astype(theano.config.floatX),
                        name='cov_1d_layer_b_%s' % (name))
        else:
            self.b = theano.shared(init_b,name='cov_1d_layer_b_%s' % (name))



    def output(self, input):
        return conv.conv2d(input,self.W, border_mode='valid') + self.b.dimshuffle('x', 0, 'x', 'x')

    def params(self):
        return [self.W,self.b]

