__author__ = 'huang'

import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv

class SrlConvLayer(object):
    '''
    this layer ouput the linear opt of inputs,
    the nonlinear part is done by the other part of nerual network
    '''

    def __init__(self,name,rng,inputs, hiden_size,input_size,window_size,feature_num,init_W=None,init_b=None):
        self.hiden_size = hiden_size
        self.window_size = window_size
        self.feature_num = feature_num
        self.conv_window  = self.window_size * self.feature_num
        if init_W == None:
            self.W = theano.shared(np.asarray(rng.uniform(low=-2.0, high=2.0, size=(hiden_size,input_size,1,self.conv_window)), dtype=inputs.dtype)
                ,name='srl_cov_layer_W_%s' %(name))
        else:
            self.W = theano.shared(init_W,name='srl_cov_layer_W_%s' %(name))

        if init_b == None:
            self.b = theano.shared(np.asarray(rng.uniform(low=-2.0, high=2.0, size=(hiden_size)), dtype=inputs.dtype)
                ,name='srl_cov_layer_b_%s' % (name))
        else:
            self.b = theano.shared(init_b,name='srl_cov_layer_b_%s' % (name))


        if input_size == 1:
            self.linear = conv.conv2d(inputs,self.W,subsample=(1,self.feature_num)) + self.b.dimshuffle('x', 0, 'x', 'x')
            self.out = self.linear.dimshuffle(0,2,1,3)
        else:
            reshuffle_inputs = inputs.dimshuffle(1,0,'x',2,3)
            self.linear,_updates = theano.scan(lambda x_i: conv.conv2d(x_i,self.w,\
                    subsample=(1,self.feature_num))
                    + self.b.dimshuffle('x', 0, 'x', 'x'), sequences=[reshuffle_inputs])
            self.out = self.linear.dimshuffle(1,0,2,3,4).reshape(inputs.shape[0],inputs.shape[1],1,-1)
