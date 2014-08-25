__author__ = 'huang'

import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv

class SrlConvLayer(object):
    def __init__(self, inputs, hiden_size,window_size, feature_num_lst,feature_map_size=None,init_W=None,init_b=None):
        rng = np.random.RandomState(1234)
        self.hiden_size = hiden_size
        #self.feature_map_size = feature_map_size
        self.cat_num= len(feature_num_lst)
        self.window_size = window_size
        self.feature_num_lst = feature_num_lst
        self.conv_window  = [i*self.window_size for i in self.feature_num_lst]
        if init_W == None:
            self.W = [theano.shared(np.asarray(rng.uniform(low=-2.0, high=2.0, size=(hiden_size,1,1,self.conv_window[i])), dtype=inputs[0].dtype)
                ,name='srl_cov_layer_W%d' %(i)) for i in range(self.cat_num)]
        else:
            self.W = [theano.shared(init_W[i],name='srl_cov_layer_W%d' %(i)) for i in range(self.cat_num)]

        if init_b == None:
            self.b = theano.shared(np.asarray(rng.uniform(low=-2.0, high=2.0, size=(hiden_size)), dtype=inputs[0].dtype)
                ,name='srl_cov_layer_b')
        else:
            self.b = theano.shared(init_b,name='srl_cov_layer_b')


        self.linear = T.add(*[conv.conv2d(inputs[i],self.W[i]) for i in range(self.cat_num)]) + self.b.dimshuffle('x', 0, 'x', 'x')
        #self.linear = T.add(*[conv.conv2d(inputs[i],self.W[i]) for i in range(self.cat_num)])
        self.linear = self.linear.reshape((inputs[0].shape[0],self.hiden_size))
        self.out = T.nnet.sigmoid(self.linear)

    def pprint(self):
        for i in xrange(self.cat_num):
            print '*' * 10
            print 'W No. %d' % (i)
            print '*' * 10
            print self.W[i].get_value()
        print '*' * 10
        print 'b '
        print '*' * 10
        print self.b.get_value()
        print '*' * 10
        print 'linear '
        print '*' * 10
