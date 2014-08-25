__author__ = 'huang'


#from knowledge.language.core.corpora import Corpora
#from knowledge.language.neural_model.neural_model import NeuralLanguageModel
from knowledge.machine.neuralnetwork.layer.srl_cov_layer import SrlConvLayer

import theano
import theano.tensor as T
import sklearn
import sklearn.cross_validation
import numpy as np
import sys

def test_srl_conv():
    rng = np.random.RandomState(23455)
    hiden_size = 50
    windows_size = 2
    feature_num_list = [20,30]
    cat_size = len(feature_num_list)
    init_W0 = rng.uniform(low=-2.0, high=2.0, size=(hiden_size,1,1,feature_num_list[0]*windows_size))
    '''
    init_W0 = np.zeros((hiden_size,1,1,feature_num_list[0]*windows_size))
    init_W0[0][0][0][:] = np.ones((feature_num_list[0]*windows_size))
    init_W0[0][0][1][:] = 2*np.ones((feature_num_list[0]*windows_size))
    init_W0[0][0][2][:] = 3*np.ones((feature_num_list[0]*windows_size))
    '''
    init_W1 = rng.uniform(low=-2.0, high=2.0, size=(hiden_size,1,1,feature_num_list[1]*windows_size))
    '''
    init_W1 = np.zeros((hiden_size,1,1,feature_num_list[1]*windows_size))
    init_W1[0][0][0][:] = np.ones((feature_num_list[1]*windows_size))
    init_W1[0][0][1][:] = 2*np.ones((feature_num_list[1]*windows_size))
    init_W1[0][0][2][:] = 3*np.ones((feature_num_list[1]*windows_size))
    '''
    #init_b = np.zeros(hiden_size)
    init_b = rng.uniform(low=-2.0, high=2.0, size=(hiden_size))
    '''
    print 'init_W0',init_W0
    print 'init_W1',init_W1
    print 'init_b',init_b
    '''

    rng = np.random.RandomState(1234)
    sample_size = 50000
    data0 = rng.uniform(low=-2.0, high=2.0, size=(sample_size,1,1,windows_size * feature_num_list[0]))
    #print 'data0',data0
    data1 = rng.uniform(low=-2.0, high=2.0, size=(sample_size,1,1,windows_size * feature_num_list[1]))
    #print 'data1',data1
    data = [data0,data1]

    inputs = [T.dtensor4(name='d_%d'%(i)) for i in range(cat_size)]
    conv = SrlConvLayer(inputs,hiden_size,windows_size,feature_num_list,init_W=[init_W0,init_W1],init_b=init_b)
    f = theano.function(inputs=inputs,outputs = conv.linear)
    #ff = theano.function(inputs=inputs,outputs = conv.tmp, on_unused_input='ignore')
    conv.pprint()
    out = f(data0,data1)
    '''
    print 'conv out shape', out.shape
    #out = ff(data0,data1)
    print 'conv',out
    print 'numpy'
    print 'W0 shape',init_W0.shape
    print 'data0 shape',data0.shape
    for i in xrange(sample_size):
        d0 = data0[i][0][:][:]
        d0[0]=d0[0][::-1]
        d1 = data1[i][0][:][:]
        d1[0]=d1[0][::-1]
        out = [np.dot(init_W0[k][0],d0.T) + np.dot(init_W1[k][0],d1.T) for k in xrange(hiden_size)]
        out = [i.reshape(1,)[0] for i in out]
        out += init_b
        print out
    '''


if __name__ == '__main__':
    test_srl_conv()
