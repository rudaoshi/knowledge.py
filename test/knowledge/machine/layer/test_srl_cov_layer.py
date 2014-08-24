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
    hiden_size = 3
    windows_size = 2
    feature_num_list = [2,3]
    cat_size = len(feature_num_list)
    init_W0 = np.zeros((1,1,hiden_size,feature_num_list[0]*windows_size))
    init_W0[0][0][0][:] = np.ones((feature_num_list[0]*windows_size))
    init_W0[0][0][1][:] = 2*np.ones((feature_num_list[0]*windows_size))
    init_W0[0][0][2][:] = 3*np.ones((feature_num_list[0]*windows_size))
    init_W1 = np.zeros((1,1,hiden_size,feature_num_list[1]*windows_size))
    init_W1[0][0][0][:] = np.ones((feature_num_list[1]*windows_size))
    init_W1[0][0][1][:] = 2*np.ones((feature_num_list[1]*windows_size))
    init_W1[0][0][2][:] = 3*np.ones((feature_num_list[1]*windows_size))
    init_b = np.zeros(2)
    print 'init_W0',init_W0
    print 'init_W1',init_W1
    print 'init_b',init_b

    rng = np.random.RandomState(1234)
    sample_size = 5
    data0 = rng.uniform(low=-2.0, high=2.0, size=(1,1,sample_size,windows_size * feature_num_list[0]))
    print 'data0',data0
    data1 = rng.uniform(low=-2.0, high=2.0, size=(1,1,sample_size,windows_size * feature_num_list[1]))
    print 'data1',data1
    data = [data0,data1]

    inputs = [T.dtensor4(name='d_%d'%(i)) for i in range(cat_size)]
    #conv = SrlConvLayer([data0,data1],hiden_size,windows_size,feature_num_list,init_W=[init_W0,init_W1],init_b=init_b)
    conv = SrlConvLayer(inputs,hiden_size,windows_size,feature_num_list,init_W=[init_W0,init_W1],init_b=init_b)
    #f = theano.function(inputs=inputs,outputs = conv.linear)
    ff = theano.function(inputs=inputs,outputs = conv.tmp, on_unused_input='ignore')
    conv.pprint()
    #out = f(data0,data1)
    #print 'linear',out
    out = ff(data0,data1)
    print 'linear',out
    print 'numpy'
    print init_W0.shape
    print data0.shape
    out = np.dot(init_W0[0][0],data0[0][0].T)
    print out
    print np.sum(out,axis=1)


if __name__ == '__main__':
    test_srl_conv()
