__author__ = 'huang'

from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer

import theano
import theano.tensor as T
import sklearn
import sklearn.cross_validation
import numpy as np

def test_lookup():
    inputs = T.itensor3('input')
    inputs1 = T.imatrix('inputs1')
    inputs2 = T.imatrix('inputs2')
    lookup1 = LookupTableLayer(inputs1,1000,2,4)
    lookup2 = LookupTableLayer(inputs2,1000,2,4)
    f = theano.function(inputs=[inputs],outputs=[lookup1.output,lookup2.output],givens={inputs1:inputs[0],inputs2:inputs[1]})
    d = np.asarray([[[0,1,2,3],[1,2,3,4]],[[7,8,9,10],[5,6,7,8]]],dtype=np.int32)
    out1,out2 = f(d)
    print 'lookup shape',out1.shape,out2.shape

def xtest_multilookup():

    inputs_list = [T.imatrix('input_%d' %(i)) for i in xrange(2)]
    multi_lookup = MultiLookupTableLayer(inputs_list,[10,20],[2,3])

    for i in xrange(2):
        print 'lookup_%d'%(i), multi_lookup.embeddings[i].get_value()

    d0 = np.asarray([[0,1,2,3,4],[0,6,7,8,9]],dtype=np.int32)
    d1 = np.asarray([[0,1,2,3,4,5],[10,11,12,13,14,15],[14,15,16,17,18,19]],dtype=np.int32)
    f = theano.function(inputs=[],outputs = multi_lookup.output,givens={inputs_list[0]:d0,inputs_list[1]:d1})
    out = f()
    print 'out\n',out
