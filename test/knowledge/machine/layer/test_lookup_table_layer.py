__author__ = 'huang'

from knowledge.machine.neuralnetwork.layer.lookup_table_layer import MultiLookupTableLayer

import theano
import theano.tensor as T
import sklearn
import sklearn.cross_validation
import numpy as np

def test_multilookup():

    inputs_list = [T.imatrix('input_%d' %(i)) for i in xrange(2)]
    multi_lookup = MultiLookupTableLayer(inputs_list,[10,20],[2,3])

    for i in xrange(2):
        print 'lookup_%d'%(i), multi_lookup.embeddings[i].get_value()

    d0 = np.asarray([[0,1,2,3,4],[0,6,7,8,9]],dtype=np.int32)
    d1 = np.asarray([[0,1,2,3,4,5],[10,11,12,13,14,15],[14,15,16,17,18,19]],dtype=np.int32)
    f = theano.function(inputs=[],outputs = multi_lookup.output,givens={inputs_list[0]:d0,inputs_list[1]:d1})
    out = f()
    print 'out\n',out
