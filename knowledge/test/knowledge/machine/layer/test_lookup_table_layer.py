__author__ = 'huang'

from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer

import theano.tensor as T

import numpy as np

def test_lookup():

    table_size =1000
    feature_num = 500

    lookup_table_layer = LookupTableLayer(table_size, feature_num)
    input = T.shared(np.asarray([[0,1,2,3],[1,2,3,4], [7,8,9,10],[5,6,7,8]], dtype=np.int32))

    output_flattern = lookup_table_layer.output(input)
    output_tensor = lookup_table_layer.output(input,tensor_output=True)

    flattern_shape = output_flattern.eval().shape
    tensor_shape = output_tensor.eval().shape
    assert flattern_shape == (4, 2000), "flattern shape = {0}".format(flattern_shape)
    assert tensor_shape  == (4, 4, 500), "tensor shape = {0}".format(tensor_shape)
