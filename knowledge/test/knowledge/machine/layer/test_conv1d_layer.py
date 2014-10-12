__author__ = 'huang'

from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.machine.neuralnetwork.layer.conv1d_layer import Conv1DLayer


import theano.tensor as T

import numpy as np

def test_lookup():

    table_size =1000
    feature_num = 500

    lookup_table_layer = LookupTableLayer(table_size, feature_num)
    input = T.shared(np.asarray([[0,1,2,3],[1,2,3,4], [7,8,9,10],[5,6,7,8]], dtype=np.int32))

    output_tensor = lookup_table_layer.output(input,tensor_output=True)

    tensor_shape = output_tensor.eval().shape

    assert tensor_shape  == (4, 4, 500), "lookup table output tensor shape = {0}".format(tensor_shape)

    rng = np.random.RandomState(1234)

    conv1d_layer = Conv1DLayer("test", rng, 1, 100, 10)

    conv_output = conv1d_layer.output(output_tensor.dimshuffle(0,'x',1,2))

    conv_out_shape = conv_output.eval().shape

    assert conv_out_shape == (4,100, 4, 491), "conv1d output tensor shape = {0}".format(conv_out_shape)

    batch_size = conv_out_shape[0]
    sentence_len = conv_out_shape[2]
    re_organized = conv_output.dimshuffle(0,2,1,3).reshape(
            (
                batch_size,
                sentence_len,
                -1
            )
        )

    re_organized_shape = re_organized.eval().shape
    assert re_organized_shape == (4,4, 100* 491), "reorganized output shape = {0}".format(re_organized_shape)
