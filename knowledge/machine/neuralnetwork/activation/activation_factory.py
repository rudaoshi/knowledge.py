__author__ = 'Sun'


__activation = dict()


def register_activation(type, activation_func):

    __activation[type] = activation_func


def get_activation(type):

    if type not in __activation:
        raise Exception("Unknown activation type")

    return __activation[type]

import theano.tensor as T

register_activation("linear", lambda x: x)
register_activation("tanh", T.tanh)
register_activation("sigmoid", T.nnet.sigmoid)


