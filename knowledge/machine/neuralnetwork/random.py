__author__ = 'Sun'

import numpy
import numpy.random
import time


random_generator = None


def init_rng(random_seed = None):
    global random_generator
    if random_seed is None:
        random_generator = numpy.random.RandomState()
    else:
        random_generator = numpy.random.RandomState(random_seed)


def get_numpy_rng():

    global random_generator
    assert random_generator is not None, "Random Generator is not initialized."

    return random_generator