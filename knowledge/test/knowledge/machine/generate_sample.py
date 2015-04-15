__author__ = 'Sun'

import numpy
import sys

if __name__ == "__main__":

    output_file_path = sys.argv[1]

    sample_num = int(sys.argv[2])
    sample_dim = int(sys.argv[3])

    X = numpy.random.random((sample_num + 1, sample_dim))
    X[:,0] = X[:,0] > 0.5

    numpy.savetxt(output_file_path, X, delimiter=' ')



