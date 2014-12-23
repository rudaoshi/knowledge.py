__author__ = 'Sun'


from knowledge.machine.neuralnetwork.layer.path_transition_layer import PathTransitionLayer
from knowledge.machine.neuralnetwork.random import init_rng
import numpy as np
import theano
import itertools
import math

def test_path_transition_layer():

    init_rng()
    sample_num = 5
    class_num = 10

    X = theano.tensor.nnet.softmax(np.random.random((sample_num, class_num)))
    y = (9,9,9,9,9)

    layer = PathTransitionLayer(class_num)

    cost = layer.cost(X,y).eval()
    y_pred = layer.predict(X).eval()

    y_pred_cost = layer.cost(X,y_pred).eval()

    print "optimized cost = ", cost
    print "y_pred = ", y_pred
    print "y_pred_cost", y_pred_cost

    assert y_pred_cost < cost

    y_score = 1000
    logadd_score = 0.0

    trans_mat = theano.tensor.nnet.softmax(layer.tag_trans_matrix).eval()
    X = X.eval()
    for path in itertools.product(range(class_num),repeat=sample_num):
        score = trans_mat[0, path[0]] + X[0, path[0]]
        for idx  in range(1,sample_num):
            score += trans_mat[path[idx-1] + 1, path[idx]] + X[idx, path[idx]]

        score = score#.eval()
        logadd_score += math.exp(score)

        if path == y:
            y_score = score
    logadd_score = math.log(logadd_score)

    bruteforce_cost = logadd_score - y_score

    print "bruteforce cost = {0} with logadd = {1} and selected_path_score = {2}".format(bruteforce_cost, logadd_score, y_score)


    bruteforce_y_pred = np.argmax(X, axis=1) # because trans_mat is const matrix

    print "brueforce y_pred = ", bruteforce_y_pred

    assert math.fabs(bruteforce_cost - cost) < 1e-6

    assert not np.any(y_pred - bruteforce_y_pred)


if __name__ == "__main__":

    test_path_transition_layer()




