__author__ = 'Sun'


from knowledge.machine.neuralnetwork.layer.path_transition_layer import PathTransitionLayer
from knowledge.machine.neuralnetwork.random import init_rng
import numpy as np
import theano
import theano.tensor as T
import itertools
import math
from collections import Counter

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

def test_path_transition_layer2():

    init_rng()
    sample_num = 5
    class_num = 10

    y1 = (9,9,9,9,9)
    X1 = np.zeros((sample_num, class_num))
    X1[range(sample_num),y1] = 1

    y2 = (0,1,2,3,4)
    X2 = np.zeros((sample_num, class_num))
    X2[range(sample_num),y2] = 1

    layer = PathTransitionLayer(class_num)
    cost1 = layer.cost(X1,y1).eval()
    cost2 = layer.cost(X2,y2).eval()

    cost1_2 = layer.cost(X1,y2).eval()
    cost2_1 = layer.cost(X2,y1).eval()

    y_pred1 = layer.predict(X1).eval()
    y_pred2 = layer.predict(X2).eval()

    print "X1 = ", X1
    print "X2 = ", X2

    print "y1 = ", y1
    print "y2 = ", y2

    print "cost1 = ", cost1
    print "cost2 = ", cost2
    print "cost1_2 = ", cost1_2
    print "cost2_1 = ", cost2_1
    print "y_pred1 = ", y_pred1
    print "y_pred2 = ", y_pred2

def test_path_layer_train():
    init_rng()
    class_num = 3
    trans_mat = np.zeros((class_num,class_num))
    trans_mat[0,0] = 0.8
    trans_mat[0,1] = 0.1
    trans_mat[0,2] = 0.1
    trans_mat[1,0] = 0.1
    trans_mat[1,1] = 0.1
    trans_mat[1,2] = 0.8
    trans_mat[2,0] = 0.1
    trans_mat[2,1] = 0.1
    trans_mat[2,2] = 0.8
    def gen_train(trans_mat,sample_num,class_num):
        X = []
        Y = []
        x = [1.0/class_num for i in xrange(class_num)]
        pre_y = 0
        for _i in xrange(sample_num):
            X.append(x)
            trans_prob = trans_mat[pre_y]
            y = np.random.choice(class_num,1,p=trans_prob)[0]
            Y.append(y)
            pre_y = y
        return np.asarray(X,dtype=np.float32),np.asarray(Y,dtype=np.int32)
    X,Y = gen_train(trans_mat,100,class_num)
    print Counter(Y)
    trans_mat_prior = np.zeros((class_num + 1,class_num))
    trans_mat_prior[0] = X[0]
    trans_mat_prior[1:] = trans_mat
    print 'init trains mat'
    print trans_mat_prior
    #layer = PathTransitionLayer(class_num,trans_mat_prior)
    layer = PathTransitionLayer(class_num)
    train_x = T.fmatrix("x")
    train_y = T.ivector("y")
    cost = layer.cost(train_x,train_y)

    params = layer.params()
    gparams = T.grad(cost,params)
    updates = []
    learning_rate = 0.01
    for param, gparam in zip(params, gparams):
        updates.append((param, param - learning_rate * gparam))

    iternum = 10
    for i in xrange(iternum):
        X,Y = gen_train(trans_mat,100,class_num)
        foo = theano.function(inputs=[train_x,train_y],outputs=[cost],updates=updates)
        print '%d,cost=%f' %(i, foo(X,Y)[0])
        trained_trans_mat = layer.params()[0].eval()
        print trained_trans_mat
    print 'training done.'




if __name__ == "__main__":

    test_path_transition_layer2()
    test_path_layer_train()




