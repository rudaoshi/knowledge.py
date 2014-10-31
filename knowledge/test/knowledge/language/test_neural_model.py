__author__ = 'Sun'

import numpy
import sys

from knowledge.language.core.corpora_bak import Corpora
from knowledge.language.neural_model.word_level_neural_model import WordLevelNeuralModel
from knowledge.language.problem.pos_tag_problem import PosTagProblem
import sklearn
import sklearn.cross_validation


def test_neural_language_model():

    corpora = Corpora()
    corpora.load_nltk_conll2000()


    problem = PosTagProblem(corpora)

    X, y = problem.get_data_set(window_size=11)

    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            X, y, test_size=0.2, random_state=0)

    X_train, X_valid, y_train, y_valid = sklearn.cross_validation.train_test_split(
            X_train, y_train, test_size=0.2, random_state=0)

    rng = numpy.random.RandomState(1234)

    print >> sys.stderr, "Problem Size: ", X.shape

    print >> sys.stderr, "word num = ", corpora.get_word_num()

    print >> sys.stderr, "x_train.shape ", X_train.shape
    print >> sys.stderr, X_train



    #print >> sys.stderr, " y_valid shape", y_valid.shape
    #print >> sys.stderr, y_valid
    params = dict()
    params['word_num'] = corpora.get_word_num()
    params['window_size'] = 11
    params['feature_num'] = 50
    params['hidden_layer_size'] = 300
    params['n_outs'] = problem.get_class_num()
    params['L1_reg'] = 0
    params['L2_reg'] = 0.0001


    #model = WordLevelNeuralModel(word_num = corpora.get_word_num(), window_size = 11, feature_num = 100,
    #             hidden_layer_size = 1000, n_outs = problem.get_class_num(), L1_reg = 0.00, L2_reg = 0.0001,
    #             numpy_rng= rng)

    model_name = 'pos'
    load = False
    dump = True
    model_folder = '/home/kingsfield/workspace/knowledge.py'
    init_model_name = None
    model = WordLevelNeuralModel(model_name,load,dump,model_folder,init_model_name,numpy_rng= rng, **params)

    model.fit(X_train,y_train, X_valid, y_valid)

if __name__ == '__main__':
    test_neural_language_model()
