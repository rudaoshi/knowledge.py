__author__ = 'Sun'


from knowledge.language.core.corpora import Corpora
from knowledge.language.neural_model.neural_model import NeuralLanguageModel
from knowledge.language.neural_model.problem.pos_tag_problem import PosTagProblem

import sklearn
import sklearn.cross_validation
import numpy

def test_neural_language_model():

    corpora = Corpora()
    corpora.load_nltk_conll2000()


    problem = PosTagProblem(corpora)

    X, y = problem.get_window_data_set(window_size=11)

    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            X, y, test_size=0.2, random_state=0)

    X_train, X_valid, y_train, y_valid = sklearn.cross_validation.train_test_split(
            X_train, y_train, test_size=0.2, random_state=0)

    rng = numpy.random.RandomState(1234)

    model = NeuralLanguageModel(word_num = corpora.get_word_num(), window_size = 11, feature_num = 100,
                 hidden_layer_size = 1000, n_outs = problem.get_tag_class_num(), L1_reg = 0.00, L2_reg = 0.0001,
                 numpy_rng= rng)

    model.fit(X_train,y_train, X_valid, y_valid)