__author__ = 'Sun'

import numpy
import sys
import os

from knowledge.language.neural_model.word_level_neural_model import WordLevelNeuralModel
from knowledge.language.problem.ner_problem import NERProblem
from knowledge.language.corpora.conll05 import Conll05Corpora

def test_neural_language_model():

    home = os.path.expanduser('~')
    #train_file_path = os.path.join(home,'Data/conll05/training-set')
    train_file_path = os.path.join(home,'Data/conll05/dev-set')
    valid_file_path = os.path.join(home,'Data/conll05/dev-set')

    train_corpora = Conll05Corpora()
    train_corpora.load(train_file_path)

    valid_corpora = Conll05Corpora()
    valid_corpora.load(valid_file_path)

    window_size = 11
    train_problem = NERProblem(train_corpora,window_size)
    valid_problem = NERProblem(valid_corpora,window_size)

    problem_character = train_problem.get_problem_property()


    X_train, y_train = train_problem.get_data_batch()

    X_valid, y_valid = valid_problem.get_data_batch()

    rng = numpy.random.RandomState(1234)

    params = dict()
    params['word_num'] = problem_character['word_num']
    params['window_size'] = window_size
    params['feature_num'] = 50
    params['hidden_layer_size'] = 300
    params['n_outs'] = problem_character['NE_type_num']
    params['L1_reg'] = 0
    params['L2_reg'] = 0.0001


    #model = WordLevelNeuralModel(word_num = corpora.get_word_num(), window_size = 11, feature_num = 100,
    #             hidden_layer_size = 1000, n_outs = problem.get_class_num(), L1_reg = 0.00, L2_reg = 0.0001,
    #             numpy_rng= rng)

    model_name = 'ner'
    load = False
    dump = False
    model_folder = '/home/kingsfield/workspace/knowledge.py'
    init_model_name = None
    model = WordLevelNeuralModel(model_name,load,dump,model_folder,init_model_name,numpy_rng= rng, **params)

    model.fit(X_train,y_train, X_valid, y_valid)

if __name__ == '__main__':
    test_neural_language_model()
