__author__ = 'Sun'

import numpy
import sys
import os

from knowledge.language.neural_model.word_level_neural_model import WordLevelNeuralModel
from knowledge.language.problem.chunk_problem import ChunkProblem
from knowledge.language.corpora.conll05 import Conll05Corpora

def test_neural_language_model():

    home = os.path.expanduser('~')
    train_file_path = os.path.join(home,'Data/conll05/training-set.1')
    #train_file_path = os.path.join(home,'Data/conll05/dev-set.1')
    valid_file_path = os.path.join(home,'Data/conll05/dev-set.1')

    train_corpora = Conll05Corpora()
    train_corpora.load(train_file_path,2)

    valid_corpora = Conll05Corpora()
    valid_corpora.load(valid_file_path,2)

    window_size = 11
    train_problem = ChunkProblem(train_corpora,window_size)
    valid_problem = ChunkProblem(valid_corpora,window_size)

    problem_character = train_problem.get_problem_property()


    X_train, y_train = train_problem.get_data_batch()

    X_valid, y_valid = valid_problem.get_data_batch()

    print 'train X shape',X_train.shape
    print 'train y shape',y_train.shape
    print 'valid X shape',X_valid.shape
    print 'valid y shape',y_valid.shape

    rng = numpy.random.RandomState(1234)

    params = dict()
    params['word_num'] = problem_character['word_num']
    params['window_size'] = window_size
    params['feature_num'] = 50
    params['hidden_layer_size'] = 300
    params['n_outs'] = problem_character['CHUNKING_type_num']
    params['L1_reg'] = 0
    params['L2_reg'] = 0.0001

    print params

    #model = WordLevelNeuralModel(word_num = corpora.get_word_num(), window_size = 11, feature_num = 100,
    #             hidden_layer_size = 1000, n_outs = problem.get_class_num(), L1_reg = 0.00, L2_reg = 0.0001,
    #             numpy_rng= rng)

    model_name = 'chunk'
    load = False
    dump = False
    model_folder = '/home/kingsfield/workspace/knowledge.py'
    init_model_name = None
    model = WordLevelNeuralModel(model_name,load,dump,model_folder,init_model_name,rng, **params)

    model.fit(X_train,y_train, X_valid, y_valid)

if __name__ == '__main__':
    test_neural_language_model()
