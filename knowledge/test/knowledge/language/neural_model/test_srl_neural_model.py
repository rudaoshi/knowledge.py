__author__ = 'sun'


import os
import theano
import numpy as np
from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.srl_problem import SRLProblem
from knowledge.language.neural_model.sentence_level_neural_model import SentenceLevelNeuralModel


def test_srl_neural_model():

    home = os.path.expanduser('~')
    train_file_path = os.path.join(home,'Data/conll05/training-set')
    #train_file_path = os.path.join(home,'Data/conll05/dev-set')
    valid_file_path = os.path.join(home,'Data/conll05/dev-set')

    train_corpora = Conll05Corpora()
    train_corpora.load(train_file_path)

    valid_corpora = Conll05Corpora()
    valid_corpora.load(valid_file_path)

    train_problem = SRLProblem(train_corpora)
    valid_problem = SRLProblem(valid_corpora)

    problem_character = train_problem.get_problem_property()

    network_build_params = problem_character

    network_build_params['pad_window_size'] = 0

    network_build_params['word_feature_dim'] = 50
    network_build_params['POS_feature_dim'] = 50
    network_build_params['dist_to_verb_feature_dim'] = 60
    network_build_params['dist_to_word_feature_dim'] = 60

    network_build_params['conv_window_size'] = 10
    network_build_params['conv_output_dim'] = 30
    network_build_params['hidden_output_dim_1'] = 100
    network_build_params['hidden_output_dim_2'] = 100



    rng = np.random.RandomState(1234)

    print network_build_params

    network = SentenceLevelNeuralModel(rng,**network_build_params)

    fit_params = dict()
    fit_params['L1_reg'] = 0
    fit_params['L2_reg'] = 0.00001
    fit_params["n_epochs"] = 1000
    fit_params["info"] = True
    fit_params["learning_rate"] = 0.01
    fit_params["min_learning_rate"] = 0.0001
    fit_params["learning_rate_decay_ratio"] = 0.8

    network.fit(train_problem,valid_problem, ** fit_params)

if __name__ == "__main__":

    test_srl_neural_model()
