__author__ = 'sun'


import os
import numpy as np
from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.srl_problem import SRLProblem
from knowledge.language.neural_model.sentence_level_neural_model import SentenceLevelNeuralModel


def test_srl_neural_model():

    home = os.path.expanduser('~')
    train_file_path = os.path.join(home,'Data/conll05/training-set')
    valid_file_path = os.path.join(home,'Data/conll05/dev-set')

    train_corpora = Conll05Corpora()
    train_corpora.load(train_file_path)

    valid_corpora = Conll05Corpora()
    valid_corpora.load(valid_file_path)

    problem_character = train_corpora.get_corpora_character()

    network_build_params = problem_character

    network_build_params['pad_window_size'] = 0

    network_build_params['word_feature_dim'] = 50
    network_build_params['POS_feature_dim'] = 40
    network_build_params['dist_to_verb_feature_dim'] = 60
    network_build_params['dist_to_word_feature_dim'] = 60

    network_build_params['conv_window_size'] = 100
    network_build_params['conv_output_dim'] = 100

    network_build_params['hidden_output_dim'] = 100

    train_problem = SRLProblem(train_corpora)
    valid_problem = SRLProblem(valid_corpora)

    rng = np.random.RandomState(1234)
    network = SentenceLevelNeuralModel(rng,**network_build_params)

    fit_params = dict()
    fit_params['L1_reg'] = 0
    fit_params['L2_reg'] = 0.02
    fit_params["n_epochs"] = 1000
    fit_params["info"] = True

    network.fit(train_problem,valid_problem, ** fit_params)