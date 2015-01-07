__author__ = 'sun'

import os

import sys
import numpy as np

from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.srl_problem import SRLProblem
from knowledge.language.neural_model.srl_neural_model import SRLNetowrkArchitecture, NeuralModelHyperParameter, train_srl_neural_model
from knowledge.machine.neuralnetwork.random import init_rng

def test_srl_neural_model(DATA_FOLDER):



    train_file_path = os.path.join(DATA_FOLDER,'conll05/training-set')
    valid_file_path = os.path.join(DATA_FOLDER,'conll05/dev-set')

    train_corpora = Conll05Corpora()
    train_corpora.load(train_file_path)

    valid_corpora = Conll05Corpora()
    valid_corpora.load(valid_file_path)

    train_problem = SRLProblem(train_corpora)
    valid_problem = SRLProblem(valid_corpora)

    init_rng()

    nn_architecture =  SRLNetowrkArchitecture()

    nn_architecture.word_feature_dim = 50
    nn_architecture.pos_feature_dim = 50
    nn_architecture.dist_feature_dim = 50

    nn_architecture.conv_window_height = 3
    nn_architecture.conv_output_dim = 5

    nn_architecture.hidden_layer_output_dims = [100,100]


    hyper_param = NeuralModelHyperParameter()

    hyper_param.n_epochs = 1000
    hyper_param.learning_rate = 0.001 #1
    hyper_param.learning_rate_decay_ratio = 0.8
    hyper_param.learning_rate_lowerbound = 0.0000
    hyper_param.l1_reg = 0
    hyper_param.l2_reg = 0

    train_srl_neural_model(train_problem,valid_problem, nn_architecture,hyper_param)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        DATA_FOLDER = sys.arv[1]
    elif sys.platform[:3] == 'win':
        DATA_FOLDER = "D:\\Experiment\\Projects\\nn_language\\data"  # os.path.expanduser('~')
    else:
        DATA_FOLDER = os.path.join(os.path.expanduser('~'),'Data')

    test_srl_neural_model(DATA_FOLDER)
