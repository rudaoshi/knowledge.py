import os
import theano
import numpy as np
from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.srl_problem import SRLProblem
from knowledge.language.neural_model.sentence_level_neural_model import SentenceLevelNeuralModel

import theano.tensor as T
import theano
from knowledge.language.neural_model.sentence_level_log_likelihood_layer import SentenceLevelLogLikelihoodLayer
from knowledge.machine.neuralnetwork.layer.mlp import HiddenLayer
from knowledge.machine.neuralnetwork.layer.conv1d_layer import Conv1DLayer
from knowledge.machine.neuralnetwork.layer.lookup_table_layer import LookupTableLayer
from knowledge.machine.neuralnetwork.layer.logistic_sgd import LogisticRegression

from knowledge.language.problem.locdifftypes import LocDiffToWordTypes
from theano.tensor.signal import downsample

def test_pos_neural_model():
    pass

if __name__ == "__main__":
    test_pos_neural_model()

