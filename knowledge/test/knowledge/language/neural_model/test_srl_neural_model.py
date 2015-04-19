__author__ = 'sun'

import os

import sys
import numpy as np

from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.srl_problem import SRLProblem
from knowledge.language.neural_model.srl_network import SRLNetowrkArchitecture, NeuralModelHyperParameter, SRLNetwork
from knowledge.machine.neuralnetwork.random import init_rng

from knowledge.machine.optimization.sgd_optimizer import SGDOptimizer
from knowledge.machine.optimization.cgd_optimizer import CGDOptimizer
import numpy
import theano
import time

from knowledge.language.problem.srltypes import SrlTypes


from knowledge.language.evaluation.srl_evaluate import eval_srl


def evaluate(machine, valid_problem, info_suffix):
    test_label_file_path = "test_label_" + str(info_suffix) + ".txt"
    pred_label_file_path = "pred_label_" + str(info_suffix) + ".txt"
    test_raw_label_file_path = "test_raw_label_" + str(info_suffix) + ".txt"
    pred_raw_label_file_path = "pred_raw_label_" + str(info_suffix) + ".txt"

    test_label_file = open(test_label_file_path, "w")
    pred_label_file = open(pred_label_file_path, "w")

    test_raw_label_file = open(test_raw_label_file_path, "w")
    pred_raw_label_file = open(pred_raw_label_file_path, "w")

    test_types = []
    pred_types = []
    for valid_sentence in valid_problem.sentences():
        test_labels = []
        pred_labels = []

        sentence_str = " ".join([word.content for word in valid_sentence.words()])
        for srl_x, srl_y in valid_problem.get_dataset_for_sentence(valid_sentence):
            pred_y = machine.predict(srl_x.astype(theano.config.floatX))

            test_types.append(sentence_str)
            test_types.append("\t".join([SrlTypes.LABEL_SRLTYPE_MAP[l] for l in srl_y]))
            pred_types.append(sentence_str)
            pred_types.append("\t".join([SrlTypes.LABEL_SRLTYPE_MAP[l] for l in pred_y]))

            test_labels.append(srl_y)
            pred_labels.append(pred_y)

        test_label_str = valid_problem.pretty_srl_test_label(valid_sentence, test_labels)
        pred_label_str = valid_problem.pretty_srl_predict_label(valid_sentence, pred_labels)

        test_label_file.write(test_label_str)
        pred_label_file.write(pred_label_str)

        test_raw_label_file.write("\n".join(test_types))
        pred_raw_label_file.write("\n".join(pred_types))

    test_label_file.close()
    pred_label_file.close()

    test_raw_label_file.close()
    pred_raw_label_file.close()

    valid_result = eval_srl(test_label_file_path, pred_label_file_path)
    valid_info = 'validation info {0}% '.format(
                                    valid_result)
    print valid_info


def test_srl_neural_model(train_file_path, valid_file_path):

    train_corpora = Conll05Corpora()
    train_corpora.load(train_file_path)
    train_problem = SRLProblem(train_corpora)

    valid_corpora = Conll05Corpora()
    valid_corpora.load(valid_file_path)
    valid_problem = SRLProblem(valid_corpora)

    init_rng()

    nn_architecture = SRLNetowrkArchitecture()

    nn_architecture.word_feature_dim = 50
    nn_architecture.pos_feature_dim = 50
    nn_architecture.dist_feature_dim = 50

    nn_architecture.conv_window_height = 3
    nn_architecture.conv_output_dim = 500

    nn_architecture.hidden_layer_output_dims = [500,500]


    hyper_param = NeuralModelHyperParameter()

    hyper_param.n_epochs = 1000
    hyper_param.learning_rate = 0.001 #1
    hyper_param.learning_rate_decay_ratio = 0.8
    hyper_param.learning_rate_lowerbound = 0.0000
    hyper_param.l1_reg = 0
    hyper_param.l2_reg = 0

    problem_character = train_problem.get_problem_property()

    m = SRLNetwork(problem_character, nn_architecture)

    optimizer = CGDOptimizer()

    trained_batch_num = 0
    valid_freq = 10

    for iter in range(1000):
        for sentence in train_problem.sentences():

            for X, y in train_problem.get_dataset_for_sentence(sentence):

                if trained_batch_num % valid_freq == 0:
                    evaluate(m, valid_problem, trained_batch_num/valid_freq)


                optimizer.batch_size = X.shape[0]
                optimizer.update_chunk(X, y)

                param = optimizer.optimize(m, m.get_parameter())

                m.set_parameter(param)

                trained_batch_num += 1


if __name__ == "__main__":

    train_file_path = sys.argv[1]
    valid_file_path = sys.argv[2]

    test_srl_neural_model(train_file_path, valid_file_path)
