import os

from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.srl_problem import SRLProblem



def test_srl_problem(data_file_path):

    conll05corpora = Conll05Corpora()
    conll05corpora.load(data_file_path)
    print 'load done'

    srl_problem = SRLProblem(conll05corpora)

    for sentence in srl_problem.sentences():
        for X, y in srl_problem.get_dataset_for_sentence(sentence):
            word_id = [word.id for word in sentence.words()]
            print "word_id = ", word_id
            print "X = ", X[0]

            break


        for srl in sentence.srl_structs():
            print "verb = ", srl.verb.id

        break

import sys


from knowledge.language.problem.srltypes import SrlTypes
from knowledge.language.evaluation.srl_evaluate import eval_srl
import random
def test_srl_label_formatter(data_file_path):

    conll05corpora = Conll05Corpora()
    conll05corpora.load(data_file_path)
    print 'load done'

    test_label_file_path = "test_label.txt"
    pred_label_file_path = "pred_label.txt"

    srl_problem = SRLProblem(conll05corpora)

    for valid_sentence in srl_problem.sentences():
        test_labels = []
        pred_labels = []

        test_label_file = open(test_label_file_path, "w")
        pred_label_file = open(pred_label_file_path, "w")

        for srl_x, srl_y in srl_problem.get_dataset_for_sentence(valid_sentence):
            pred_y = [random.choice(SrlTypes.LABEL_SRLTYPE_MAP.keys()) for i in range(srl_y.size)]

            test_labels.append(srl_y)
            pred_labels.append(pred_y)

        test_label_str = srl_problem.pretty_srl_test_label(valid_sentence, test_labels)
        pred_label_str = srl_problem.pretty_srl_predict_label(valid_sentence, pred_labels)

        test_label_file.write(test_label_str)
        pred_label_file.write(pred_label_str)

        test_label_file.close()
        pred_label_file.close()

        try:

            valid_result = eval_srl(test_label_file_path, pred_label_file_path)
        except:
            print "label = "
            print "\n".join([" ".join([SrlTypes.LABEL_SRLTYPE_MAP[x] for x in  label]) for label in pred_labels])
            print "formatted_label = "
            print pred_label_str



if __name__ == "__main__":
    data_file_path = sys.argv[1]
#    test_srl_problem(data_file_path)
    test_srl_label_formatter(data_file_path)
