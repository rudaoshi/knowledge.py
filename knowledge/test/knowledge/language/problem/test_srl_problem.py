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

if __name__ == "__main__":
    data_file_path = sys.argv[1]
    test_srl_problem(data_file_path)
