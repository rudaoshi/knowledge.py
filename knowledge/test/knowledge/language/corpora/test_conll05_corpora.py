__author__ = 'sun'


import os

from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.srl_problem import SRLProblem



def test_conll05_corpora():

    home = os.path.expanduser('~')
    filename = os.path.join(home,'Data/conll05/training-set')

    conll05corpora = Conll05Corpora()
    conll05corpora.load(filename)


    sentence_num = 0
    for sentence in conll05corpora.sentences():
        sentence_num += 1
        for srl in sentence.srl_structs():
            for role in srl.roles():
                if role.type == "V":
                    assert srl.verb_loc >= role.start_pos and srl.verb_loc <= role.end_pos, \
                        "Bad corpora"

    print "There are {0} sentences in the corpora".format(sentence_num)
