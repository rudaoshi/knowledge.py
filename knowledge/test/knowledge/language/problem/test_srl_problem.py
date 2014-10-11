import os

from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.srl_problem import SRLProblem



def test_srl_problem():

    home = os.path.expanduser('~')
    filename = os.path.join(home,'Data/conll05/training-set')

    conll05corpora = Conll05Corpora()
    conll05corpora.load(filename)

    srl_problem = SRLProblem(conll05corpora)

    for X, y in srl_problem.get_data_batch():
        assert X.shape , "Bad shape {0}".format(X.shape)
        assert X.shape[0] == y.shape[0], "Feature num is not equal to label num."
        assert (X.shape[1] - 6) % 4 == 0, \
            "Feature structure is not right: shape = {0}".format(X.shape)



