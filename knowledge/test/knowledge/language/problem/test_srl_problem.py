import os

from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.srl_problem import SRLProblem



def test_srl_problem():

    home = os.path.expanduser('~')
    filename = os.path.join(home,'Data/conll05/training-set')

    conll05corpora = Conll05Corpora()
    conll05corpora.load(filename)

    srl_problem = SRLProblem(conll05corpora, 5)

    for X, y in srl_problem.get_data_batch():
        assert X.shape , "Bad shape {0}".format(X.shape)
        assert X.shape[0] == y.shape[0], "Feature num is not equal to label num."
        assert (X.shape[1] - 4)/2 == X[0][0], \
            "Feature structure is not right: feature dim = {0}, sentence len = {1}".format(X.shape, X[0][0])



