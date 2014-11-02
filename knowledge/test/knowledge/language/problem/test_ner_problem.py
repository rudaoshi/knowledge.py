import os

from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.ner_problem import NERProblem



def test_ner_problem():

    home = os.path.expanduser('~')
    print 'begin'
    #filename = os.path.join(home,'Data/conll05/training-set')
    filename = os.path.join(home,'Data/conll05/dev-set')

    conll05corpora = Conll05Corpora()
    windows_size = 11
    conll05corpora.load(filename)
    print 'load done'

    ne_problem = NERProblem(conll05corpora,windows_size)
    X,y = ne_problem.get_data_batch()
    print X.shape
    print y.shape

if __name__ == "__main__":

    test_ner_problem()
