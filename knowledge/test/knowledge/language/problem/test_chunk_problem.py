import os

from knowledge.language.corpora.conll05 import Conll05Corpora
from knowledge.language.problem.chunk_problem import ChunkProblem



def test_chunk_problem():

    home = os.path.expanduser('~')
    print 'begin'
    #filename = os.path.join(home,'Data/conll05/training-set')
    filename = os.path.join(home,'Data/conll05/dev-set.1')

    conll05corpora = Conll05Corpora()
    windows_size = 11
    conll05corpora.load(filename,2)
    print 'load done'

    chunk_problem = ChunkProblem(conll05corpora,windows_size)
    X,y = chunk_problem.get_data_batch()
    print X.shape
    print y.shape

if __name__ == "__main__":

    test_chunk_problem()
