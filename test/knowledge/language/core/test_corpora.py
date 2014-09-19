from knowledge.language.core.corpora import Conll05

def test_Cornll05():
    print 'test cornll 05'
    raw_corpora = Conll05.loadraw('/home/kingsfield/data/conll05/training-set')
    for sent in raw_corpora:
        print sent
        Conll05.sentence2iobsentece(sent)
    print 'test cornll 05 done'

