from knowledge.language.core.corpora import Conll05
from knowledge.language.neural_model.problem.srl_problem import SrlProblem
from knowledge.language.core.corpora import Corpora
from knowledge.language.core.word import Word

def test_SrlProblem():
    print '*' * 20
    print 'test_srl_problem'
    raw_corpora = Conll05.loadraw('/home/kingsfield/data/conll05/training-set')
    srl_sents = []
    for sent in raw_corpora:
        iobsent = Conll05.sentence2iobsentece(sent)
        #print iobsent
        #for i in iobsent:
        #    print i
        #break
        srl_sents += iobsent

    words = Corpora(pading_lst=[Word.padding_word(),Word.padding_word2()])
    pos = Corpora(pading_lst=[Word.padding_pos(),Word.padding_pos2()])
    srl_problem = SrlProblem(words,pos,srl_sents)
    window_size = 11
    pos_conv_size = 15
    max_size = 141 + window_size
    data = srl_problem.get_data_set(pos_conv_size = pos_conv_size, window_size = window_size, max_size = max_size)

    print 'corpora size=%d' % (len(srl_sents))
    print 'test_srl_problem done'
    print '*' * 20

