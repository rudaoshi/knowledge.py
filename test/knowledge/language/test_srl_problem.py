from knowledge.language.core.corpora import Conll05
from knowledge.language.neural_model.problem.srl_problem import SrlProblem
from knowledge.language.core.corpora import Corpora
from knowledge.language.core.word import Word
import os

def xtest_SrlProblem():
    print '*' * 20
    print 'test_srl_problem'
    home = os.path.expanduser('~')
    filename = os.path.join(home,'data/conll05/training-set')
    raw_corpora = Conll05.loadraw(filename)
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
    max_term_per_sent = 141
    window_size = 11
    pos_conv_size = 15
    max_size = max_term_per_sent + window_size - 1
    print 'window_size' , window_size
    print 'pos_conv_size', pos_conv_size
    print 'max_term_per_sent', max_term_per_sent
    print 'max_size',max_size
    for idx,data in enumerate(srl_problem.get_batch(batch_size = 1000000,pos_conv_size = pos_conv_size, window_size = window_size, max_size = max_size)):
        print 'data %d' % (idx)
        X,Y,sent_len,masks = data
        print '\tX shape', X.shape
        print '\tY shape', Y.shape
        print '\tsent_len shape', sent_len.shape
        print '\tmasks shape', masks.shape


    print 'corpora size=%d' % (len(srl_sents))
    print 'test_srl_problem done'
    print '*' * 20

