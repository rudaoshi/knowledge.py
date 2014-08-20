from knowledge.language.core.corpora import Conll05

if __name__ == '__main__':
    for idx,sentence in enumerate(Conll05.loadraw('/home/kingsfield/data/conll05/training-set')):
        print idx,sentence
        print '\n'
        #break
    print 'done'
