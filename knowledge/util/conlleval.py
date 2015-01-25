__author__ = 'kingsfield'

from knowledge.language.problem import srltypes
from knowledge.language.core.word import word, word_repo
import subprocess
import os


def append_prop_text(fw_gold, fw_pred, X, y, pred):
    assert len(y) == len(pred)
    sentence_len = len(y)
    for i in xrange(sentence_len):
        if y[i] == 39: pred[i] = 39
    def line(X, y):
        for i in xrange(sentence_len):
            if y[i] in [0, 1, 55]:
                label = '*'
            else:
                label = ''
                if i == 0 or y[i] != y[i-1]:
                    label += '(' + srltypes.SrlTypes.ID_SRLTYPE_MAP[y[i]] + '*'
                if i > 0 and y[i] == y[i-1]:
                    label += '*'
                if i >= sentence_len - 1 or y[i] != y[i+1]:
                    label += ')'

            if y[i] == 39 and ((i==0) or (i > 0 and y[i] != y[i-1])):
                word = 'word'
            else:
                word = '-'
            yield word, label

    sep = '                       '
    for word,label in line(X, y):
        fw_gold.write("%s%s%s\n" % (word, sep, label))
    fw_gold.write('\n')
    for word,label in line(X, pred):
        fw_pred.write("%s%s%s\n" % (word, sep, label))
    fw_pred.write('\n')

def conlleval(f1,f2):
    f1.close()
    f2.close()
    goldenfile = f1.name
    predfile = f2.name
    current = os.getcwd()
    # conllperl = os.path.abspath(os.path.join(current, os.pardir))
    conllperl = os.path.join(current,'knowledge')
    conllperl = os.path.join(conllperl,'scripts','srl-eval.pl')
    print subprocess.call(['perl', conllperl ,goldenfile, predfile])

if __name__ == '__main__':
    # testfile = '/home/kingsfield/Data/train.02.props'
    # print subprocess.call(['perl','/home/kingsfield/workspace/knowledge.py/knowledge/scripts/srl-eval.pl',testfile, testfile])
    # f1 = open('/tmp/test1','w')
    # f2 = open('/tmp/test2','w')
    # y = range(20) + [39] + range(20)
    # append_prop_text(f1,f2,None,y,y)
    # # f1 = open('/tmp/test1','r')
    # # f2 = open('/tmp/test2','r')
    # conlleval(f1,f1)
    print len(srltypes.SrlTypes.ID_SRLTYPE_MAP)
