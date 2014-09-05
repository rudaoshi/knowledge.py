__author__ = 'huang'


from knowledge.language.core.definition import PosTags
from knowledge.language.core.definition import ChunkTypes
import numpy as np

import sys

class SrlProblem(object):


    def __init__(self, corpora, srl_sents):
        '''
        srl_sents format:
        [[vb_idx,[[w1,f1_1,f1_2,...,f1_k,label1],[w2,f2_1,f2_2,...,f2_k,label2],...,[wn,fn_1,fn_2,...,fn_k,labeln]]],
        [...],
        ...
        [...]]
        '''

        self.__corpora = corpora
        self.srl_sents = srl_sents
        self.feature_sz = len(self.srl_sents[0][1]) - 1

    def get_class_num(self):

        #return len(self.__srl_types)
        pass

    def word_dist(self,sent_sz,idx,win_sz,pading_sz):
        ret = [0] * sent_sz
        tpos = idx + pading_sz
        for i in xrange(sent_sz+2*pading_sz):
            pos = i - tpos
            if pos < -win_sz:
                pos = -win_sz
            elif pos > win_sz:
                pos = win_sz
            pos += win_sz
            ret[i] = pos
        return ret


    def pad_sent_word(sentence,max_sz = 40):
        sz = len(sentence)
        assert sz <= max_sz, 'maxium length of sentenc should no be greater than %d' % (d)
        sent = sentence
        for idx in xrange(0,max_sz - sz):
            sent.append(Word.padding_word2(),'-','-')
        return sent


    def get_data_set(self, **kwargs):
        x = []
        y = []

        for sentence in self.__iob_sents:
            # verb position
            vpos = sentence[0]

            common_part = []
            pading_sent = pad_sent_word(sentence[1])
            one_x = []
            one_y = []
            for word, tag ,_ in  pading_sent:
                common_part.extend([self.__corpora.alloc_global_word_id(word),PosTags.POSTAG_ID_MAP[tag]])

            one_x.append(common_part)

            for pos, (word, tag, vpos , srl_type) in enumerate(sentence):
                one_x.append(self.word_dist(len(sentence[1]),pos))
                one_y.append(srl_type)

            y.append(y)
            x.append(x)
        Y = [ChunkTypes.CHUNKTYPE_ID_MAP[type] for type in y]
        X = np.array(x)

        return X, Y




