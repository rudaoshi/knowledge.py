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

    def word_dist(self,sent_sz,idx,pos_cov_size,window_sz,max_sz):
        assert window_sz % 2 == 1, 'windows_size should be an odd number'
        ret = [0] * (sent_sz+window_sz-1)
        half_win = (window_sz-1) / 2
        tpos = idx + half_win
        sz = len(ret)
        for i in xrange(sz):
            pos = i - tpos
            if pos < -pos_cov_size:
                pos = -pos_cov_size
            elif pos > pos_cov_size:
                pos = pos_cov_size
            pos += pos_cov_size
            # all pos is add 1, because 0 is a padding position id
            ret[i] = pos + 1
        if sz < max_sz:
            ret = ret + [0] * (max_sz - sz)
        return ret


    def pad_sent_word(sentence,window_sz,max_sz):
        assert window_sz % 2 == 1, 'windows_size should be an odd number'
        sent = sentence
        pad1 = [Word.padding_word() for _i in xrange((window_sz-1)/2)]
        sent = pad1 + sent + pad1
        sz = len(sent)
        assert sz <= max_sz, 'maxium length of sentenc should no be greater than %d' % (d)
        for idx in xrange(0,max_sz - sz):
            sent.append(Word.padding_word2(),'-','-')
        return sent


    def get_data_set(self, **kwargs):
        x = []
        y = []
        window_size = kwargs['window_size']
        pos_cov_size = kwargs['pos_cov_size']
        max_size = kwargs['max_size']

        for sentence in self.__iob_sents:
            # verb position
            vpos = sentence[0]

            common_part = []
            pading_sent = pad_sent_word(sentence[1],window_size,max_size)
            one_x = []
            one_y = []
            for word, tag ,_ in  pading_sent:
                common_part.extend([self.__corpora.alloc_global_word_id(word),PosTags.POSTAG_ID_MAP[tag]])

            one_x.append(common_part)
            one_x.append(self.word_dist(len(sentence[1]),vpos,pos_cov_size,window_size,max_size))

            for pos, (word, tag, srl_type) in enumerate(sentence):
                one_x.append(self.word_dist(len(sentence[1]),pos,pos_cov_size,window_size,max_size))
                one_y.append(srl_type)

            y.append(y)
            x.append(x)
        Y = [ChunkTypes.CHUNKTYPE_ID_MAP[type] for type in y]
        X = np.array(x)

        return X, Y




