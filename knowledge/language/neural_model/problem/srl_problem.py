__author__ = 'huang'


from knowledge.language.core.definition import PosTags
from knowledge.language.core.definition import SrlTypes
from knowledge.language.core.word import Word
import numpy as np

import sys

class SrlProblem(object):


    def __init__(self, corpora, pos, srl_sents):
        '''
        srl_sents format:
        [[vb_idx,sentence_size,[[w1,f1_1,f1_2,...,f1_k,label1],[w2,f2_1,f2_2,...,f2_k,label2],...,[wn,fn_1,fn_2,...,fn_k,labeln]]],
        [...],
        ...
        [...]]
        '''

        self.__corpora = corpora
        self.__pos = pos
        self.__srl_sents = srl_sents

    def get_class_num(self):

        #return len(self.__srl_types)
        pass

    def word_dist(self,sent_sz,idx,pos_conv_size,window_sz,max_term_per_sent):
        '''
        return the position feature
        '''
        assert window_sz % 2 == 1, 'windows_size should be an odd number'
        assert sent_sz <= max_term_per_sent, 'maxium length of sentence should not be greater than %d' %(max_sz - window_sz + 1)
        ret = [0] * (sent_sz+window_sz-1)
        half_win = (window_sz-1) / 2
        tpos = idx + half_win
        sz = len(ret)
        for i in xrange(sz):
            pos = i - tpos
            if pos < -pos_conv_size:
                pos = -pos_conv_size
            elif pos > pos_conv_size:
                pos = pos_conv_size
            pos += pos_conv_size
            # all position is added 1, because 0 is a padding position id
            ret[i] = pos + 1
        if sent_sz < max_term_per_sent:
            ret = ret + [0] * (max_term_per_sent - sent_sz)
        return ret


    def pad_sent_word(self,sentence,window_sz,max_term_per_sent):
        '''
        there exist two types of padding in this version
        pad1_word is expected to have some semantics meanings,
        while pad2 is just padding to make all sentence have the same length
        '''
        sz = len(sentence)
        assert window_sz % 2 == 1, 'windows_size should be an odd number'
        assert sz <= max_term_per_sent, 'maxium length of sentence should not be greater than %d' % (max_term_per_sent)
        pad1_word = [Word.padding_word().id for _i in xrange((window_sz-1)/2)]
        pad1_pos = [Word.padding_pos().id for _i in xrange((window_sz-1)/2)]
        sent_word = [self.__corpora.alloc_global_word_id(word) for word,pos,tag in sentence]
        sent_word = pad1_word + sent_word + pad1_word
        sent_pos = [self.__pos.alloc_global_word_id(pos) for word,pos,tag in sentence]
        sent_pos = pad1_pos + sent_pos + pad1_pos
        if sz < max_term_per_sent:
            sent_word += [Word.padding_word2().id] * (max_term_per_sent - sz)
            sent_pos += [Word.padding_pos2().id] * (max_term_per_sent - sz)
        return sent_word,sent_pos


    def get_batch(self, **kwargs):
        x = []
        y = []
        sent_len = []
        masks = []
        batch_size = kwargs['batch_size']
        window_size = kwargs['window_size']
        pos_conv_size = kwargs['pos_conv_size']
        max_size = kwargs['max_size']
        # max_size is the maxium size of sum of terms and paddings
        max_term_per_sent = max_size - window_size + 1
        padding_sent = [0] * max_size

        cnt = 0

        for sentence in self.__srl_sents:
            # verb position
            vpos = sentence[0]

            sent_word,sent_pos = self.pad_sent_word(sentence[1],window_size,max_term_per_sent)
            sent_vpos = self.word_dist(len(sentence[1]),vpos,pos_conv_size,window_size,max_term_per_sent)
            one_x = []
            one_y = []

            one_x.append(sent_word)
            one_x.append(sent_pos)
            one_x.append(sent_vpos)

            for pos, (word, tag, srl_type) in enumerate(sentence[1]):
                one_x.append(self.word_dist(len(sentence[1]),pos,pos_conv_size,window_size,max_term_per_sent))
                one_y.append(srl_type)

            # padding sentence
            if len(sentence[1]) < max_term_per_sent:
                for _i in xrange(max_term_per_sent - len(sentence[1])):
                    one_x.append(padding_sent)
                    one_y.append('#')

            sent_len.append(len(sentence[1]))
            masks.append([1] * len(sentence[1]) + [0] * (max_term_per_sent - len(sentence[1])))
            # TODO
            one_y = [SrlTypes.SRL_ID_MAP.get(t,-1) for t in one_y]
            y.append(one_y)
            x.append(one_x)
            cnt += max_term_per_sent
            if cnt >= batch_size:
                Y = np.array(y)
                X = np.array(x)
                sent_len = np.array(sent_len)
                masks = np.array(masks)
                yield X, Y, sent_len, masks
                x = []
                y = []
                sent_len = []
                masks = []
                cnt = 0

        Y = np.array(y)
        X = np.array(x)
        sent_len = np.array(sent_len)
        masks = np.array(masks)
        yield X, Y, sent_len, masks





