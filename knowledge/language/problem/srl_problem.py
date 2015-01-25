__author__ = 'huang'

import numpy as np

from knowledge.language.problem.postags import PosTags
from knowledge.language.problem.srltypes import SrlTypes
from knowledge.language.problem.locdifftypes import LocDiffToVerbTypes, LocDiffToWordTypes, LocTypes
from knowledge.language.core.word.word import Word

from knowledge.language.problem.problem import Problem
from knowledge.language.core.word import word_repo
from knowledge.language.problem import postags, srltypes, locdifftypes

class SRLFeatureBatch(object):

    def __init__(self):
        self.word_id = []
        self.pos_id = []
        self.other_feature = []

    def finsh_batch(self):

        self.word_id = np.array(self.word_id)
        self.pos_id = np.array(self.pos_id)
        self.other_feature = np.array(self.other_feature)


class SRLProblem(Problem):


    def __init__(self, corpora):

        self.__corpora = corpora

        # parse the corpora and fill the dicts
        for X,y in self.get_data_batch():
            pass


    def get_problem_property(self):

        character = dict()
        character['word_num'] = word_repo.get_word_num()
        character['POS_type_num'] = len(postags.PosTags.POSTAG_ID_MAP)
        character['SRL_type_num'] = len(srltypes.SrlTypes.SRLTYPE_ID_MAP)
        character['loc_type_num'] = len(locdifftypes.LocTypes.LOC_ID_MAP)
        character['dist_to_verb_num'] = len(locdifftypes.LocDiffToVerbTypes.DIFF_ID_MAP)
        character['dist_to_word_num'] = len(locdifftypes.LocDiffToWordTypes.DIFF_ID_MAP)


        return character

    def __get_dataset_for_sentence(self, sentence):
        """
        Extract features for sentences. The extracted features are as follows:
        * word id: used in a lookup layer to find word embeddings
        * word pos: pos tags of the word. used as a int feature, or in a look up table for tag embeddings
        * distance to verb: distance from a word to a given verb.
                            used as int feature or in a look up table for tag embeddings
        :param sentence:
        :return:
        """

#        sentence.pad_sentece(window_size)


        '''
        word_id_vec = [word.id for word in sentence.words()]
        pos_id_vec = [PosTags.POSTAG_ID_MAP[word_prop.pos]
                      for word_prop in sentence.word_properties()
                     ]
        '''




        sentence_len = sentence.word_num()
        for srl in sentence.srl_structs():
            X = [] #SRLFeatureBatch()
            y = []

            verb = srl.verb_infinitive
            verb_loc = srl.verb_loc  #given a verb 

            loc_to_verb = [LocDiffToVerbTypes.get_locdiff_id(word_loc - verb_loc)
                    for word_loc in range(sentence.word_num())]

            label = [ SrlTypes.SRLTYPE_ID_MAP[SrlTypes.PADDING_SRL_TYPE] ] * sentence.word_num()

            for role in srl.roles():
                for pos in range(role.start_pos, role.end_pos + 1):
                    label[pos] = SrlTypes.SRLTYPE_ID_MAP[role.type]

            for word_loc, wd in enumerate(sentence.words()): # for each word

                loc_to_word = [LocDiffToWordTypes.get_locdiff_id(idx - word_loc)
                        for idx in range(sentence.word_num())]


                X.append(
                         [sentence_len, wd.id, verb.id,
                          PosTags.POSTAG_ID_MAP[sentence.get_word_property(verb_loc).pos],
                          PosTags.POSTAG_ID_MAP[sentence.get_word_property(word_loc).pos],
                          LocTypes.get_loc_id(word_loc),
                          LocTypes.get_loc_id(verb_loc),
                          LocDiffToWordTypes.get_locdiff_id(verb_loc - word_loc),
                          LocDiffToVerbTypes.get_locdiff_id(word_loc - verb_loc )
                         ] +  loc_to_word + loc_to_verb
                         )


                y.append(label[word_loc])

            yield np.array(X), np.array(y)

    def get_data_batch(self):


        for sentence in  self.__corpora.sentences():

            for X, y in self.__get_dataset_for_sentence(sentence):
                if len(y) == 0:
                    continue

                yield X, y

    def get_trans_mat_prior(self):
        class_num = len(srltypes.SrlTypes.SRLTYPE_ID_MAP)
        trans_mat_prior = np.zeros((class_num + 1, class_num))
        for sentence in self.__corpora.sentences():
            label = [ SrlTypes.SRLTYPE_ID_MAP[SrlTypes.PADDING_SRL_TYPE] ] * sentence.word_num()
            for srl in sentence.srl_structs():
                for role in srl.roles():
                    for pos in range(role.start_pos, role.end_pos + 1):
                        label[pos] = SrlTypes.SRLTYPE_ID_MAP[role.type]
                for l in label:
                    trans_mat_prior[0,l] += 1
                for l1, l2 in zip(label, label[1:]):
                    trans_mat_prior[l1,l2] += 1
        mat_sum = np.sum(trans_mat_prior, axis=1).reshape((class_num + 1))
        mat_sum = np.repeat(mat_sum, class_num, axis=1)
        trans_mat_prior = trans_mat_prior / mat_sum
        return trans_mat_prior













class SrlProblem2(object):


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

    def word_dist(self,sent_sz,idx,window_size,max_term_per_sent):
        '''
        return the position feature
        '''
        assert window_size % 2 == 1, 'windows_size should be an odd number'
        assert sent_sz <= max_term_per_sent, 'maxium length of sentence should not be greater than %d' %(max_term_per_sent)
        ret = [0] * (sent_sz+window_size-1)
        position_conv_half_window = (window_size-1) / 2
        tpos = idx + position_conv_half_window
        sz = len(ret)
        for i in xrange(sz):
            pos = i - tpos
            if pos < -position_conv_half_window:
                pos = -position_conv_half_window
            elif pos > position_conv_half_window:
                pos = position_conv_half_window
            pos += position_conv_half_window
            # all position is added 1, because 0 is a padding position id
            ret[i] = pos + 1
        if sent_sz < max_term_per_sent:
            ret = ret + [0] * (max_term_per_sent - sent_sz)
        return ret


    def pad_sent_word(self,sentence,window_size,max_term_per_sent):
        '''
        there exist two types of padding in this version
        pad1_word is expected to have some semantics meanings,
        while pad2 is just padding to make all sentence have the same length
        '''
        sz = len(sentence)
        assert window_size % 2 == 1, 'windows_size should be an odd number'
        assert sz <= max_term_per_sent, 'maxium length of sentence should not be greater than %d' % (max_term_per_sent)
        # padding for word vector
        pad1_word = [Word.padding_word().id for _i in xrange((window_size-1)/2)]
        # padding for position vector
        pad1_pos = [Word.padding_pos().id for _i in xrange((window_size-1)/2)]

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
        #position_conv_half_window = kwargs['position_conv_half_window']
        max_size = kwargs['max_size']
        # max_size is the maximum number of sum of terms and paddings
        max_term_per_sent = max_size - window_size + 1
        padding_sent = [0] * max_size

        cnt = 0

        for sentence in self.__srl_sents:
            # verb position
            vpos = sentence[0]

            sent_word,sent_pos = self.pad_sent_word(sentence[1],window_size,max_term_per_sent)
            #sent_vpos = self.word_dist(len(sentence[1]),vpos,position_conv_half_window,window_size,max_term_per_sent)
            sent_vpos = self.word_dist(len(sentence[1]),vpos,window_size,max_term_per_sent)
            one_x = []
            one_y = []

            one_x.append(sent_word)
            one_x.append(sent_pos)
            one_x.append(sent_vpos)

            for pos, (word, tag, srl_type) in enumerate(sentence[1]):
                one_x.append(self.word_dist(len(sentence[1]),pos,window_size,max_term_per_sent))
                one_y.append(srl_type)

            # padding sentence
            if len(sentence[1]) < max_term_per_sent:
                for _i in xrange(max_term_per_sent - len(sentence[1])):
                    one_x.append(padding_sent)
                    one_y.append('#')

            sent_len.append(len(sentence[1]))
            masks.append([1] * len(sentence[1]) + [0] * (max_term_per_sent - len(sentence[1])))
            one_y = [SrlTypes.SRLTYPE_ID_MAP[t] for t in one_y]
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





