#coding=utf8
__author__ = 'huang'

import numpy as np
import cStringIO

from knowledge.language.problem.postags import PosTags
from knowledge.language.problem.srltypes import SrlTypes
from knowledge.language.problem.locdifftypes import LocDiffToVerbTypes, LocDiffToWordTypes, LocTypes
from knowledge.language.core.word.word import Word

from knowledge.language.problem.problem import Problem
from knowledge.language.core.word import word_repo
from knowledge.language.problem import postags, srltypes, locdifftypes

class SRLFeatureBatch(object):
    """
    对每个verb，所有word与该verb的关系特征
    """

    def __init__(self):
        self.sentence_len = 0
        self.verb_num = 0

        #
        self.sentence_word_id = [] #当前句子的全局word id 列表
        self.sentence_pos_id = [] #当前句子的全局词性 id 列表

        #每个<word, verb> pair 一条记录
        self.cur_word_id = []  # 当前word 的词id
        self.cur_verb_id = []  # 当前verb 的词id
        self.cur_word_pos_id = []  # 当前word的词性 id
        self.cur_verb_pos_id = []  # 当前verb的词性 id
        self.cur_word_loc_id = []  # 当前word的位置 id
        self.cur_verb_loc_id = []  # 当前verb的位置 id
        self.cur_word2verb_dist_id = []  # 当前word 到 当前verb的位置距离 id
        self.cur_verb2word_dist_id = []  # 当前verb 到 当前word的位置距离 id
        self.other_word2verb_dist_id = []  # 其他word 到当前verb的位置距离 id
        self.other_word2word_dist_id = []  # 其他word 到当前word的位置距离 id


    def get_sample(self):

        feature_num = 1 + 2 * self.sentence_len + \
            6 + 2 + 2 * self.sentence_len

        feature = np.zeros((self.sentence_len, feature_num))


        for i in range(self.sentence_len):
            start_idx = 0
            feature[i,start_idx] = self.sentence_len
            start_idx += 1
            feature[i, start_idx:start_idx+self.sentence_len] = self.sentence_word_id
            start_idx += self.sentence_len
            feature[i, start_idx:start_idx+self.sentence_len] = self.sentence_pos_id
            start_idx += self.sentence_len
            feature[i,start_idx] = self.cur_word_id[i]
            start_idx += 1
            feature[i,start_idx] = self.cur_verb_id[i]
            start_idx += 1
            feature[i,start_idx] = self.cur_word_pos_id[i]
            start_idx += 1
            feature[i,start_idx] = self.cur_verb_pos_id[i]
            start_idx += 1
            feature[i,start_idx] = self.cur_word_loc_id[i]
            start_idx += 1
            feature[i,start_idx] = self.cur_verb_loc_id[i]
            start_idx += 1
            feature[i,start_idx] = self.cur_word2verb_dist_id[i]
            start_idx += 1
            feature[i,start_idx] = self.cur_verb2word_dist_id[i]
            start_idx += 1
            feature[i, start_idx:start_idx+self.sentence_len] = self.other_word2verb_dist_id[i]
            start_idx += self.sentence_len
            feature[i, start_idx:start_idx+self.sentence_len] = self.other_word2word_dist_id[i]
            start_idx += self.sentence_len

        return feature



import itertools

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
        character['SRL_type_num'] = len(srltypes.SrlTypes.SRLTYPE_LABEL_MAP)
        character['loc_type_num'] = len(locdifftypes.LocTypes.LOC_ID_MAP)
        character['dist_to_verb_num'] = len(locdifftypes.LocDiffToVerbTypes.DIFF_ID_MAP)
        character['dist_to_word_num'] = len(locdifftypes.LocDiffToWordTypes.DIFF_ID_MAP)


        return character

    def sentences(self):

        for sentence in self.__corpora.sentences():
            yield sentence


    def get_dataset_for_sentence(self, sentence):
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


        for srl in sentence.srl_structs():
            X = SRLFeatureBatch()
            y = []

            X.sentence_len =  sentence.word_num()
            X.sentence_word_id = [word.id for word in sentence.words()]
            X.sentence_pos_id = [PosTags.POSTAG_ID_MAP[prop.pos] for prop in sentence.word_properties()]


            label = [ SrlTypes.OTHERTYPE_LABEL]  * sentence.word_num()

            for role in srl.roles():
                if role.length == 1:
                    label[role.start_pos] = SrlTypes.SRLTYPE_LABEL_MAP["S_" + role.type]
                else:
                    label[role.start_pos] = SrlTypes.SRLTYPE_LABEL_MAP["B_" + role.type]
                    for pos in range(role.start_pos+1, role.end_pos ):
                        label[pos] = SrlTypes.SRLTYPE_LABEL_MAP["I_" + role.type]
                    label[role.end_pos] = SrlTypes.SRLTYPE_LABEL_MAP["E_" + role.type]



            verb = srl.verb_infinitive
            verb_loc = srl.verb_loc  #given a verb

            loc_to_verb = [LocDiffToVerbTypes.get_locdiff_id(word_loc - verb_loc)
                    for word_loc in range(sentence.word_num())]


            for word_loc, wd in enumerate(sentence.words()): # for each word

                loc_to_word = [LocDiffToWordTypes.get_locdiff_id(idx - word_loc)
                        for idx in range(sentence.word_num())]

                X.cur_word_id.append(wd.id)
                X.cur_verb_id.append(verb.id)
                X.cur_word_pos_id.append(PosTags.POSTAG_ID_MAP[sentence.get_word_property(word_loc).pos])
                X.cur_verb_pos_id.append(PosTags.POSTAG_ID_MAP[sentence.get_word_property(verb_loc).pos])
                X.cur_word_loc_id.append(LocTypes.get_loc_id(word_loc))
                X.cur_verb_loc_id.append(LocTypes.get_loc_id(verb_loc))
                X.cur_word2verb_dist_id.append(LocDiffToVerbTypes.get_locdiff_id(word_loc - verb_loc ))
                X.cur_verb2word_dist_id.append(LocDiffToWordTypes.get_locdiff_id(verb_loc - word_loc))
                X.other_word2word_dist_id.append(loc_to_word)
                X.other_word2verb_dist_id.append(loc_to_verb)

                y.append(label[word_loc])

            yield X.get_sample(), np.array(y)


    def pretty_srl_predict_label(self, sentence, labels):

        word_column = [ "-" ] * sentence.word_num()
        readable_type_columns = []

        def get_tag(tag_str):
            x = tag_str.split("_",1)
            if len(x) >= 2:
                return x[1]
            else:
                return x[0]


        for idx, srl in enumerate(sentence.srl_structs()):
            word_column[srl.verb_loc] = sentence.get_word(srl.verb_loc).content

            label = labels[idx]
            type_column = [SrlTypes.LABEL_SRLTYPE_MAP[l] for l in label]

            readable_type_column = ["*"] * len(type_column)

            unique_groups = []
            start_idx = 0
            for tag, group in itertools.groupby(type_column, key = lambda x:get_tag(x)):
                group = list(group)
                end_idx = start_idx + len(group)
                if tag == "*":
                    start_idx = end_idx

                else:
                    # for i in range(1, len(group)-1):
                    #     if group[i].startswith("E_") and group[i+1].startswith("B_"):
                    #
                    #         unique_groups.append([start_idx, start_idx + i + 1, tag])
                    #         start_idx += i + 1

                    unique_groups.append([start_idx, end_idx, tag])
                    start_idx = end_idx

            try:
                for start, end, tag in unique_groups:
                    if start == end - 1:
                        readable_type_column[start] = "(" + tag + "*)"
                    else:
                        readable_type_column[start] = "(" + tag + "*"
                        for i in range(start + 1, end - 1):
                            readable_type_column[i] = "*"
                        readable_type_column[end-1] = "*)"
            except:
                print len(type_column)
                print len(readable_type_column)
                print unique_groups
                raise

            readable_type_columns.append(readable_type_column)

        s = cStringIO.StringIO()

        for idx in range(len(word_column)):
            s.write(word_column[idx] + "\t")
            s.write("\t".join([column[idx] for column in readable_type_columns]))
            s.write("\n")

        s.write("\n") # blank line after each sentence

        return s.getvalue()

    def pretty_srl_test_label(self, sentence, labels):
        """
        将句子与预测结果以Conll05可读形式输出出来
        :param sentence:
        :param labels:
        :return:
        """

        word_column = [ "-" ] * sentence.word_num()
        readable_type_columns = []
        for idx, srl in enumerate(sentence.srl_structs()):
            word_column[srl.verb_loc] = sentence.get_word(srl.verb_loc).content

            label = labels[idx]
            type_column = [SrlTypes.LABEL_SRLTYPE_MAP[l] for l in label]

            readable_type_column = ["*"] * len(type_column)

            for idx, type in enumerate(type_column):

                if type.startswith("S_"):
                    readable_type_column[idx] = "(" + type[2:] + "*)"
                elif type.startswith("B_"):
                    readable_type_column[idx] = "(" + type[2:] + "*"
                elif type.startswith("I_"):
                    readable_type_column[idx] = "*"
                elif type.startswith("E_"):
                    readable_type_column[idx] = "*)"
                else:
                    readable_type_column[idx] = "*"


            readable_type_columns.append(readable_type_column)


        s = cStringIO.StringIO()

        for idx in range(len(word_column)):
            s.write(word_column[idx] + "\t")
            s.write("\t".join([column[idx] for column in readable_type_columns]))
            s.write("\n")

        s.write("\n") # blank line after each sentence

        return s.getvalue()



    def get_data_batch(self):


        for sentence in  self.__corpora.sentences():

            for X, y in self.get_dataset_for_sentence(sentence):
                if len(y) == 0:
                    continue

                yield X, y

    def get_trans_mat_prior(self):
        class_num = len(srltypes.SrlTypes.SRLTYPE_LABEL_MAP)
        trans_mat_prior = np.zeros((class_num + 1, class_num))
        for sentence in self.__corpora.sentences():
            label = [ SrlTypes.OTHERTYPE_LABEL ] * sentence.word_num()
            for srl in sentence.srl_structs():
                for role in srl.roles():
                    for pos in range(role.start_pos, role.end_pos + 1):
                        label[pos] = SrlTypes.SRLTYPE_LABEL_MAP[role.type]
                for l in label:
                    trans_mat_prior[0,l] += 1
                for l1, l2 in zip(label, label[1:]):
                    trans_mat_prior[l1,l2] += 1
        mat_sum = np.sum(trans_mat_prior, axis=1).reshape((class_num + 1, 1))
        mat_sum += .1
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
            one_y = [SrlTypes.SRLTYPE_LABEL_MAP[t] for t in one_y]
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





