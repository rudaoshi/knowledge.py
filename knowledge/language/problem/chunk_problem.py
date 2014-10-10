__author__ = 'Sun'


from knowledge.language.problem.postags import PosTags
from knowledge.language.problem.postags import ChunkTypes
import numpy as np


class ChunkProblem(object):


    def __init__(self, corpora, iob_sents):

        self.__corpora = corpora
        self.__iob_sents = iob_sents
        self.__chunk_types = set([x[-1] for x in self.__iob_sents])

    def get_class_num(self):

        return len(self.__chunk_types)

    def get_data_set(self, **kwargs):

        x = []
        y = []

        for sentence in self.__iob_sents:

            common_part = []
            for word, tag ,_ in  sentence:

                common_part.extend([self.__corpora.alloc_global_word_id(word),PosTags.POSTAG_ID_MAP[tag]])

            for pos, (word, tag, chunk_type) in enumerate(sentence):

                x.append([pos] + common_part)
                y.append(chunk_type)

        Y = [ChunkTypes.CHUNKTYPE_ID_MAP[type] for type in y]
        X = np.array(x)

        return X, Y





