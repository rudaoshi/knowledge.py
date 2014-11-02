__author__ = 'Huang'

import numpy as np

from knowledge.language.problem.postags import PosTags
from knowledge.language.problem.chunktypes import ChunkTypes
from knowledge.language.problem.nertypes import NERTypes
from knowledge.language.problem.locdifftypes import LocDiffToVerbTypes, LocDiffToWordTypes
from knowledge.language.core.word.word import Word

from knowledge.language.problem.problem import Problem
from knowledge.language.core.word import word_repo
from knowledge.language.problem import postags, srltypes, locdifftypes

class NERProblem(Problem):


    def __init__(self, corpora, windows_size):

        self.__corpora = corpora
        self.windows_size = windows_size

        # parse the corpora and fill the dicts
        self.get_data_batch()


    def get_problem_property(self):

        character = dict()
        character['word_num'] = word_repo.get_word_num()
        character['POS_type_num'] = len(postags.PosTags.POSTAG_ID_MAP)
        character['NER_type_num'] = len(NERTypes.NERTYPE_ID_MAP)
        character['CHUNKING_type_num'] = len(ChunkTypes.CHUNKTYPE_ID_MAP)
        character['SRL_type_num'] = len(srltypes.SrlTypes.SRLTYPE_ID_MAP)
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

        X = [] #SRLFeatureBatch()
        y = []
        for onex, oney in zip(sentence.word_windows(self.windows_size),sentence.nes_lst()):
            word_ids = [w.id for w in onex]
            tag = NERTypes.NERTYPE_ID_MAP[oney]
            X.append(word_ids)
            y.append(tag)

        return X,y

    def get_data_batch(self):

        X = []
        y = []
        for sentence in  self.__corpora.sentences():
            sentX, senty = self.__get_dataset_for_sentence(sentence)
            if len(senty) == 0:
                continue
            X.extend(sentX)
            y.extend(senty)

        # return X,y
        return np.array(X),np.array(y)




