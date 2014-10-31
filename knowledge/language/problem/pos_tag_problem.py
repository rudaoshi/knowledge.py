__author__ = 'Sun'


from knowledge.language.problem.postags import PosTags
import numpy as np


class PosTagProblem(object):


    def __init__(self, corpora):

        self.__corpora = corpora

    def get_class_num(self):

        return len(PosTags.POSTAG_ID_MAP)

    def get_data_set(self, window_size, **kwargs):

        x = []
        y = []

        for doc in self.__corpora.documents:

            for sentence in doc.sentences:

                for word_window in sentence.word_windows(window_size):

                    x.append([word.id for word in word_window])
                    y.append(word_window[window_size/2].tag)

        Y = [PosTags.POSTAG_ID_MAP[tag] for tag in y]
        X = np.array(x)

        return X, Y





