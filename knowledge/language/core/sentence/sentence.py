__author__ = 'sun'

from collections import OrderedDict

from knowledge.language.core.word.word import Word
from knowledge.util.data_process import moving_window

class Sentence(object):

    def __init__(self):
        self.__words = []
        self.__word_properties = []
        self.__verbs = OrderedDict()

        self.__trunk = None
        self.__phrase = None
        self.__syntree = None



    def add_word(self, word_obj, word_property = None):
        self.__words.append(word_obj)

        if word_property:
            self.__word_properties.append(word_property)

    def words(self):

        for word in self.__words:
            yield word

    def get_word(self, pos):
        return self.__words[pos]

    def word_properties(self):

        for prop in self.__word_properties:
            yield prop

    def get_word_property(self, pos):
        return self.__word_properties[pos]

    def add_verb(self, pos, verb):

        self.__verbs[pos] = verb
        verb.owner = self


    def verbs(self):

        for verb in self.__verbs:
            yield verb


    def get_verb(self, pos):
        return self.__verbs[pos]






    def word_windows(self, windows_size):

        assert (windows_size+1)%2 == 0, "window size should be an odd number"

        padding_num = (windows_size - 1)/2

        paddings = [Word.padding_word() for i in range(padding_num)]

        extended_words = paddings + self.words + paddings

        return moving_window(extended_words, windows_size)

