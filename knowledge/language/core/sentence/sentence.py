__author__ = 'sun'

from knowledge.language.core.sentence.property import WordProperty
from knowledge.language.core.word.word_repo import get_padding_word
from knowledge.util.data_process import moving_window
from knowledge.language.problem.postags import PosTags


class Sentence(object):

    def __init__(self):
        self.__words = []
        self.__word_properties = []
        self.__srl_structs = []

        self.__chunks = []
        self.__phrases = []
        self.__nes = []
#        self.__phrase = None
#        self.__syntree = None



    def add_word(self, word_obj, word_property = None):
        self.__words.append(word_obj)

        if word_property:
            self.__word_properties.append(word_property)

    def words(self):

        for word in self.__words:
            yield word

    def word_num(self):
        return len(self.__words)

    def get_word(self, pos):
        return self.__words[pos]

    def word_properties(self):

        for prop in self.__word_properties:
            yield prop

    def get_word_property(self, pos):
        return self.__word_properties[pos]

    def add_srl_struct(self, srl):

        srl.owner = self
        self.__srl_structs.append(srl)

    def srl_num(self):
        return len(self.__srl_structs)

    def srl_structs(self):

        for verb in self.__srl_structs:
            yield verb

    def add_chunk(self, chunk):

        self.__chunks.append(chunk)

    def chunk_num(self):
        return len(self.__chunks)

    def chunks(self):
        for chunk in self.__chunks:
            yield chunk

    def add_phrase(self, chunk):

        self.__phrases.append(chunk)

    def phrase_num(self):
        return len(self.__phrases)

    def phrases(self):
        for chunk in self.__phrases:
            yield chunk

    def add_ne(self, chunk):
        self.__nes.append(chunk)

    def ne_num(self):
        return len(self.__nes)

    def nes(self):
        for chunk in self.__nes:
            yield chunk


    def pad_sentece(self, windows_size):
        """
        Padding the sentence with nonsence word so that the moving window method can be applied
        :param windows_size:
        :return:
        """

        assert (windows_size+1)%2 == 0, "window size should be an odd number"

        padding_num = (windows_size - 1)/2

        paddings = [get_padding_word()] * padding_num
        properties = [WordProperty()] * padding_num
        for property in properties:
            property.pos = PosTags.PADDING_POS_TAG

        self.__words = paddings + self.__words + paddings
        self.__word_properties = properties + self.__word_properties + properties
        for srl in self.__srl_structs:
            srl.pos_shift(padding_num)

        for ne in self.__nes:
            ne.pos_shift(padding_num)

        for chunk in self.__chunks:
            chunk.pos_shift(padding_num)

        for phrase in self.__phrases:
            phrase.pos_shift(padding_num)




    def word_windows(self, windows_size):

        assert (windows_size+1)%2 == 0, "window size should be an odd number"

        padding_num = (windows_size - 1)/2

        paddings = [get_padding_word()] * padding_num



        extended_words = paddings + self.__words + paddings

        return moving_window(extended_words, windows_size)

