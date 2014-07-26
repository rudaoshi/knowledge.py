__author__ = 'sun'

from knowledge.language.core.word import Word
from knowledge.util.data_process import moving_window

class Sentence(object):

    def __init__(self, id, content = ""):
        self.id = id
        self.content = content
        self.words = []


    def segement_words(self, word_segmenter):

        self.words = word_segmenter.segment(self.content)

    def add_word(self, word_obj):
        self.words.append(word_obj)


    def word_windows(self, windows_size):

        assert (windows_size+1)%2 == 0, "window size should be an odd number"

        padding_num = (windows_size - 1)/2

        paddings = [Word.padding_word() for i in range(padding_num)]

        extended_words = paddings + self.words + paddings

        return moving_window(extended_words, windows_size)

