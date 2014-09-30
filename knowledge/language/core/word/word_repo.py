__author__ = 'sunmingming01'

from .wordid_allocator import PADDING_WORD_STR
from .word import Word

word_repo = dict({PADDING_WORD_STR: Word(PADDING_WORD_STR)})

def get_word(word_str):

    if word_str not in word_repo:
        word_repo[word_str] = Word(word_str)

    return word_repo[word_str]

def get_padding_word():

    return word_repo[PADDING_WORD_STR]
