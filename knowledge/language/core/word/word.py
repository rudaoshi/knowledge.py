__author__ = 'sun'

from .wordid_allocator import alloc_word_id, PADDING_WORD_STR

class Word(object):

    def __init__(self, content):
        self.id = alloc_word_id(content)

        self.content = content

