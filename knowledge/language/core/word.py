__author__ = 'sun'


class Word(object):

    def __init__(self, id, content):
        self.id = id
        self.content = content
        self.pos = None

    @classmethod
    def padding_word(cls):

        padding_word = Word(id = 0, content = "$$$$")
        return padding_word

    @classmethod
    def padding_word2(cls):

        padding_word = Word(id = 1, content = "####")
        return padding_word


