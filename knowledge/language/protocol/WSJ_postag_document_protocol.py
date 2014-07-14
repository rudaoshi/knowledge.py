__author__ = 'Sun'

from pyparsing import *
def build_WSJ_postag_parser():

    escape = Literal('\\') + Word(printables,exact = 1)
    pos_tag = OneOrMore(Word(printables, excludeChars = '|/', exact = 1) | escape)
    pos_tags = pos_tag + Optional(Literal("|") + pos_tag)

    word = Word(printables)
    tagged_word = word + "/" + pos_tags

    fragment = '[' + OneOrMore(tagged_word) + "]"

    sentence = OneOrMore(fragment | tagged_word)

    return sentence



class WSJPosTagDocumentProtocol(object):


    def read_document(self, doc_file_path):
        pass
