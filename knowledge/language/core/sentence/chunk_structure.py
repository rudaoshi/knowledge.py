__author__ = 'sun'


class Chunk(object):
    """
    Class for a chunk in chunk recognition problem
    """
    def __init__(self, type, start_pos, end_pos):

        self.__type = type
        self.__start_pos = start_pos
        self.__end_pos = end_pos

    @property
    def type(self):
        return self.__type

    @property
    def start_pos(self):
        return self.__start_pos

    @property
    def end_pos(self):
        return self.__end_pos

    def pos_shift(self, shift):
        self.__start_pos += shift
        self.__end_pos += shift

    def belongto(self, pos):

        return pos >= self.__start_pos and pos <= self.__end_pos

