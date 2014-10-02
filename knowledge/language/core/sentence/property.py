__author__ = 'sunmingming01'


class WordProperty(object):

    def __init__(self):

        self.__pos = None
        self.__ne = None

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, value):
        self.__pos = value

    @property
    def ne(self):
        return self.__ne

    @ne.setter
    def ne(self, value):
        self.__ne = value

