__author__ = 'Sun'


from abc import ABCMeta, abstractmethod

class Cost(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def cost(self, X, y = None):
        pass

    @abstractmethod
    def __getstate__(self):
        pass

    @abstractmethod
    def __setstate__(self, state):
        pass

