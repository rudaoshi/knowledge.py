__author__ = 'Sun'

from abc import ABCMeta, abstractmethod

class Layer(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def input_dim(self):
        pass

    @abstractmethod
    def output_dim(self):
        pass

    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def output(self, X):
        pass

    @abstractmethod
    def __getstate__(self):
        pass

    @abstractmethod
    def __setstate__(self, state):
        pass

