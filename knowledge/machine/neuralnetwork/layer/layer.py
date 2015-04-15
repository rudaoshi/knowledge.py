__author__ = 'Sun'

from abc import ABCMeta, abstractmethod

class Layer(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_parameter_size(self):
        pass

    @abstractmethod
    def get_parameter(self):
        pass

    @abstractmethod
    def set_parameter(self, parameter_vec):
        pass

    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def output(self, X, **kwargs):
        pass

    @abstractmethod
    def __getstate__(self):
        pass

    @abstractmethod
    def __setstate__(self, state):
        pass

