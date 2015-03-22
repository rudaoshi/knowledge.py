__author__ = 'Sun'

from abc import ABCMeta, abstractmethod

class GradientOptimizable(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_parameter(self):
        pass

    @abstractmethod
    def set_parameter(self, param):
        pass

    @abstractmethod
    def object(self, X, y=None):
        pass

    @abstractmethod
    def gradient(self, X, y=None):
        pass
