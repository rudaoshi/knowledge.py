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
    def object_gradient(self, X, y):
        pass

