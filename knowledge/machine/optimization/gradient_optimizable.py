__author__ = 'Sun'

from abc import ABCMeta, abstractmethod

class BatchStocasticGradientOptimizable(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_parameter(self):
        pass

    @abstractmethod
    def set_parameter(self, param):
        pass

    @abstractmethod
    def object(self, batch_id):
        pass

    @abstractmethod
    def gradient(self, batch_id):
        pass
