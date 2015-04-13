__author__ = 'Sun'

from abc import ABCMeta, abstractmethod


class BatchOptimizer(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def optimize(self, machine, param):
        pass

    @abstractmethod
    def get_batch_size(self):
        pass
