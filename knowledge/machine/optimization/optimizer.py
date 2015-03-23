__author__ = 'Sun'

from abc import ABCMeta, abstractmethod


class Optimizer(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def optimize(self, machine, param, X, y = None):
        pass
