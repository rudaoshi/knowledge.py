__author__ = 'Sun'

__optim_creator = dict()


def register_creator(optim_type, creator):

    __optim_creator[optim_type] = creator


def create_optimizer(optim_param):

    if "type" not in optim_param:
        raise Exception("Optimizer type is not provided")

    optim_type = optim_param["type"]

    del optim_param["type"]

    if optim_type not in __optim_creator:
        raise Exception("Unknown layer type")

    return __optim_creator[optim_type](optim_param)

from knowledge.machine.optimization.sgd_optimizer import SGDOptimizer
from knowledge.machine.optimization.cgd_optimizer import CGDOptimizer

register_creator("sgd", lambda param: SGDOptimizer(**param))
register_creator("cgd", lambda param: CGDOptimizer(**param))

