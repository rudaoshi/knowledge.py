__author__ = 'Sun'


__optimizer_creator = dict()


def register_creator(optimizer_type, creator):

    __optimizer_creator[optimizer_type] = creator


def create_optimizer(optimizer_param):

    if "type" not in optimizer_param:
        raise Exception("Optimizer type is not provided")

    optimizer_type = optimizer_param["type"]

    del optimizer_param["type"]

    if optimizer_type not in __optimizer_creator:
        raise Exception("Unknown layer type")

    return __optimizer_creator[optimizer_type](optimizer_param)

from knowledge.machine.optimization.sgd_optimizer import SGDOptimizer
from knowledge.machine.optimization.cgd_optimizer import CGDOptimizer

register_creator("sgd", lambda param: SGDOptimizer(**param))
register_creator("cgd", lambda param: CGDOptimizer(**param))

