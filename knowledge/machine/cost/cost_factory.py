__author__ = 'Sun'


__cost_creator = dict()


def register_creator(cost_type, creator):

    __cost_creator[cost_type] = creator


def create_cost(cost_param):

    cost_param = cost_param.copy()

    if "type" not in cost_param:
        raise Exception("Cost type is not provided")

    cost_type = cost_param["type"]

    del cost_param["type"]

    if cost_type not in __cost_creator:
        raise Exception("Unknown layer type")

    return __cost_creator[cost_type](cost_param)

from knowledge.machine.cost.binary_cross_entropy import BinaryCrossEntropyCost
from knowledge.machine.cost.cross_entropy import CrossEntropyCost
from knowledge.machine.cost.mse import MSECost

register_creator("binary_cross_entropy", lambda param: BinaryCrossEntropyCost(**param))
register_creator("cross_entropy", lambda param: CrossEntropyCost(**param))
register_creator("mse", lambda param: MSECost(**param))

