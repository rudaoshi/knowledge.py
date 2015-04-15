__author__ = 'Sun'


__neuralnet_creator = dict()


def register_creator(neuralnet_type, creator):

    __neuralnet_creator[neuralnet_type] = creator


def create_neuralnet(neuralnet_param):

    if "type" not in neuralnet_param:
        raise Exception("Optimizer type is not provided")

    neuralnet_type = neuralnet_param["type"]

    del neuralnet_param["type"]

    if neuralnet_type not in __neuralnet_creator:
        raise Exception("Unknown layer type")

    return __neuralnet_creator[neuralnet_type](neuralnet_param)

from knowledge.machine.neuralnetwork.mlp import MultiLayerPerception

register_creator("mlp", lambda param: MultiLayerPerception(**param))

