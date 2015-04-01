__author__ = 'Sun'


__layer_creator = dict()


def register_creator(layer_type, creator):

    __layer_creator[layer_type] = creator


def create_layer(layer_param):

    if "type" not in layer_param:
        raise Exception("Layer type is not provided")

    layer_type = layer_param["type"]

    del layer_param["type"]

    if layer_type not in __layer_creator:
        raise Exception("Unknown layer type")

    return __layer_creator[layer_type](layer_param)

from knowledge.machine.neuralnetwork.layer.perception import PerceptionLayer

register_creator("perception", lambda param: PerceptionLayer(**param))

