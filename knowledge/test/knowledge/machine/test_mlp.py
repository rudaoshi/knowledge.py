__author__ = 'Sun'


from knowledge.machine.neuralnetwork.mlp import MultiLayerPerception
from knowledge.machine.optimization.sgd_optimizer import SGDOptimizer
from knowledge.machine.optimization.cgd_optimizer import CGDOptimizer
import numpy
import theano

def test_sdg_optimizer():


    layer_setting = [{"type": "perception", "activator_type": "sigmoid", "input_dim":50, "output_dim": 100},
                     {"type": "perception", "activator_type": "sigmoid", "input_dim":100, "output_dim": 100},
                     {"type": "perception", "activator_type": "sigmoid", "input_dim":100, "output_dim": 100},
                     {"type": "perception", "activator_type": "sigmoid", "input_dim":100, "output_dim": 1}]

    cost = {"type":"mse"}

    m = MultiLayerPerception(layer_setting, cost)

    optimizer = SGDOptimizer(batch_size=100)

    X = numpy.random.random((1000, 50))
    y = numpy.random.random((1000, 1))

    optimizer.update_chunk(X,y)

    param = optimizer.optimize(m, m.get_parameter())

    m.set_parameter(param)


def test_cdg_optimizer():


    layer_setting = [{"type": "perception", "activator_type": "sigmoid", "input_dim":50, "output_dim": 100},
                     {"type": "perception", "activator_type": "sigmoid", "input_dim":100, "output_dim": 100},
                     {"type": "perception", "activator_type": "sigmoid", "input_dim":100, "output_dim": 100},
                     {"type": "perception", "activator_type": "sigmoid", "input_dim":100, "output_dim": 1}]

    cost = {"type":"mse"}

    m = MultiLayerPerception(layer_setting, cost)

    optimizer = CGDOptimizer(max_epoches=3, batch_size=500)

    X = numpy.random.random((1000, 50))
    y = numpy.random.random((1000, 1))

    optimizer.update_chunk(X,y)

    param = optimizer.optimize(m, m.get_parameter())

    m.set_parameter(param)



if __name__ == "__main__":


    test_sdg_optimizer()
    test_cdg_optimizer()