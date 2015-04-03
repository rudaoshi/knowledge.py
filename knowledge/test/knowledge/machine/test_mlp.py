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

    optimizer = SGDOptimizer()
    optimizer.batch_size = 100

    X = numpy.random.random((1000, 50))
    y = numpy.random.random((1000, 1))

    old_cost = m.object(X,y)

    param = optimizer.optimize(m, m.get_parameter(), X, y)

    m.set_parameter(param)
    new_cost = m.object(X, y)

    print old_cost, new_cost

def test_cdg_optimizer():


    layer_setting = [{"type": "perception", "activator_type": "sigmoid", "input_dim":50, "output_dim": 100},
                     {"type": "perception", "activator_type": "sigmoid", "input_dim":100, "output_dim": 100},
                     {"type": "perception", "activator_type": "sigmoid", "input_dim":100, "output_dim": 100},
                     {"type": "perception", "activator_type": "sigmoid", "input_dim":100, "output_dim": 1}]

    cost = {"type":"mse"}

    m = MultiLayerPerception(layer_setting, cost)

    optimizer = CGDOptimizer()
    optimizer.batch_size = 100

    X = numpy.random.random((1000, 50))
    y = numpy.random.random((1000, 1))

    old_cost = m.object(X,y)

    param = optimizer.optimize(m, m.get_parameter(), X, y)

    m.set_parameter(param)
    new_cost = m.object(X, y)

    print old_cost, new_cost



if __name__ == "__main__":


    test_sdg_optimizer()
    test_cdg_optimizer()