__author__ = 'Sun'

from knowledge.machine.optimization.optimizer import Optimizer

class SGDOptimizer(Optimizer):

    def __init__(self):
        self.max_epoches = 10
        self.learning_rate = 0.01
        self.decay_rate = 0.9
        self.batch_size = 10000

    def optimize(self, machine, param, X, y = None):

        cur_learning_rate = self.learning_rate


        for i in range(self.max_epoches):

            for batch_id in range(0, X.shape[0], step = self.batch_size):

                end_idx = min(batch_id + self.batch_size, X.shape[0])
                X_batch = X[batch_id: end_idx]
                y_batch = y[batch_id: end_idx] if y is not None else None

                machine.set_parameter(param)
                param = param - cur_learning_rate * machine.gradient(X_batch, y_batch)


            cur_learning_rate *= self.decay_rate

        return param

