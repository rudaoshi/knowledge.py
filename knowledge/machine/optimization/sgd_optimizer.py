__author__ = 'Sun'

from knowledge.machine.optimization.batchoptimizer import BatchOptimizer
import theano

class SGDOptimizer(BatchOptimizer):

    def __init__(self,
                 max_epoches = 10,
                 learning_rate = 0.01,
                 decay_rate = 0.9,
                 batch_size = 10000):

        self.max_epoches = max_epoches
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def optimize(self, machine, param):

        cur_learning_rate = self.learning_rate

        for i in range(self.max_epoches):

            for batch_id in range(machine.get_batch_num()):

                machine.set_parameter(param)
                print "cost before opt:", machine.object(batch_id)
                gradient = machine.gradient(batch_id)

                param = param - cur_learning_rate * gradient

                machine.set_parameter(param)
                print "cost after opt:", machine.object(batch_id)

            cur_learning_rate *= self.decay_rate

        return param


