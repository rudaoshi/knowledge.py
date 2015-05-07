__author__ = 'Sun'

from knowledge.machine.optimization.batch_gradient_optimizer import BatchGradientOptimizer
import theano

class SGDOptimizer(BatchGradientOptimizer):

    def __init__(self,
                 learning_rate = 0.01,
                 decay_rate = 0.9,
                 batch_size = 10000):
        super(SGDOptimizer, self).__init__(batch_size=batch_size,
                                           max_epoches=1)

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        self.cur_learning_rate = self.learning_rate


    def optimize(self, param):

        for batch_id in range(self.batch_num):

#            print "cost before opt:", object_func(batch_id, param)

#            new_param = param - self.cur_learning_rate * self.wrapped_grad(batch_id, param)
             self.wrapped_train(batch_id)
#            print "cost after opt:", object_func(batch_id, new_param)

#            param = new_param

        return param

    def get_update(self, param, object_, gradient_):

        update = [param[i] - self.cur_learning_rate* gradient_[i] for i in range(len(param))]
        return update


    def one_turn_finished(self):
        self.cur_learning_rate = self.learning_rate * self.decay_rate

