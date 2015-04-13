__author__ = 'Sun'

from scipy.optimize import fmin_cg


from knowledge.machine.optimization.batchoptimizer import BatchOptimizer


class CGDOptimizer(BatchOptimizer):

    def __init__(self,
                 max_epoches = 10,
                 batch_size = 10000,
                 batch_optim_step = 3):

        self.max_epoches = max_epoches
        self.batch_size = batch_size
        self.batch_optim_step = batch_optim_step

    def get_batch_size(self):
        return self.batch_size

    def optimize(self, machine, param):

        for i in range(self.max_epoches):

            for batch_id in range(machine.get_batch_num()):

                def train_func(p):

                    machine.set_parameter(p)
                    return machine.object(batch_id)

                def train_grad_func(p):
                    machine.set_parameter(p)
                    return machine.gradient(batch_id)

                best_param = fmin_cg(
                    f = train_func,
                    x0=param,
                    fprime=train_grad_func,
                    disp=0,
                    maxiter=self.batch_optim_step
                )

                print "current cost:", train_func(param)

                param = best_param

        return param

