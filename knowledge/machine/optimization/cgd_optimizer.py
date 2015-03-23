__author__ = 'Sun'

from scipy.optimize import fmin_cg


from knowledge.machine.optimization.optimizer import Optimizer


class SGDOptimizer(Optimizer):

    def __init__(self):
        self.max_epoches = 10
        self.batch_size = 10000
        self.batch_optim_step = 3

    def optimize(self, machine, param, X, y = None):

        for i in range(self.max_epoches):

            for batch_id in range(0, X.shape[0], step = self.batch_size):

                end_idx = min(batch_id + self.batch_size, X.shape[0])
                X_batch = X[batch_id: end_idx]
                y_batch = y[batch_id: end_idx] if y is not None else None

                def train_func(p):

                    machine.set_parameter(p)
                    return machine.object(X_batch, y_batch)

                def train_grad_func(p):
                    machine.set_parameter(p)
                    return machine.gradient(X_batch, y_batch)

                best_param = fmin_cg(
                    f = train_func,
                    x0=param,
                    fprime=train_grad_func,
                    disp=0,
                    maxiter=self.batch_optim_step
                )

                param = best_param.xopt

        return param
