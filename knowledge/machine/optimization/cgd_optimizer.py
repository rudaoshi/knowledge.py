__author__ = 'Sun'

from scipy.optimize import fmin_cg


from knowledge.machine.optimization.batch_gradient_optimizer import BatchGradientOptimizer


class CGDOptimizer(BatchGradientOptimizer):

    def __init__(self, max_epoches=10, batch_size=10000):

        super(CGDOptimizer, self).__init__(batch_size=batch_size,
                                           max_epoches=max_epoches)


    def optimize_internal(self, object_func, grad_func, param):

        for batch_id in range(self.batch_num):

            print "cost before opt:", object_func(batch_id, param)

            best_param = fmin_cg(
                f = lambda p: object_func(batch_id, p),
                x0=param,
                fprime=lambda p: grad_func(batch_id, p),
                disp=0,
                maxiter=self.max_epoches
            )

            print "cost after opt:", object_func(batch_id, best_param)

            param = best_param

        return param

