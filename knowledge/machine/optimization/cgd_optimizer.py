__author__ = 'Sun'

from knowledge.machine.optimization.core.approx_fmin_cg import approx_fmin_cg

from knowledge.machine.optimization.batch_gradient_optimizer import BatchGradientOptimizer


class CGDOptimizer(BatchGradientOptimizer):

    def __init__(self, max_epoches=10, batch_size=10000, xtol=1e-6):

        super(CGDOptimizer, self).__init__(batch_size=batch_size,
                                           max_epoches=max_epoches)

        self.xtol = xtol


    def optimize(self, param):

        for batch_id in range(self.batch_num):

            print "cost before opt:", self.wrapped_object(batch_id, param)

            best_param = approx_fmin_cg(
                f = lambda p: self.wrapped_object(batch_id, p),
                x0=param,
                fprime=lambda p: self.wrapped_grad(batch_id, p),
                disp=0,
                maxiter=self.max_epoches,
                xtol=self.xtol
            )

            print "cost after opt:", self.wrapped_object(batch_id, best_param)

            param = best_param

        return param


