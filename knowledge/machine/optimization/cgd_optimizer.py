__author__ = 'Sun'

from knowledge.machine.optimization.core.approx_fmin_cg import approx_fmin_cg

from knowledge.machine.optimization.batch_gradient_optimizer import BatchGradientOptimizer

from knowledge.machine.optimization.core.nr_cg import cg_optimize
class CGDOptimizer(BatchGradientOptimizer):

    def __init__(self, max_epoches=10, batch_size=10000,
                 linesearch_iter = 5,
                 ftol=1e-6):

        super(CGDOptimizer, self).__init__(batch_size=batch_size,
                                           max_epoches=max_epoches)

        self.linesearch_iter = linesearch_iter
        self.ftol = ftol


    def optimize(self, param):

        for batch_id in range(self.batch_num):

            print "cost before opt:", self.wrapped_object(batch_id, param)

            # best_param = approx_fmin_cg(
            #     f = lambda p: self.wrapped_object(batch_id, p),
            #     x0=param,
            #     fprime=lambda p: self.wrapped_grad(batch_id, p),
            #     disp=0,
            #     maxiter=self.max_epoches,
            #     xtol=self.xtol
            # )

            best_param = cg_optimize(f=lambda p: self.wrapped_object(batch_id, p),
                                     gf = lambda p: self.wrapped_grad(batch_id, p),
                                     x0 =param,
                                     max_epoches= self.max_epoches,
                                     linesearch_iter= self.linesearch_iter,
                                     ftol= self.ftol)

            print "cost after opt:", self.wrapped_object(batch_id, best_param)

            param = best_param

        return param


