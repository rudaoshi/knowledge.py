__author__ = 'Sun'


"""
.. todo::

    WRITEME
"""
import theano
from theano import tensor
from theano.ifelse import ifelse

def linear_cg(fn, params, tol=1e-3, max_iters=1000, floatX=None):
    """
    Minimizes a POSITIVE DEFINITE quadratic function via linear conjugate
    gradient using the R operator to avoid explicitly representing the Hessian.

    If you have several variables, this is cheaper than Newton's method, which
    would need to invert the Hessian. It is also cheaper than standard linear
    conjugate gradient, which works with an explicit representation of the
    Hessian. It is also cheaper than nonlinear conjugate gradient which does a
    line search by repeatedly evaluating f.

    For more information about linear conjugate gradient, you may look at
    http://en.wikipedia.org/wiki/Conjugate_gradient_method .

    (This reference describes linear CG but not converting it to use
    the R operator instead of an explicit representation of the Hessian)

    Parameters
    ----------
    params : WRITEME
    f : theano_like
        A theano expression which is quadratic with POSITIVE DEFINITE hessian
        in x
    x : list
        List of theano shared variables that influence f
    tol : float
        Minimization halts when the norm of the gradient is smaller than tol

    Returns
    -------
    rval : theano_like
        The solution in form of a symbolic expression (or list of
        symbolic expressions)
    """
    provided_as_list = True
    if not isinstance(params, (list,tuple)):
        params = [params]
        provided_as_list = False

    n_params = len(params)
    def loop(rsold, *args):
        ps = args[:n_params]
        rs = args[n_params:2*n_params]
        xs = args[2*n_params:]

        Aps = []
        for param in params:
            rval = tensor.Rop(tensor.grad(fn,param), params, ps)
            if isinstance(rval, (list, tuple)):
                Aps.append(rval[0])
            else:
                Aps.append(rval)
        alpha = rsold/ sum( (x*y).sum() for x,y in zip(Aps, ps) )
        xs = [ x - alpha*p for x,p in zip(xs,ps)]
        rs = [ r - alpha*Ap for r,Ap in zip(rs,Aps)]
        rsnew = sum( (r*r).sum() for r in rs)
        ps = [ r + rsnew/rsold*p for r,p in zip(rs,ps)]
        return [rsnew]+ps+rs+xs, theano.scan_module.until(rsnew < tol)

    r0s = tensor.grad(fn, params)
    if not isinstance(r0s, (list,tuple)):
        r0s = [r0s]
    p0s = [x for x in r0s]
    x0s = params
    rsold = sum( (r*r).sum() for r in r0s)
    outs, updates = theano.scan( loop,
                                outputs_info = [rsold] + p0s+r0s+x0s,
                                n_steps = max_iters,
                                name = 'linear_conjugate_gradient')
    fxs = outs[1+2*n_params:]
    fxs = [ifelse(rsold <tol, x0, x[-1]) for x0,x in zip(x0s, fxs)]
    if not provided_as_list:
        return fxs[0]
    else:
        return fxs


from scipy.optimize import fmin_cg


from knowledge.machine.optimization.batch_gradient_optimizer import BatchGradientOptimizer


class LinearCGDGradientOptimizer(BatchGradientOptimizer):

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

                print "cost before opt:", train_func(param)

                best_param = fmin_cg(
                    f = train_func,
                    x0=param,
                    fprime=train_grad_func,
                    disp=0,
                    maxiter=self.batch_optim_step
                )

                print "cost after opt:", train_func(best_param)

                param = best_param

        return param

