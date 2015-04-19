__author__ = 'Sun'


from knowledge.machine.cost.cost import Cost

import theano.tensor as T

class MSECost(Cost):

    def cost(self, X, y = None):
        """

        :param X: the likely-hood of a sample belong to a class
        :param y: the correct label of the sample
        :return: the cross entropy between the predicted likely-hood and the real label
        """

#        assert X.shape == y.shape, \
#            "The size of the likely-hood is not equal to that of the label " + str(X.shape) + "\t" + str(y.shape)

        return T.pow(X-y, 2).mean()

    def __getstate__(self):
        return {"type": "mse"}

    def __setstate__(self, state):
        assert state["type"] == "mse"
        pass