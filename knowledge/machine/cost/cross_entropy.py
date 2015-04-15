__author__ = 'Sun'


from knowledge.machine.cost.cost import Cost

import theano.tensor as T

class CrossEntropyCost(Cost):

    def cost(self, X, y = None):
        """

        :param X: the likely-hood of a sample belong to a class
        :param y: the correct label of the sample
        :return: the cross entropy between the predicted likely-hood and the real label
        """

        assert X.shape == y.shape, "The size of the likely-hood is not equal to that of the label"

        if y.shape[1] >= 2:
            # standard multi-class problem
            return T.nnet.categorical_crossentropy(X, y)
        elif y.shape[1] == 1:
            return T.nnet.binary_crossentropy(X, y)

    def __getstate__(self):
        return {"type": "cross_entropy"}

    def __setstate__(self, state):
        assert state["type"] == "cross_entropy"
        pass