import numpy as np
from dataset import dataset
from OutputData import OutputData
from sklearn.preprocessing import StandardScaler


class BoundProcessor:

    def compute_parameters(self, data):
        pass

    def pre_process(self, data):
        pass


class BoundProcessorAddConst(BoundProcessor):

    def __init__(self, number_of_const=1):
        self.number_of_const = number_of_const

    def compute_parameters(self, data):
        return

    def pre_process(self, data):
        return add_const(data, self.number_of_const)


def add_const(data, number_of_const=1):
    """
    Adds a certain number of ones at the end of each RHS.

    By default, a single 1 will be added at the end of each RHS. This
    is a standard procedure before fitting a set of data to a neural network.

    Arguments
    ---------
    data : dataset instance
        data to be preprocessed
    number_of_const : int (1 by default)
        the number of constants to be added at the end of each RHS
    """
    n = data.size()
    const_matrix = np.ones((n, number_of_const))
    processed_data = dataset(np.hstack((data.get_RHS(), const_matrix)), data.get_solutions())
    return processed_data


class BoundProcessorNormalise(BoundProcessor):

    def __init__(self):
        self.scaler = StandardScaler()

    def compute_parameters(self, data):
        self.scaler.fit(data.get_RHS())

    def pre_process(self, data):
        new_RHS = self.scaler.transform(data.get_RHS())
        data.set_RHS(new_RHS)


def apply_on_bounds(data, f):
    """
    Applies the function f to every element in data.RHS.

    Arguments
    ---------
    data : dataset instance
    f : a real function
    """
    RHS_list = data.get_RHS()
    for i in range(data.size()):
        RHS_list[i] = f(RHS_list[i])
    data.set_RHS(RHS_list)


def apply_linear_on_bounds(data, a, b):
    """
    Applies the linear transformation x -> a * x + b to every element of data.RHS.

    Arguments
    ---------
    data : a dataset instance
    a : float
    b : float
    """
    apply_on_solutions(data, lambda x: a * x + b)




class SolutionProcessor:

    def compute_parameters(self, data):
        pass

    def pre_process(self, data):
        pass

    def post_process(self, data):
        pass


class SolutionProcessorNormalise(SolutionProcessor):
    # The solution array of dimensions (n,1) has to be reshaped to (1,n),
    # since the scaler always scales on the features axis (axis=1).

    def __init__(self):
        self.scaler = StandardScaler()

    def compute_parameters(self, data):
        self.scaler.fit(data.get_solutions().reshape(-1, 1))

    def pre_process(self, data):
        new_solutions = self.scaler.transform(data.get_solutions().reshape(-1, 1))
        data.set_solutions(new_solutions)

    def post_process(self, data):
        new_solutions = self.scaler.inverse_transform(data.get_solutions().reshape(-1, 1))
        new_predictions = self.scaler.inverse_transform(data.get_predictions().reshape(-1, 1))
        data.set_solutions(new_solutions)
        data.set_predictions(new_predictions)


def apply_on_solutions(solutions, f):
    """
    Applies the function f to every element in data.solutions.

    Arguments
    ---------
    solutions : float list
    f : a real function
    """
    for i in range(len(solutions)):
        solutions[i] = f(solutions[i])
    return solutions


def apply_linear_on_solutions(solutions, a, b):
    """
    Applies the linear transformation x -> a * x + b to every element of data.solutions.

    Arguments
    ---------
    solutions : float list
    a : float
    b : float
    """
    return apply_on_solutions(solutions, lambda x: a * x + b)


class SolutionProcessorLinearMean(SolutionProcessor):
    """
    Linear preprocessing that transforms the solutions of a dataset instance so that their
    mean is 0.5 and all values are between 0 and 1.
    """

    def __init__(self):
        self.a = 0
        self.b = 0

    def compute_parameters(self, data):
        mean_value = np.mean(data.get_solutions())
        max_abs = np.max(np.absolute(data.get_solutions() - mean_value))
        self.a = 1 / (2.1 * max_abs)  # Choosing factor > 2 concentrates the values tighter around 0.5
        self.b = - mean_value / (2.1 * max_abs) + 0.5

    def pre_process(self, data):
        solutions = data.get_solutions()
        new_solutions = apply_linear_on_solutions(solutions, self.a, self.b)
        data.set_solutions(new_solutions)

    def post_process(self, data):
        solutions = data.get_solutions()
        predictions = data.get_predictions()
        new_solutions = apply_linear_on_solutions(solutions, 1/self.a, -self.b/self.a)
        new_predictions = apply_linear_on_solutions(predictions, 1/self.a, -self.b/self.a)
        data.set_solutions(new_solutions)
        data.set_predictions(new_predictions)


class SolutionProcessorLinearMax(SolutionProcessor):

    def __init__(self):
        self.max = 0

    def compute_parameters(self, data):
        self.max = 1 / np.max(abs(data.get_solutions()))

    def pre_process(self, data):
        solutions = data.get_solutions()
        new_solutions = apply_linear_on_solutions(solutions, self.max, 0)
        data.set_solutions(new_solutions)

    def post_process(self, data):
        solutions = data.get_solutions()
        predictions = data.get_predictions()
        new_solutions = apply_linear_on_solutions(solutions, 1 / self.max, 0)
        new_predictions = apply_linear_on_solutions(predictions, 1 / self.max, 0)
        data.set_solutions(new_solutions)
        data.set_predictions(new_predictions)
