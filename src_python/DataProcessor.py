import numpy as np
from dataset import dataset
from OutputData import OutputData
from sklearn.preprocessing import StandardScaler

"""
The classes of this module implement pre- and postprocessing of the training
and evaluation data of a neural network (see class NeuralNetwork).

Note that none of those data processors creates new dataset or OutputData 
instances. They do only modify the content of given dataset or OutputData
instances
"""


class BoundProcessor:
    """
    Bound-processors pre-process the values in the bounds-array stocked in a
    given dataset instance.

    This abstract class has several implementations (BoundProcessorAddConst,
    BoundProcessorNormalise).
    """
    def compute_parameters(self, data):
        """
        Arguments
        ---------
        data : dataset instance
        """
        pass

    def activate(self):
        """
        Activates processor. Only activated processors will be taken into
        account by the training and prediction methods of NeuralNetwork.
        """
        pass

    def is_activated(self) -> bool:
        pass

    def pre_process(self, data):
        """
        Arguments
        ---------
        data : dataset instance
        """
        pass


class BoundProcessorAddConst(BoundProcessor):
    """
    Implementation of BoundProcessor.

    Adds a certain number of constants (1) at the end of each RHS stocked in a
    given dataset instance.

    Attributes
    ----------
    number_of_const : int
        number of constant to be added at the end of each RHS, 1 by default
    """
    def __init__(self, number_of_const=1):
        self.number_of_const = number_of_const
        self.activated = False

    def activate(self):
        self.activated = True

    def is_activated(self):
        return self.activated

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
    data.set_RHS(np.hstack((const_matrix, data.get_RHS())))


class BoundProcessorNormalise(BoundProcessor):
    """
    Implementation of BoundProcessor.

    Computes mean and deviation of the RHS stocked in a given dataset instance constraint
    by constraint and saves their values. Then normalises by subtracting the respective
    mean and dividing by the respective deviation constraint by constraint.
    """

    def __init__(self, mean_list=None, dev_list=None):
        self.scaler = StandardScaler()
        self.activated = False
        self.mean_list = np.array(mean_list)
        self.dev_list = np.array(dev_list)

    def activate(self):
        self.activated = True

    def is_activated(self):
        return self.activated

    def compute_parameters(self, data):
        """Computes mean(i) and dev(i) for all i in data.RHS and saves them."""
        self.scaler.fit(data.get_RHS())

    def pre_process(self, data):
        if self.mean_list is None or len(self.mean_list.shape) == 0:
            new_RHS = self.scaler.transform(data.get_RHS())
            data.set_RHS(new_RHS)
        else:
            rhs = data.get_RHS()
            for i in range(len(rhs)):
                for j in range(len(rhs[0])):
                    rhs[i][j] = (rhs[i][j] - self.mean_list[j])/self.dev_list[j]


def apply_on_bounds(bounds, f, number=None):
    """
    Applies the function f to every element in data.RHS.

    Arguments
    ---------
    bounds : numpy array
    f : a real function
    number : int
        if number is specified, only the first number elements of
        bounds are processed.
    """
    if number is None:
        number = len(bounds)
    for i in range(number):
        for j in range(len(bounds[0])):
            bounds[i][j] = f(bounds[i][j])


def apply_linear_on_bounds(bounds, a, b):
    """
    Applies the linear transformation x -> a * x + b to every element of data.RHS.

    Arguments
    ---------
    bounds : float list or numpy array
    a : float
    b : float
    """
    apply_on_bounds(bounds, lambda x: a * x + b)


def apply_second_deg_on_bounds(bounds, a, b, c):
    """
    Applies the linear transformation x -> a * x^2 + b * x + c to every element of data.RHS.

    Arguments
    ---------
    bounds : float list or numpy array
    a : float
    b : float
    c : float
    """
    apply_on_bounds(bounds, lambda x: a * x * x + b * x + c)


class BoundProcessorSecondDeg(BoundProcessor):
    """
    Processor that applies quadratic function on all bound vectors stocked
    in a given dataset instance.
    """
    def __init__(self, a, b, c, dev=None, avg=None):
        self.a = a
        self.b = b
        self.c = c
        self.dev = dev
        self.avg = avg
        self.activated = False

    def activate(self):
        self.activated = True

    def is_activated(self):
        return self.activated

    def compute_parameters(self, data):
        return

    def pre_process(self, data):
        bounds = data.get_RHS()
        apply_second_deg_on_bounds(bounds, self.a, self.b, self.c)
        apply_linear_on_bounds(bounds, self.dev, self.avg)


class BoundProcessorApplyFunction(BoundProcessor):
    """
    Processor that applies given function on all bound vectors stocked
    in a given dataset instance. Rather used for plots than pre-
    processing before training.

    If attribute number is specified, only the first number elements of
    each bound vector are processed.
    """
    def __init__(self, function, number=None):
        self.function = function
        self.activated = False
        self.number = number

    def activate(self):
        self.activated = True

    def is_activated(self):
        return self.activated

    def compute_parameters(self, data):
        return

    def pre_process(self, data):
        bounds = data.get_RHS()
        apply_on_bounds(bounds, self.function, self.number)


class SolutionProcessor:
    """
    Bound-processors pre-process the values in the solutions-array stocked in a
    given dataset instance and post-processes the values stocked in the solutions-
    and predictions-arrays of the OutputData instance created by a NeuralNetwork
    after prediction on the given dataset instance.

    This abstract class has several implementations (SolutionProcessorNormalise,
    SolutionProcessorLinearMean, SolutionProcessorLinearMax).
    """
    def activate(self) -> bool:
        """
        Activates processor. Only activated processors will be taken into
        account by the training and prediction methods of NeuralNetwork.
        """
        pass

    def is_activated(self):
        pass

    def compute_parameters(self, data):
        """
        Arguments
        ---------
        data : dataset instance
        """
        pass

    def pre_process(self, data):
        """
        Arguments
        ---------
        data : dataset instance
        """
        pass

    def post_process(self, data):
        """
        Arguments
        ---------
        data = OutputData instance
        """
        pass


class SolutionProcessorNormalise(SolutionProcessor):
    """
    Implementation of SolutionProcessor.

    Computes mean m and deviation d of the solutions stocked in a given dataset instance
    and saves their values. Normalises all other datasets by subtracting m and dividing
    by d.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.activated = False

    def activate(self):
        self.activated = True

    def is_activated(self):
        return self.activated

    def compute_parameters(self, data):
        # The solution array of dimensions (n,1) has to be reshaped to (1,n),
        # since the scaler always scales on the features axis (axis=1).
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
    solutions : float list or numpy array
    f : a real function
    """
    for i in range(len(solutions)):
        solutions[i] = f(solutions[i])


def apply_linear_on_solutions(solutions, a, b):
    """
    Applies the linear transformation x -> a * x + b to every element of data.solutions.

    Arguments
    ---------
    solutions : float list or numpy array
    a : float
    b : float
    """
    apply_on_solutions(solutions, lambda x: a * x + b)


class SolutionProcessorLinearMean(SolutionProcessor):
    """
    Linear pre-processing that transforms the solutions of a dataset
    instance such that their mean is 0.5 and all values are between 0 and 1.
    """
    def __init__(self):
        self.a = 0
        self.b = 0
        self.activated = False

    def activate(self):
        self.activated = True

    def is_activated(self):
        return self.activated

    def compute_parameters(self, data):
        mean_value = np.mean(data.get_solutions())
        max_abs = np.max(np.absolute(data.get_solutions() - mean_value))
        self.a = 1 / (2.1 * max_abs)  # Choosing factor > 2 concentrates the values tighter around 0.5
        self.b = - mean_value / (2.1 * max_abs) + 0.5

    def pre_process(self, data):
        apply_linear_on_solutions(data.get_solutions(), self.a, self.b)

    def post_process(self, data):
        apply_linear_on_solutions(data.get_solutions(), 1/self.a, -self.b/self.a)
        apply_linear_on_solutions(data.get_predictions(), 1/self.a, -self.b/self.a)


class SolutionProcessorLinearMax(SolutionProcessor):
    """
    Linear pre-processing that divides all values in the solutions-array
    of a given dataset instance by the max of their absolute values. Thus all values
    are between 0 and 1 after pre-processing.
    """
    def __init__(self):
        self.max = 0
        self.activated = False

    def activate(self):
        self.activated = True

    def is_activated(self):
        return self.activated

    def compute_parameters(self, data):
        self.max = 1 / np.max(np.abs(data.get_solutions()))

    def pre_process(self, data):
        apply_linear_on_solutions(data.get_solutions(), self.max, 0)

    def post_process(self, data):
        apply_linear_on_solutions(data.get_solutions(), 1 / self.max, 0)
        apply_linear_on_solutions(data.get_predictions(), 1 / self.max, 0)
