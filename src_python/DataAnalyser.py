from dataset import *
from OutputData import *
import matplotlib.pyplot as plt

"""
The classes of this module contain methods to analyse instances of the different
data stocking classes of this python project (RHS, solutions, dataset, OutputData).

Each of the classes listed above has its own data analyser. Nevertheless, BoundAnalyser, 
SolutionAnalyser and DatasetAnalyser all take a dataset instance for initialisation.

Note that all data analysers copy the content of the dataset or OutputData instances
given for initialisation to guarantee that their content is not modified.
"""


class BoundAnalyser:
    """
    Attributes
    ---------
    data : dataset instance
    """
    def __init__(self, data):
        assert isinstance(data, dataset), "must be applied on dataset instance"
        self.content = data.get_RHS().copy()

    def max(self):
        """
        Returns an array with the max value for each constraint.

        Returns
        -------
        lmax : np.array
        """
        lmax = []
        (n, p) = self.content.shape
        for j in range(p):
            lmax.append(max(self.content[:, j]))
        return np.array(lmax)

    def min(self):
        """
        Returns an array with the min value for each constraint.

        Returns
        -------
        lmin : np.array
        """
        lmin = []
        (n, p) = self.content.shape
        for j in range(p):
            lmin.append(min(self.content[:, j]))
        return np.array(lmin)

    def range(self):
        """
        Returns an array with the range for each constraint.

        Returns
        -------
        ranges : np.array
        """
        return self.max() - self.min()

    def boxplot_range(self):
        """Creates a boxplot of the range of the different constraints."""
        plt.boxplot(self.range(), whis=[2.5, 97.5])

    def get_constraint_devs(self):
        """
        Returns the biased standard deviation of each constraint of the RHS in an array.

        Returns
        -------
        variations : np.array
            list of standard deviations
        """
        variations = variation(self.content, axis=1)
        return variations

    def get_dev(self):
        """
        Returns teh mean of the biased standard deviations of the different constraints of the RHS.

        Returns
        -------
        dev : float
            mean of the standard deviations
        """
        variations = self.get_constraint_devs()
        return np.mean(variations)


class SolutionAnalyser:
    """
    Attributes
    ---------
    data : dataset instance
    """
    def __init__(self, data):
        assert isinstance(data, dataset), "must be applied on dataset instance"
        self.content = data.get_solutions().copy()

    def mean(self):
        """
        Returns the mean solutions stocked in content.

        Returns
        -------
        mean : float
        """
        return np.mean(self.content)

    def box(self):
        """boxplot of the solutions"""
        plt.boxplot(self.content, whis=[2.5, 97.5])


class DatasetAnalyser:
    """
    Attributes
    ---------
    data : dataset instance
    """
    def __init__(self, data):
        assert isinstance(data, dataset), "must be applied on dataset instance"
        self.bounds = data.get_RHS().copy()
        self.solutions = data.get_solutions().copy()

    def plot2D_sol_fct_of_RHS(self, save=False, path=None, name=None, clear=True):
        """
        Plots the solutions as a function of the constraints in the RHS.
        Asserts that the number of constraints is equal to 1.
        """
        assert len(self.bounds[0]) <= 1, "plots must be 2D"
        if clear:
            plt.clf()
        plt.plot(self.bounds, self.solutions, 'bo')
        plt.legend(["exact solutions"], loc="upper left")

        path = os.path.join("." if path is None else path, "Solution_curves")
        if not os.path.exists(path):
            os.makedirs(path)

        if save is True:
            new_path = os.path.join(path, name)
            plt.savefig(new_path)

        plt.show()


class OutputDataAnalyser:
    """
    Attributes
    ---------
    data : OutputData instance
    """
    def __init__(self, data):
        assert isinstance(data, OutputData), "must be applied on OutputData instance"
        self.solutions = data.get_solutions().copy()
        self.predictions = data.get_predictions().copy()

    def mean_squared_error(self):
        """
        Returns mean squared error of the predictions.

        Returns
        -------
        mse : float
            mean squared error
        """
        p = np.linalg.norm(self.solutions - self.predictions())
        return p*p/len(self.solutions)

    def mean_precision_error(self):
        """
        Returns mean precision error of the predictions.

        Returns
        -------
        mpe : float
            mean precision error
        """
        return np.mean(np.absolute((self.predictions - self.solutions) / self.solutions))

    def rate_over_precision(self, hoped_precision):
        """
        Returns the ratio of predictions where the mean precision error is smaller
        than a given number.

        Returns
        -------
        ratio : float
        """
        number_over_precision = 0
        for i in range(len(self.solutions)):
            if abs((self.predictions[i] - self.solutions[i]) / self.solutions[i]) > hoped_precision:
                number_over_precision += 1
        return number_over_precision / len(self.solutions)


class DataAnalyser:
    """
    The only data analyser that is not fit on a single data structure. Takes a dataset instance
    and the OutputData instance generated by NeuralNetwork while predicting on the given dataset.
    Contains methods to plot graphs to visualise the performance of a neural network.

    Attributes:
    -----------
    input : dataset instance
    output : OutputData instance
    """
    def __init__(self, input, output):
        assert isinstance(input, dataset) and isinstance(output, OutputData), "init takes a dataset and" \
                                                                              "an OutputData instance."
        self.bounds = input.get_RHS().copy()
        self.solutions = input.get_solutions().copy()
        self.predictions = output.get_predictions().copy()
        self.analyser_name = output.get_analyser_name()

    def performance_plot_2D(self, save=False, path=None, name=None, clear=True):
        """
        Plots self.solutions and self.predictions as a function of self.bounds on a
        single graph.

        Asserts that the number of constraints is equal to 1.
        """
        assert len(self.bounds[0]) <= 1, "plots must be 2D"
        if clear:
            plt.clf()
        plt.plot(self.bounds, self.solutions, 'bo')
        plt.plot(self.bounds, self.predictions, 'ro')
        plt.legend(["exact solutions", "predictions"], loc="upper left")

        path = os.path.join("." if path is None else path, "Performance_plots")

        if not os.path.exists(path):
            os.makedirs(path)

        if save is True:
            if name is None:
                new_path = os.path.join(path, self.analyser_name)
                plt.savefig(new_path)
            else:
                new_path = os.path.join(path, name)
                plt.savefig(new_path)

        plt.show()
