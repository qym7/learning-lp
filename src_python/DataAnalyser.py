from dataset import *
from OutputData import *


class BoundAnalyser:

    def __init__(self, data):
        assert isinstance(data, dataset), "must be applied on dataset instance"
        self.content = data.get_RHS()

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

    def __init__(self, data):
        assert isinstance(data, dataset), "must be applied on dataset instance"
        self.content = data.get_solutions()

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

    def __init__(self, data):
        assert isinstance(data, dataset), "must be applied on dataset instance"
        self.content = data

    def plot2D_sol_fct_of_RHS(self):
        """
        Plots the solutions as a function of the constraints in the RHS.
        Asserts that the number of constraints is equal to 1.
        """
        x = self.content.get_RHS()
        y = self.content.get_solutions()
        assert len(x[0]) <= 1, "plots must be 2D"
        plt.scatter(x, y)
        plt.show()


class OutputDataAnalyser:

    def __init__(self, data):
        assert isinstance(data, OutputData), "must be applied on OutputData instance"
        self.content = data

    def mean_squared_error(self):
        p = np.linalg.norm(self.content.get_solutions() - self.content.get_predictions())
        return p*p/self.content.size()

    def mean_precision_error(self):
        solutions = self.content.get_solutions()
        predictions = self.content.get_predictions()
        return np.mean(np.absolute((predictions - solutions) / solutions))

    def rate_over_precision(self, hoped_precision):
        solutions = self.content.get_solutions()
        predictions = self.content.get_predictions()

        number_over_precision = 0
        for i in range(len(solutions)):
            if abs((predictions[i] - solutions[i]) / solutions[i]) > hoped_precision:
                number_over_precision += 1
        return number_over_precision / self.content.size()

