import numpy as np
import matplotlib.pyplot as plt
from dataset import solutions


class OutputData:
    def __init__(self, sol_list, predict_list, network=None):
        self.solutions = solutions(sol_list)
        self.predictions = solutions(predict_list)
        s1, s2 = self.solutions.size(), self.predictions.size()
        assert s1 == s2, "Solutions and predictions do not have the same size"
        self.used_network = network

    def add_used_network(self, neural_network):
        self.used_network = neural_network

    def get_solutions(self):
        return self.solutions.get_solutions()

    def get_predictions(self):
        return self.predictions.get_solutions()

    def set_solutions(self, new_solutions):
        assert len(new_solutions) == self.size()
        self.solutions.set_solutions(new_solutions)

    def set_predictions(self, new_predictions):
        assert len(new_predictions) == self.size()
        self.predictions.set_solutions(new_predictions)

    def size(self):
        """
        Returns number of samples in dataset instance.

        Returns
        -------
        size : int
            number of samples
        """
        return self.solutions.size()

    def to_csv(self, name, path=None):
        """
        Saves content in two distinct files with format csv.

        The first file contains self.solutions, the second one self.predictions.
        Both files are saved in the same directory and have names that start
        with the string given as an argument.

        Arguments
        ---------
        name : str
            name of the new file
        path : str
            path to file
        """
        self.solutions.save_csv(name + "_sol", path)
        self.predictions.save_csv(name + "_pred", path)
