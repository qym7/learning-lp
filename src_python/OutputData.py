"""
This module contains a class that primarily stocks data.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset import solutions
import os


class OutputData:
    """
    OutputData is a class stocking the exact solutions of a list of linear optimisation and
    the respective predictions of that solutions, as computed by a neural network (see
    class NeuralNetwork). Using the method NeuralNetwork.predict will automatically initiate
    an instance of OutputData.

    Attributes
    ----------
    solutions : float list or numpy array
        solutions to a list of linear optimisation problems
    predictions : float list or numpy array
        respective predictions of that solutions by a neural network
    """
    def __init__(self, sol_list, predict_list, network=None):
        self.solutions = solutions(sol_list.copy())
        self.predictions = solutions(predict_list.copy())
        s1, s2 = self.solutions.size(), self.predictions.size()
        assert s1 == s2, "Solutions and predictions do not have the same size"
        self.used_network = network

    def add_used_network(self, neural_network):
        self.used_network = neural_network

    def get_used_network(self):
        return self.used_network

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

    def get_analyser_name(self):
        return self.used_network.get_analyser_name()

    def size(self):
        """
        Returns number of samples in dataset instance.

        Returns
        -------
        size : int
            number of samples
        """
        return self.solutions.size()

    def copy(self):
        """Copies self and returns the copy."""
        return OutputData(self.get_solutions().copy(), self.get_predictions().copy())

    def to_csv(self, name, path=None, single_file=False):
        """
        Saves content in a single or two distinct files with format csv.

        If single_file is True, the content is saved in a single file.
        Else the content is saved in two separate files. More precisely,
        the first file contains self.solutions, the second one self.predictions.
        Both files are saved in the same directory and have names that start
        with the string given as an argument.

        Arguments
        ---------
        name : str
            name of the new file
        path : str
            path to file
        single_file : bool
            states whether self.RHS and self.solutions are saved in a single file
            or two separate files
        """
        import csv
        if single_file:
            reshaped_sol = np.reshape(self.get_solutions(), (self.size(), 1))
            reshaped_pre = np.reshape(self.get_predictions(), (self.size(), 1))
            content = np.concatenate((reshaped_sol, reshaped_pre), axis=1)
            full_name = name + ".csv"
            csv_path = os.path.join("." if path is None else path, full_name)
            with open(csv_path, 'w', newline='') as file_cont:
                writer = csv.writer(file_cont, delimiter=';')
                writer.writerows(content)
        else:
            self.solutions.save_csv(name + "_sol", path)
            self.predictions.save_csv(name + "_pre", path)
