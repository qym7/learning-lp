import tensorflow as tf
from OutputData import OutputData
from dataset import dataset
from DataProcessor import *
import numpy as np
import os
import matplotlib.pyplot as plt


# This class implements a neural network. The neural_network is trained and tested with an instance of dataset
# This class allows to modify the neural network

### example : how to create a basic neural network with a pre_processing
#
#   neural_network = NeuralNetwork()
#   neural_network.basic_nn([20, 20], last_activation = "sigmoid")     /!\ It is possible to set last_activation = None
#   neural_network.add_processing_linear_mean
#   neural_network.add_processing_add_const(1)
#
#   and neural_network is ready !


class NeuralNetwork:
    def __init__(self, model=tf.keras.models.Sequential(), file_name=None):
        self.model = model
        self.loss = "mean_absolute_percentage_error"
        self.optimizer = "Adam"
        self.metrics = ["mean_absolute_percentage_error"]
        self.file_name = file_name
        self.compile = True
        self.bound_processing = []
        self.solutions_processing = []

    def basic_nn(self, list_neurons=None, last_activation=None):
        """
        Reinitialises self.model with a given architecture.

        The number of layers and neurons are given in list_neurons. More precisely the element n of
        list_neurons gives the number of neurons in the layer n+2 of the neural network. Note that the first
        layer is generated automatically by tensorflow when fitting the training data. Furthermore the
        method generates an additional last layer with a single neuron. Except for the last layer,
        activation functions will be "relu" by default.

        Arguments
        ---------
        list_neurons : int list
            giving the architecture of the network to be generated (see description above).
        last_activation : str, tf.keras.activation instance, tf.nn instance or other compatible function
           activation function of the last layer. Should be a tensorflow activation function
           (class tf.keras.activation or tf.nn) or the name (string) of a tensorflow activation function
           (ex. "relu", "sigmoid" etc.). If None, the last layer is created without activation.
        """
        self.__init__()
        if list_neurons is not None:
            for nb_neurons in list_neurons:
                self.add_relu(nb_neurons)
        if last_activation is None:
            self.add_no_activation(1)
        else:
            self.model.add(tf.keras.layers.Dense(1, activation=last_activation))

    def add_relu(self, nb_neurons):
        self.model.add(tf.keras.layers.Dense(nb_neurons, activation="relu"))

    def add_sigmoid(self, nb_neurons):
        self.model.add(tf.keras.layers.Dense(nb_neurons, activation="sigmoid"))

    def add_no_activation(self, nb_neurons):
        self.model.add(tf.keras.layers.Dense(nb_neurons))

    def compile_model(self):
        if self.compile:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            self.compile = False

    def set_optimizer(self, optimizer_name):
        self.optimizer = optimizer_name

    def set_loss(self, loss_name):
        self.loss = loss_name

    def set_metrics(self, metrics_name):
        self.metrics = [metrics_name]

    def set_comp_parameters(self, optimizer_name, loss_name, metrics_name):
        self.set_optimizer(optimizer_name)
        self.set_loss(loss_name)
        self.set_metrics(metrics_name)
        self.compile = True

    def is_compiled(self):
        self.compile = False

    def fit(self, pb_train, sol_train, epochs, validation_split, callbacks):
        return self.model.fit(x=pb_train, y=sol_train, epochs=epochs, callbacks=callbacks,
                              validation_split=validation_split)

    def fit_generator(self, generator, epochs, steps_per_epoch, callbacks, validation_generator):
        return self.model.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                                        workers=0, validation_data=validation_generator, validation_steps=1)

    def add_bound_processors(self, bound_processor_list):
        self.bound_processing.extend(bound_processor_list)

    def add_solution_processors(self, solution_processor_list):
        self.solutions_processing.extend(solution_processor_list)

    def compute_processing_parameters(self, data):
        for processor in self.bound_processing:
            processor.compute_parameters(data)

        for processor in self.solutions_processing:
            processor.compute_parameters(data)

    def pre_process_data(self, data):
        for processor in self.bound_processing:
            processor.pre_process(data)

        for processor in self.solutions_processing:
            processor.pre_process(data)

    def post_process_data(self, data):
        for processor in self.solutions_processing:
            processor.post_process(data)

    def evaluate(self, dataset_instance):
        """ Evaluates the network with the dataset. Arguments : class dataset Out : class to_analyze"""
        evaluation = self.model.evaluate(dataset_instance.get_RHS(), dataset_instance.get_solutions())
        return evaluation

    def predict(self, data):
        self.compile_model()
        new_data = data.copy()

        self.pre_process_data(new_data)

        object_to_analyze = OutputData(new_data.get_solutions(),
                                       self.model.predict(new_data.get_RHS()).flatten())

        self.post_process_data(object_to_analyze)
        object_to_analyze.add_used_network(self)

        return object_to_analyze

    def train_with(self, initial_data, epochs, validation_split, callbacks):
        """ Trains the network using the dataset. Arguments : class dataset Out : class to_analyze"""
        self.compile_model()
        data = initial_data.copy()

        self.compute_processing_parameters(data)
        self.pre_process_data(data)

        history = self.fit(data.get_RHS(), data.get_solutions(), epochs=epochs,
                           validation_split=validation_split, callbacks=callbacks)
        object_to_analyze = self.predict(initial_data)
        object_to_analyze.add_learning_history(history)
        object_to_analyze.add_used_network(self)

        return object_to_analyze

    def train_with_generator(self, generator, epochs, steps_per_epoch, validation_generator, callbacks):
        """ Trains the network using the dataset. Arguments : class dataset Out : class to_analyze"""
        self.compile_model()

        history = self.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                     validation_generator=validation_generator,
                                     callbacks=callbacks)  # training the network

    def save_hdf5(self, name=None, path=None):
        """
        Saves the model in file with HDF5 format.

        Arguments
        ---------
        name : str
            name of the new file. If name is None, the file name will be self.name. Else self.name
            will be reset to name.
        path : str
            path to file
        """
        if name is None:
            assert self.file_name is not None, "No name :("
            name = self.file_name
        else:
            self.file_name = name
        path = os.path.join("." if path is None else path, "Trained_networks", name)
        self.model.save(path + ".h5")

    def save_model(self, name=None, path=None):
        """
        Saves the model in file with SavedModel format.

        Arguments
        ---------
        name : str
            name of the new file. If name is None, the file name will be self.name. Else self.name
            will be reset to name.
        path : str
            path to file
        """
        if name is None:
            assert self.file_name is not None, "No name :("
            name = self.file_name
        else:
            self.file_name = name
        path = os.path.join("." if path is None else path, "Trained_networks", name)
        self.model.save(path)

    def get_details(self):
        """
        Prints all the details of the network.
        """
        self.model.summary()


def load_model(file_name, path=None):
    """
    Loads a neural network from a file into a NeuralNetwork instance.

    The format of the file should be either hdf5 or the tensorflow SavedModel
    format. The function loads a complete model and not only the weights of
    a trained model. In particular, the model is possibly already compiled.
    Since the optimizer-state is recovered, training can be resumed where
    it was interrupted.

    Arguments
    ---------
    path : str
       path to file
    file_name : str
       name of file containing neural_network
    """
    path = os.path.join("." if path is None else path, file_name)
    new_model = tf.keras.models.load_model(path)
    network = NeuralNetwork(new_model, file_name)
    network.is_compiled()
    return network
