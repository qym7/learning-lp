import tensorflow as tf
from OutputData import OutputData
from dataset import dataset
from DataProcessor import *
import numpy as np
import os
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    The class NeuralNetwork implements a neural networks and provides methods the build,
    modify and use a neural network.

    A NeuralNetwork instance takes a dataset instance (see dataset.dataset) as input for
    training, evaluation and prediction. The output after prediction is an OutputData instance
    (see OutputData.OutputData).

    Attributes
    ----------
    model : tensorflow.keras.Model or tensorflow.keras.Sequential()
        a tensorflow neural network
    loss : str, tf.keras.losses instance or other compatible function
        loss function of the network. Should be a tensorflow loss function (class tf.keras.losses)
        or the name (string) of a tensorflow loss function (ex. "mean_absolute_percentage_error",
        "mean_squared_error" etc.).
    optimizer : str, tf.keras.optimizers instance or other compatible optimizer
        optimizer of the network. Should be a tensorflow optimizer (class tf.keras.optimizers)
        or the name (string) of a tensorflow optimizer (ex. "Adam", "sgd" etc.).
    metrics : list containing str, tf.keras.metrics or other compatible metrics
        list of metrics of the network. Should be a list containing tensorflow metrics
        (class tf.keras.metrics) and/or the names of tensorflow metrics (ex. "mean_absolute_percentage_error",
        "mean_squared_error" etc.).
    file_name : str
        If network was loaded from file, name of that file. Else name that will be given to file if
        self network is stocked in file.
    analyser_name : str
        If graphs build on network predictions are saved in file, the file name will
        start with analyser_name
    compile : bool
        states whether network should be compiled before training or predicting
    bound_processing : list of instances of BoundProcessor subclasses
        list of BoundProcessors to be applied on the training or predictions dataset
        bounds before training or prediction (see DataProcessor).
    solutions_processing : list of instances of SolutionProcessor subclasses
        list of SolutionProcessors to be applied on the training or predictions dataset
        solutions before and after training or prediction (see DataProcessor).
    history : tensorflow keras History instance
        learning history of neural network
    """
    def __init__(self, model=None, file_name=None):
        self.model = tf.keras.Sequential() if model is None else model
        self.loss = "mean_absolute_percentage_error"
        self.optimizer = "Adam"
        self.metrics = ["mean_absolute_percentage_error"]
        self.file_name = file_name
        self.analyser_name = "predictions_" + file_name + ".pdf" if file_name is not None else "predictions.pdf"
        self.compile = True
        self.bound_processing = []
        self.solutions_processing = []
        self.history = None

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
        self.__init__(file_name=self.file_name)
        if list_neurons is not None:
            for nb_neurons in list_neurons:
                self.add_relu(nb_neurons)
        if last_activation is None:
            self.add_no_activation(1)
        else:
            self.model.add(tf.keras.layers.Dense(1, activation=last_activation))

    def add_relu(self, nb_neurons):
        """
        Adds layer with a given number of neurons and a relu activation to the network.

        Arguments
        ---------
        nb_neurons : int
            number of neurons in the layer to be added
        """
        self.model.add(tf.keras.layers.Dense(nb_neurons, activation="relu"))

    def add_sigmoid(self, nb_neurons):
        """
        Adds layer with a given number of neurons and a sigmoid activation to the network.

        Arguments
        ---------
        nb_neurons : int
            number of neurons in the layer to be added
        """
        self.model.add(tf.keras.layers.Dense(nb_neurons, activation="sigmoid"))

    def add_no_activation(self, nb_neurons):
        """
        Adds layer with a given number of neurons and no activation to the network.

        Arguments
        ---------
        nb_neurons : int
            number of neurons in the layer to be added
        """
        self.model.add(tf.keras.layers.Dense(nb_neurons))

    def compile_model(self):
        """
        Compiles the model if self.compile is True.

        The model is compiled with self.loss, self.optimizer and self.metrics.
        """
        if self.compile:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            self.compile = False

    def set_optimizer(self, optimizer_name):
        self.optimizer = optimizer_name

    def set_loss(self, loss_name):
        self.loss = loss_name

    def set_metrics(self, metrics):
        if isinstance(metrics, list):
            self.metrics = metrics
        else:
            self.metrics = [metrics]

    def get_analyser_name(self):
        return self.analyser_name

    def set_comp_parameters(self, optimizer_name, loss_name, metrics_name):
        """
        Sets parameters for compilation.
        """
        self.set_optimizer(optimizer_name)
        self.set_loss(loss_name)
        self.set_metrics(metrics_name)
        self.compile = True

    def is_compiled(self):
        """
        Sets self.compile to False.
        """
        self.compile = False

    def fit(self, pb_train, sol_train, epochs, validation_split, callbacks):
        """
        Auxiliary function to train_with (see NeuralNetwork.train_with).
        """
        return self.model.fit(x=pb_train, y=sol_train, epochs=epochs, callbacks=callbacks,
                              validation_split=validation_split)

    def fit_generator(self, generator, epochs, steps_per_epoch, callbacks, validation_generator):
        """
        Auxiliary function to train_with_generator (see NeuralNetwork.train_with_generator).
        """
        return self.model.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                                        workers=0, validation_data=validation_generator, validation_steps=1)

    def add_bound_processors(self, bound_processor_list):
        """
        Adds bound processors to self.bound_processing.
        """
        self.bound_processing.extend(bound_processor_list)

    def add_solution_processors(self, solution_processor_list):
        """
        Adds solution processors to self.solutions_processing.
        """
        self.solutions_processing.extend(solution_processor_list)

    def activate_processors(self):
        """
        Activates all DataProcessors in self.bound_processing
        and self.solutions_processing.
        """
        for processor in self.bound_processing:
            processor.activate()

        for processor in self.solutions_processing:
            processor.activate()

    def compute_processing_parameters(self, data):
        """
        Uses data (input data) to set the processing parameters of all DataProcessors in self.bound_processing
        and self.solutions_processing.

        Arguments
        ---------
        data : dataset instance
        """
        for processor in self.bound_processing:
            processor.compute_parameters(data)

        for processor in self.solutions_processing:
            processor.compute_parameters(data)

    def pre_process_data(self, data):
        """
        Pre-processes data (input data) using all activated DataProcessors in
        self.bound_processing and self.solutions_processing.

        Arguments
        ---------
        data : dataset instance
        """
        for processor in self.bound_processing:
            if processor.is_activated():
                processor.pre_process(data)

        for processor in self.solutions_processing:
            if processor.is_activated():
                processor.pre_process(data)

    def post_process_data(self, data):
        """
        Post.processes data (output data) using all DataProcessors in self.solution_processing
        in inverse order.

        Arguments
        ---------
        data : OutputData instance
        """
        nb_pro = len(self.solutions_processing)
        for i in range(nb_pro):
            processor = self.solutions_processing[nb_pro - 1 - i]
            if processor.is_activated():
                processor.post_process(data)

    def evaluate(self, data):
        """
        Evaluates the network's performance on data.

        Arguments
        ---------
        data : dataset instance

        Returns
        -------
        evaluation : float or float list
        """
        evaluation = self.model.evaluate(data.get_RHS(), data.get_solutions())
        return evaluation

    def predict(self, data):
        """
        Applies network on a given dataset instance and returns predictions of the solutions
        of the linear optimisation problems.

        If the network is not compiled yet (self.compile is True), it will be compiled
        before predicting (will only happen if network is not trained yet). Builds the
        first layer such that it fits the data if the model has not been used before.

        If self.bound_processing or self.solutions_processing are not empty the data
        is pre-processed by all activated DataProcessors in self.bound_processing and
        self.solutions_processing before applying the network. The predicted
        solutions are then post-processed by all activated SolutionProcessors in
        self.solutions_processing in inverse order.

        Arguments
        ---------
        data : dataset instance
            problems whose solutions should be predicted

        Returns
        -------
        object_to_analyze : OutputData instance
            containing predicted solutions and theoretical solutions
        """
        self.compile_model()
        new_data = data.copy()

        self.pre_process_data(new_data)

        object_to_analyze = OutputData(new_data.get_solutions(),
                                       self.model.predict(new_data.get_RHS()).flatten())

        self.post_process_data(object_to_analyze)
        object_to_analyze.add_used_network(self)

        return object_to_analyze

    def train_with(self, initial_data, epochs, validation_split, callbacks=None):
        """
        Trains the network self.model on a dataset instance given as an argument.

        If the network is not compiled yet (self.compile is True), it will be compiled
        before training. Builds the first layer such that it fits the data if the model
        has not been used before.

        If self.bound_processing or self.solutions_processing are not empty, the
        processing parameters of each DataProcessor are recomputed on the training
        data and the training data is pre-processed before training. self.history
        is reset.

        Arguments
        ---------
        initial_data : dataset instance
            training data
        epochs : int
            number of training epochs
        validation_split : float between 0 and 1
            ratio of data to be used for cross-validation
        callbacks : list of callback functions
        """
        self.compile_model()
        data = initial_data.copy()

        self.generate_name_for_analyser(data)

        self.compute_processing_parameters(data)
        self.activate_processors()
        self.pre_process_data(data)

        self.history = self.fit(data.get_RHS(), data.get_solutions(), epochs=epochs,
                                validation_split=validation_split, callbacks=callbacks)

    def train_with_generator(self, generator, epochs, steps_per_epoch, validation_generator, callbacks):
        """
        Method behaves like NeuralNetwork.train_with, except it requires a data generating function
        as argument instead of a training data set. Data for training will be generated while training
        by the generator function.

        Arguments
        ---------
        generator : generator function
            generates training data while training is in progress
        epochs : int
            number of training epochs
        steps_per_epoch : int
            size of dataset to be generated during each epoch
        validation_generator : float between 0 and 1
            generates cross-validation data while training is in progress
        callbacks : list of callback functions
        """
        self.compile_model()

        history = self.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                     validation_generator=validation_generator,
                                     callbacks=callbacks)

    def generate_name_for_analyser(self, data):
        number = data.size()
        if self.file_name is not None:
            name = "predictions_" + self.file_name + "trained_on_{}.pdf".format(number)
        else:
            name = "predictions_network_trained_on_{}.pdf".format(number)
        self.analyser_name = name

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
