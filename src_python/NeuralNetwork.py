import tensorflow as tf
from OutputData import OutputData
from dataset import dataset
from DataProcessor import *
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time


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
    input_names : str
        names of constraints that were varied to create the data the network
        was trained on
    """
    def __init__(self, model=None, file_name=None, input_names=None):
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
        self.input_names = input_names

    def basic_nn(self, list_neurons=None, last_activation=None, specify_input=False, input_shape=None):
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
        specify_input : bool
            states whether first layer should be built
        input_shape : int
            size of input data
        """
        self.__init__(file_name=self.file_name, input_names=self.input_names)
        if specify_input:
            if list_neurons is not None:
                nb_layers = len(list_neurons)
                self.add_input_relu(list_neurons[0], input_shape)
                for i in range(1, nb_layers):
                    self.add_relu(list_neurons[i])
            if last_activation is None:
                self.add_no_activation(1)
            else:
                self.model.add(tf.keras.layers.Dense(1, activation=last_activation))
        else:
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

    def add_input_relu(self, nb_neurons, input_shape):
        """
        Adds layer with a given number of neurons and a relu activation to the network,
        where input shape is specified. Useful when creating the first layer.

        Arguments
        ---------
        nb_neurons : int
            number of neurons in the layer to be added
        input_shape : int
            size of input data
        """
        self.model.add(tf.keras.layers.Dense(nb_neurons, input_shape=(input_shape,), activation='relu'))

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

    def predict(self, data, print_time=True):
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
        print_time : True
            states whether time network took for prediction should be printed
        """
        self.compile_model()
        new_data = data.copy()

        begin = time()

        self.pre_process_data(new_data)

        object_to_analyze = OutputData(new_data.get_solutions(),
                                       self.model.predict(new_data.get_RHS()).flatten())

        self.post_process_data(object_to_analyze)

        end = time()

        if print_time:
            print(end - begin)

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
        self.generate_input_names(initial_data)

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

    def generate_input_names(self, data):
        input_names = data.get_input_names()
        if input_names is None:
            nb_inp = len(data.get_RHS()[0])
            names = nb_inp * [None]
            for i in range(nb_inp):
                names[i] = "Input{}".format(i)
            self.input_names = names
        else:
            self.input_names = input_names

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

    def graph_save(self, name=None, path=None, light_mode=False):
        """
        Saves the model in file with .model format.

        Arguments
        ---------
        name : str
            name of the new file. If name is None, the file name will be self.name. Else self.name
            will be reset to name.
        path : str
            path to file
        light_mode : bool
            if True, file is stocked in more compact format
        """
        if self.input_names is None:
            raise Exception("Set input_names or save in another format.")

        if name is None:
            assert self.file_name is not None, "No name :("
            name = self.file_name
        else:
            self.file_name = name

        layers = self.model.layers
        nb_layers = len(layers)
        nb_input = len(self.input_names)

        name_layers = name + "_layers.model"
        name_activations = name + "_activations.model"
        name_processing = name + "_processing.model"
        file_layers = open(os.path.join("." if path is None else path, name_layers), "w")
        file_activations = open(os.path.join("." if path is None else path, name_activations), "w")
        file_processing = open(os.path.join("." if path is None else path, name_processing), "w")

        file_layers.write("Layer" + (7 * " ") + "Neuron" + (7 * " ") + "Connected_with" + (7 * " ") + "Weight\n")
        file_activations.write("Layer" + (7 * " ") + "Neuron" + (7 * " ") + "Output_Activation\n")

        add_bias_number_first_layer = 0

        for processor in self.bound_processing:

            if isinstance(processor, BoundProcessorAddConst):
                add_bias_number_first_layer += processor.number_of_const

            if isinstance(processor, BoundProcessorNormalise):
                if processor.mean_list is None or len(processor.mean_list.shape) == 0:
                    mean_list = processor.scaler.mean_
                    dev_list = processor.scaler.scale_
                else:
                    mean_list = processor.mean_list
                    dev_list = processor.dev_list

                file_processing.write("Input-processing" + (7 * " ") + "Normalisation" + (7 * " ") + "Mean_list"
                                      + (7 * " "))
                for elem in mean_list:
                    file_processing.write(str(elem) + (7 * " "))
                file_processing.write("\n")

                file_processing.write("Input-processing" + (7 * " ") + "Normalisation" + (7 * " ") + "Dev_list"
                                      + (8 * " "))
                for elem in dev_list:
                    file_processing.write(str(elem) + (7 * " "))
                file_processing.write("\n")

        if add_bias_number_first_layer > 0:
            file_processing.write("Input-processing" + (7 * " ") + "Additional_bias" + (5 * " ")
                                  + str(add_bias_number_first_layer) + "\n")

        for processor in self.solutions_processing:

            if isinstance(processor, SolutionProcessorLinearMax):
                file_processing.write("Output-processing" + (6 * " ") + "Multiply_by" + (9 * " ") +
                                      str(processor.max) + "\n")

        if light_mode:

            file_layers.write("Input\n")
            file_activations.write("Input\n")

            layer = layers[0]
            weights = layer.get_weights()
            nb_bias = len(weights[0]) - nb_input

            file_layers.write((12 * " ") + "Bias0_{}\n".format(0))
            file_activations.write((12 * " ") + "Bias0_{}\n".format(0))
            file_activations.write((25 * " ") + "Linear\n")

            nb_conn = len(weights[0][0])

            for i in range(nb_conn):
                file_layers.write((25 * " ") + "Neuron1_{}".format(i) +
                                  ((12 - int(np.log10(max(i, 1)))) * " ") + str(weights[1][i]) + "\n")

            for i in range(nb_bias):
                file_layers.write((12 * " ") + "Bias0_{}\n".format(i + 1))
                file_activations.write((12 * " ") + "Bias0_{}\n".format(i + 1))
                file_activations.write((25 * " ") + "Linear\n")
                for j in range(nb_conn):
                    file_layers.write((25 * " ") + "Neuron1_{}".format(j) +
                                      ((12 - int(np.log10(max(1, j)))) * " ") + str(weights[0][i][j]) + "\n")

            for i in range(nb_input):
                file_layers.write((12 * " ") + self.input_names[i] + "\n")
                file_activations.write((12 * " ") + self.input_names[i] + "\n")
                file_activations.write((25 * " ") + "Linear\n")
                for j in range(nb_conn):
                    file_layers.write((25 * " ") + "Neuron1_{}".format(j) +
                                      ((12 - int(np.log10(max(1, j)))) * " ") +
                                      str(weights[0][i + nb_bias][j]) + "\n")

            for i in range(1, nb_layers):
                layer = layers[i]
                weights = layer.get_weights()
                nb_neurons = len(weights[0])
                nb_conn = len(weights[0][0])

                file_layers.write("Layer{}\n".format(i))
                file_activations.write("Layer{}\n".format(i))

                file_layers.write((12 * " ") + "Bias{}_0\n".format(i))
                file_activations.write((12 * " ") + "Bias{}_0\n".format(i))
                file_activations.write((25 * " ") + "Relu\n")

                for j in range(nb_conn):
                    file_layers.write((25 * " ") + "Neuron{}_{}".format(i + 1, j) +
                                      ((12 - int(np.log10(max(1, j))) - int(np.log10(max(1, i + 1)))) * " ")
                                      + str(weights[1][j]) + "\n")

                for j in range(nb_neurons):
                    file_layers.write((12 * " ") + "Neuron{}_{}\n".format(i, j))
                    file_activations.write((12 * " ") + "Neuron{}_{}\n".format(i, j))
                    file_activations.write((25 * " ") + "Relu\n")

                    for k in range(nb_conn):
                        file_layers.write((25 * " ") + "Neuron{}_{}".format(i + 1, k) +
                                          ((12 - int(np.log10(max(k, 1))) - int(np.log10(max(1, i + 1)))) * " ")
                                          + str(weights[0][j][k]) + "\n")

            file_layers.write("Output\n")
            file_layers.write((12 * " ") + "Neuron{}_0\n".format(nb_layers))

            file_activations.write("Output\n")
            file_activations.write((12 * " ") + "Neuron{}_0\n".format(nb_layers))
            file_activations.write((25 * " ") + "Linear\n")

        else:

            layer = layers[0]
            weights = layer.get_weights()

            nb_bias = len(weights[0]) - nb_input
            nb_conn = len(weights[0][0])

            for i in range(nb_conn):
                file_activations.write("Input" + (7 * " ") + "Bias0_0" + (6 * " ") + "Linear\n")
                file_layers.write("Input" + (7 * " ") + "Bias0_0" + (6 * " ") + "Neuron1_{}".format(i) +
                                  ((12 - int(np.log10(max(i, 1)))) * " ") + str(weights[1][i]) + "\n")

            for i in range(nb_bias):
                file_activations.write("Input" + (7 * " ") + "Bias0_{}".format(i + 1) + (6 * " ") + "Linear\n")
                for j in range(nb_conn):
                    file_layers.write("Input" + (7 * " ") + "Bias0_{}".format(i + 1) + (6 * " ") +
                                      "Neuron1_{}".format(j) +
                                      ((12 - int(np.log10(max(1, j)))) * " ") + str(weights[0][i][j]) + "\n")

            for i in range(nb_input):
                elem = self.input_names[i]
                length = len(elem)
                file_activations.write("Input" + (7 * " ") + elem + ((13 - length) * " ") + "Linear\n")
                for j in range(nb_conn):
                    file_layers.write("Input" + (7 * " ") + elem + ((13 - length) * " ") +
                                      "Neuron1_{}".format(j) +
                                      ((12 - int(np.log10(max(1, j)))) * " ") + str(weights[0][i + nb_bias][j]) + "\n")

            for i in range(1, nb_layers):
                layer = layers[i]
                weights = layer.get_weights()
                nb_neurons = len(weights[0])
                nb_conn = len(weights[0][0])

                file_activations.write("Layer{}".format(i) + ((6 - int(np.log10(max(1, i)))) * " ") +
                                       "Bias{}_0".format(i) + ((6 - int(np.log10(max(1, i)))) * " ") + "Relu\n")

                for j in range(nb_conn):
                    file_layers.write("Layer{}".format(i) + ((6 - int(np.log10(max(1, i)))) * " ") +
                                      "Bias{}_0".format(i) + ((6 - int(np.log10(max(1, i)))) * " ") +
                                      "Neuron{}_{}".format(i + 1, j) +
                                      ((12 - int(np.log10(max(1, j))) - int(np.log10(max(1, i+1)))) * " ")
                                      + str(weights[1][j]) + "\n")

                for j in range(nb_neurons):
                    file_activations.write("Layer{}".format(i) + ((6 - int(np.log10(max(1, i)))) * " ")
                                           + "Neuron{}_{}".format(i, j) +
                                           ((4 - int(np.log10(max(1, i))) - int(np.log10(max(1, j)))) * " ") + "Relu\n")

                    for k in range(nb_conn):
                        file_layers.write("Layer{}".format(i) + ((6 - int(np.log10(max(1, i)))) * " ") +
                                          "Neuron{}_{}".format(i, j) +
                                          ((4 - int(np.log10(max(1, i))) - int(np.log10(max(1, j)))) * " ")
                                          + "Neuron{}_{}".format(i + 1, k) +
                                          ((12 - int(np.log10(max(1, k))) - int(np.log10(max(1, i + 1)))) * " ")
                                          + str(weights[0][j][k]) + "\n")

            file_layers.write("Output" + (6 * " ") + "Neuron{}_0\n".format(nb_layers))
            file_activations.write("Output" + (6 * " ") + "Neuron{}_0".format(nb_layers) + (4 * " ") + "Linear\n")
            
    def get_details(self):
        """
        Prints all the details of the network.
        """
        self.model.summary()


def load_model(file_name, path=None, input_names=None):
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
    input_names : str list
        names of constraints that were varied to create the data the network
        was trained on
    """
    path = os.path.join("." if path is None else path, file_name)
    new_model = tf.keras.models.load_model(path)
    network = NeuralNetwork(new_model, file_name, input_names=input_names)
    network.is_compiled()
    return network


def graph_load(file_layers, file_activations, file_processing, path=None):
    file_layers_path = os.path.join("." if path is None else path, file_layers)
    file_activations_path = os.path.join("." if path is None else path, file_activations)
    file_processing_path = os.path.join("." if path is None else path, file_processing)

    file_l = open(file_layers_path, "r")
    file_a = open(file_activations_path, "r")
    file_p = open(file_processing_path, "r")

    weight_list = file_l.readlines()[1:]
    nb_lines = len(weight_list)

    network = []
    layer = [None, None]
    neurons_weights = []
    bias_weights = []
    neuron_weights = []
    first_of_layer = True

    layer_name = None
    neuron_name = None

    for i in range(nb_lines):
        line = weight_list[i].split()
        if layer_name is None:
            layer_name = line[0]
            neuron_name = line[1]
            if neuron_name.startswith("Bias") and first_of_layer:
                bias_weights.append(float(line[3]))
            else:
                neuron_weights.append(float(line[3]))
        else:
            if layer_name == line[0]:
                if neuron_name == line[1]:
                    if neuron_name.startswith("Bias") and first_of_layer:
                        bias_weights.append(float(line[3]))
                    else:
                        neuron_weights.append(float(line[3]))
                else:
                    if first_of_layer:
                        first_of_layer = False
                    else:
                        neurons_weights.append(neuron_weights)
                    neuron_name = line[1]
                    neuron_weights = [float(line[3])]
            else:
                first_of_layer = True
                layer_name = line[0]
                neuron_name = line[1]
                neurons_weights.append(neuron_weights)
                layer[0] = np.array(neurons_weights)
                layer[1] = np.array(bias_weights)
                network.append(layer)
                layer = [None, None]
                neurons_weights = []
                neuron_weights = []
                bias_weights = []
                if neuron_name.startswith("Bias") and first_of_layer:
                    bias_weights.append(float(line[3]))
                else:
                    if len(line) >= 3:
                        neuron_weights.append(float(line[3]))

    processors = file_p.readlines()
    nb_proc = len(processors)
    processor = None
    boundprocesors = []
    solutionprocessors = []

    for i in range(nb_proc):

        line = processors[i].split()

        if line[1] == "Normalisation":
            if line[2] == "Mean_list":
                mean_list = line[3:]
                dev_list = processors[i+1].split()[3:]
                for i in range(len(mean_list)):
                    mean_list[i] = float(mean_list[i])
                    dev_list[i] = float(dev_list[i])
                processor = BoundProcessorNormalise(mean_list, dev_list)
            else:
                if line[0] == "Input-processing":
                    boundprocesors.append(processor)
                else:
                    solutionprocessors.append(processor)

        if line[1] == "Additional_bias":
            processor = BoundProcessorAddConst(number_of_const=int(line[2]))
            boundprocesors.append(processor)

        if line[1] == "Multiply_by":
            processor = SolutionProcessorLinearMax()
            processor.max = float(line[2])
            solutionprocessors.append(processor)

    nb_layers = len(network)
    layer_sizes = nb_layers * [None]

    for i in range(nb_layers):
        size = len(network[i][0])
        layer_sizes[i] = size

    neural_network = NeuralNetwork(file_name=file_layers[:-12])
    neural_network.basic_nn(list_neurons=layer_sizes[1:], specify_input=True, input_shape=layer_sizes[0])

    layers = neural_network.model.layers

    for i in range(nb_layers):
        layers[i].set_weights(network[i])

    neural_network.add_bound_processors(boundprocesors)
    neural_network.add_solution_processors(solutionprocessors)
    neural_network.activate_processors()

    return neural_network
