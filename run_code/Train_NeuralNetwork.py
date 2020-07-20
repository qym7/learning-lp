import sys
import tensorflow as tf
from dataset import load_csv
from NeuralNetwork import NeuralNetwork, load_model
from DataProcessor import *
from DataAnalyser import *
from LearningRateSchedulers import *
from Losses import MeanLogarithmicError, SymmetricMeanAbsolutePercentageError
from CustomCallbacks import EscapeLocalMinima
from problem_generator import problem_generator
from problem_interface import Problem, Problem_factory
from problem_cplex import Cplex_Problem_Factory
from Problem_xpress import Xpress_Problem_Factory

"""Run this file to train a neural network."""

if __name__ == '__main__':

    if False:

        """Loading data from files. File names are given as script parameters."""

        path = sys.argv[1]

        file_name = sys.argv[2]
        data = load_csv_single_file(file_name, path=path)

        pred_file_name = sys.argv[3]
        data_for_prediction = load_csv_single_file(pred_file_name, path=path)

        """Setting training parameters"""

        depth = 4
        layers = [(20 - 4 * i) for i in range(depth)]
        epochs = 120
        validation_split = 0.3

        """Setting callbacks"""

        callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler_LandS_opt_on_model_1_100_1),
                     tf.keras.callbacks.ModelCheckpoint(filepath="D:\\repository\\learning-lp\\data\\Model_checkpoints",
                                                        save_weights_only=True, verbose=0)]
        #callback = EscapeLocalMinima(scheduler_LandS_opt_on_model_1_100_1, considered_epochs=3, threshold=4e-3)

        """Creating neural network."""

        network = NeuralNetwork(file_name="LandS_network_1_20_16_12_8_4_1_")
        network.basic_nn(layers)
        network.add_bound_processors([BoundProcessorNormalise(), BoundProcessorAddConst()])
        network.add_solution_processors([SolutionProcessorLinearMax()])
        #network.set_loss(MeanLogarithmicError())

        """Training neural network."""

        network.train_with(data, epochs, validation_split, callbacks=callbacks)

        """Saving model."""

        network.save_model()

        """Predicting solutions on data_for_prediction."""

        Output = network.predict(data_for_prediction)

        """Analysing network's performance."""

        #DataAnalyser(data_for_prediction, Output).performance_plot_2D(save=True)

    if False:

        layer_size_list = [30000]
        data_size_list = [3, 5, 10, 100, 1000]

        prediction_set_size = 100

        for neurons in layer_size_list:
            for elem in data_size_list:
                """Generating data."""

                nb_prob = int(sys.argv[1])
                nb_cons = int(sys.argv[2])
                nb_vars = int(sys.argv[3])

                if nb_prob == 1:
                    prob_list = [sys.argv[4]]
                else:
                    prob_list = sys.argv[4:4 + nb_prob]
                if nb_cons == 0:
                    cons_to_vary = None
                else:
                    cons_to_vary = sys.argv[4 + nb_prob:4 + nb_prob + nb_cons]
                if nb_vars == 0:
                    vars_to_vary = None
                else:
                    vars_to_vary = sys.argv[4 + nb_prob + nb_cons:]

                Number = elem
                Deviation = 0

                data = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary,
                                         Xpress_Problem_Factory(), save=True)

                Number = prediction_set_size
                Deviation = 0

                data_for_prediction = problem_generator(prob_list, Number, Deviation, cons_to_vary,
                                                        vars_to_vary, Xpress_Problem_Factory(), save=False)

                """Setting training parameters"""

                layers = [neurons]
                epochs = 100
                validation_split = 0.3

                """Creating neural network."""

                network = NeuralNetwork(file_name="network_1_{}_1_".format(neurons))
                network.basic_nn(layers)
                network.add_bound_processors([BoundProcessorNormalise(), BoundProcessorAddConst()])
                network.add_solution_processors([SolutionProcessorNormalise()])

                """Training neural network."""

                network.train_with(data, epochs, validation_split)

                """Saving model."""

                network.save_model("test_network")

                """Predicting solutions on data_for_prediction."""

                Output = network.predict(data_for_prediction)

                """Analysing network's performance."""

                DataAnalyser(data_for_prediction, Output).performance_plot_2D(save=True)

    if True:

        network = load_model("LandS_network_1_500_1_5e-4", path="D:\\repository\\learning-lp\\data\\Trained_networks",
                             input_names=["S2C5", "S2C6", "S2C7", "X1", "X2", "X3", "X4"])
        network.graph_save("LandS_network_1_500_1_5e-4", path="D:\\repository\\learning-lp\\data\\Trained_networks")




