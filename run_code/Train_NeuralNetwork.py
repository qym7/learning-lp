import sys
import tensorflow as tf
from NeuralNetwork import NeuralNetwork
from DataProcessor import *
from DataAnalyser import *
from LearningRateSchedulers import *
from Losses import SafeguardedAbsolutePercentageError

"""Run this file to train a neural network."""

if __name__ == '__main__':

    if True:

        """Loading data from files. File names are given as script parameters."""

        path = sys.argv[1]

        file_name = sys.argv[2]
        data = load_csv_single_file(file_name, path=path)

        pred_file_name = sys.argv[3]
        data_for_prediction = load_csv_single_file(pred_file_name, path=path)

        """Setting training parameters"""

        depth = 10
        layers = [500 - 50 * i for i in range(depth)]
        epochs = 80
        validation_split = 0.3

        """Setting callbacks"""

        callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler_ssn_opt_on_model_input_5times40_1),
                     tf.keras.callbacks.ModelCheckpoint(filepath="D:\\repository\\learning-lp\\data\\Model_checkpoints",
                                                        save_weights_only=True, verbose=0)]
        #callback = EscapeLocalMinima(scheduler_LandS_opt_on_model_1_100_1, considered_epochs=3, threshold=4e-3)

        """Creating neural network."""

        network = NeuralNetwork(file_name="ssn_network_input_(500 - 50 * layer)_1")
        network.basic_nn(layers)
        network.add_bound_processors([BoundProcessorNormalise()])
        network.add_solution_processors([SolutionProcessorLinearMax()])
        network.set_loss(SafeguardedAbsolutePercentageError())

        """Training neural network."""

        network.train_with(data, epochs, validation_split, callbacks=callbacks)

        """Saving model."""

        network.graph_save(path="D:\\repository\\learning-lp\\data\\Trained_networks")
        #network.save_model()

        """Predicting solutions on data_for_prediction."""

        Output = network.predict(data_for_prediction)

        """Analysing network's performance."""

        print(OutputDataAnalyser(Output).mean_precision_error())
        #DataAnalyser(data_for_prediction, Output).performance_plot_2D(save=True)

    if False:
        """Loading data from files. File names are given as script parameters."""

        path = sys.argv[1]

        file_name = sys.argv[2]
        data = load_csv_single_file(file_name, path=path)

        pred_file_name = sys.argv[3]
        data_for_prediction = load_csv_single_file(pred_file_name, path=path)

        """Setting training parameters"""

        depth = 1
        layers = [500 for i in range(depth)]
        epochs = 80
        validation_split = 0.3
        good_enough = 1e-4
        mape = 1

        """Setting callbacks"""

        callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler_LandS_opt_on_model_1_500_1),
                     tf.keras.callbacks.ModelCheckpoint(filepath="D:\\repository\\learning-lp\\data\\Model_checkpoints",
                                                        save_weights_only=True, verbose=0)]
        #callback = EscapeLocalMinima(scheduler_LandS_opt_on_model_1_100_1, considered_epochs=3, threshold=4e-3)

        while mape >= good_enough:

            """Creating neural network."""

            network = NeuralNetwork(file_name="LandS_network_7_500_1")
            network.basic_nn(layers)
            network.add_bound_processors([BoundProcessorNormalise()])
            network.add_solution_processors([SolutionProcessorLinearMax()])
            #network.set_loss(MeanLogarithmicError())

            """Training neural network."""

            network.train_with(data, epochs, validation_split, callbacks=callbacks)

            """Predicting solutions on data_for_prediction."""

            Output = network.predict(data_for_prediction)

            """Analysing network's performance."""

            mape = OutputDataAnalyser(Output).mean_precision_error()
            print(mape)
            # DataAnalyser(data_for_prediction, Output).performance_plot_2D(save=True)

        """Saving model."""

        network.graph_save(path="D:\\repository\\learning-lp\\data\\Trained_networks")
        #network.save_model()


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

    if False:

        network = load_model("LandS_network_1_500_1_5e-4", path="D:\\repository\\learning-lp\\data\\Trained_networks",
                             input_names=["S2C5", "S2C6", "S2C7", "X1", "X2", "X3", "X4"])
        network.graph_save("LandS_network_1_500_1_5e-4", path="D:\\repository\\learning-lp\\data\\Trained_networks")

    if False:
        network = graph_load("LandS_network_1_500_1_layers_3e-6.model", "LandS_network_1_500_1_activations_3e-6.model",
                             "LandS_network_1_500_1_processing_3e-6.model",
                             path="D:\\repository\\learning-lp\\data\\Trained_networks")

        path = sys.argv[1]

        pred_file_name = sys.argv[3]
        data_for_prediction = load_csv_single_file(pred_file_name, path=path)

        Output = network.predict(data_for_prediction)

        print(OutputDataAnalyser(Output).mean_precision_error())



