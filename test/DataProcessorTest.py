from problem_generator import problem_generator
from Problem import Xpress_Problem_Factory
from dataset import dataset
from DataProcessor import *
import numpy as np


def test_preprocessing_postprocessing_linearmax():
    prob_list = ["petit_probleme.lp"]
    cons_to_vary = ["demand_0"]
    vars_to_vary = None
    number = 100
    deviation = 0.1

    data = problem_generator(prob_list, number, deviation, cons_to_vary, vars_to_vary,
                             Xpress_Problem_Factory(), save=False)
    test_data = data.copy()

    processor = SolutionProcessorLinearMax()
    processor.compute_parameters(test_data)

    processor.pre_process(test_data)

    processed_solutions = test_data.get_solutions().copy()
    output_test = OutputData(processed_solutions, processed_solutions)

    processor.post_process(output_test)

    solutions = data.get_solutions()
    test_solutions = output_test.get_solutions()

    size = test_data.size()

    for i in range(size):
        assert np.isclose(test_solutions[i], solutions[i], atol=1e-06, rtol=1e-06)


def test_preprocessing_postprocessing_normalise():
    prob_list = ["petit_probleme.lp"]
    cons_to_vary = ["demand_0"]
    vars_to_vary = None
    number = 100
    deviation = 0.1

    data = problem_generator(prob_list, number, deviation, cons_to_vary, vars_to_vary,
                             Xpress_Problem_Factory(), save=False)
    test_data = data.copy()

    processor = SolutionProcessorNormalise()
    processor.compute_parameters(test_data)

    processor.pre_process(test_data)

    processed_solutions = test_data.get_solutions().copy()
    output_test = OutputData(processed_solutions, processed_solutions)

    processor.post_process(output_test)

    solutions = data.get_solutions()
    test_solutions = output_test.get_solutions()

    size = test_data.size()

    for i in range(size):
        assert np.isclose(test_solutions[i], solutions[i], atol=1e-06, rtol=1e-06)
