import sys
import os
import numpy as np
from problem_generator import problem_generator
from problem_interface import Problem, Problem_factory
from problem_cplex import Cplex_Problem_Factory
from Problem_xpress import Xpress_Problem_Factory
from DataAnalyser import DatasetAnalyser
from GenerationMode import GenerationModeClassic, GenerationMode, \
    GenerationModeMasterSlaveContinuous, GenerationModeMasterSlaveDiscreet

if __name__ == '__main__':

    if False:

        number_list = [50000]

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

        Deviation = 0

        for N in number_list:
            Number = N

            data = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary, Xpress_Problem_Factory(),
                                     save=True, single_file=True)

            #DatasetAnalyser(data).plot2D_sol_fct_of_RHS()

    if False:

        var_list = ["C1", "C6", "C7", "C8", "C9"]
        Number = 10000
        Deviation = 0
        prob_list = ["problem_rte_2.lp"]
        cons_to_vary = None
        for elem in var_list:
            vars_to_vary = [elem]
            data = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary, Xpress_Problem_Factory(),
                                     save=True, single_file=True)

    if False:
        """Generates data for test in Analyse_data.py. """

        prob_list = ["problem_rte_1.lp"]
        vars_to_vary = ["C1"]
        cons_to_vary = None
        Number = 100
        Deviation = 0

        data1 = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary, Xpress_Problem_Factory(),
                                  save=True)

        data2 = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary, Xpress_Problem_Factory(),
                                  save=True, name="the_same_again")

    if True:

        number_list = [300000]
        Deviation = 0

        path = sys.argv[1]
        prob_list = sys.argv[2:5]

        for N in number_list:
            Number = N

            data = problem_generator(prob_list, Number, Deviation, factory=Xpress_Problem_Factory(),
                                     save=True, single_file=True, mode=GenerationModeMasterSlaveDiscreet(),
                                     find_path=path)

