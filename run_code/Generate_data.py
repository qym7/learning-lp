import sys
from problem_generator import problem_generator
from ProblemTypeCplex import CplexProblemFactory
from ProblemTypeXpress import XpressProblemFactory
from GenerationMode import GenerationModeMasterSlaveDiscreet, GenerationModeMasterSlaveContinuous, GenerationModeClassic
from GenerationModules import GenerationModuleStorm


if __name__ == '__main__':

    if False:

        number_list = [50000]

        path = sys.argv[1]
        nb_prob = int(sys.argv[2])
        nb_cons = int(sys.argv[3])
        nb_vars = int(sys.argv[4])
        determine_cons_to_vary = False

        if nb_prob == 1:
            prob_list = [sys.argv[5]]
        else:
            prob_list = sys.argv[5:5 + nb_prob]
        if nb_cons == 0:
            cons_to_vary = None
            determine_cons_to_vary = True
        else:
            cons_to_vary = sys.argv[5 + nb_prob:5 + nb_prob + nb_cons]
        if nb_vars == 0:
            vars_to_vary = None
        else:
            vars_to_vary = sys.argv[5 + nb_prob + nb_cons:]

        Deviation = 0

        for N in number_list:
            Number = N

            mode = GenerationModeMasterClassic(prob_list, cons_to_vary, vars_to_vary, determine_cons_to_vary)
            data = problem_generator(Number, Deviation, mode=mode, factory=XpressProblemFactory(), save=True,
                                     single_file=True, find_path=path)

            #DatasetAnalyser(data).plot2D_sol_fct_of_RHS()

    if True:

        number_list = [300]
        Deviation = 0

        path = sys.argv[1]
        prob_list = sys.argv[2:5]
        problem = prob_list[0]
        master = prob_list[1]
        sto = prob_list[2]

        for N in number_list:
            Number = N
            mode = GenerationModeMasterSlaveDiscreet(master, problem, sto, use_random_vertices_method=True,
                                                     modules=[GenerationModuleStorm()])
            data = problem_generator(Number, Deviation, mode=mode, factory=XpressProblemFactory(), save=True,
                                     single_file=True, find_path=path)

    # if False:
    #
    #     var_list = ["C1", "C6", "C7", "C8", "C9"]
    #     Number = 10000
    #     Deviation = 0
    #     prob_list = ["problem_rte_2.lp"]
    #     cons_to_vary = None
    #     for elem in var_list:
    #         vars_to_vary = [elem]
    #         data = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary,
    #                                  Xpress_Problem_Factory(), save=True, single_file=True)
    #
    # if False:
    #     """Generates data for test in Analyse_data.py. """
    #
    #     prob_list = ["problem_rte_1.lp"]
    #     vars_to_vary = ["C1"]
    #     cons_to_vary = None
    #     Number = 100
    #     Deviation = 0
    #
    #     data1 = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary, Xpress_Problem_Factory(),
    #                               save=True)
    #
    #     data2 = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary, Xpress_Problem_Factory(),
    #                               save=True, name="the_same_again")

