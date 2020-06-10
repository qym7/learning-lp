"""
The purpose of this code is to provide a user-friendly function
generating a dataset with N random new RHS from a list of chosen
linear optimization problems and their N associated solutions.
The name of this function is "problem_generator".

For more information see its description of problem_generator.
"""

import numpy as np
from problem_selector import extract
from problem import Problem, Problem_factory, Cplex_Problem_Factory


class lin_opt_pbs:
    """
    lin_opt_pbs is a class representing linear optimization problems.
    It will be used to generate new linear optimization problems
    by adding some noise (gaussian) to some chosen coefficients of a given problem.

    Attributes
    ----------
    problem : cplex.Cplex
        a linear optimisation problem
    RHS_list : (int, float) list list
        a list of RHS in the format used by cplex.
        The first elements of the tuples represent indices of constraints, so they should
        all be different and never exceed the number of constraints of self.problem.
    generated_problems : (int, float) list list
        a list of newly created RHS in the format used by cplex.
    dev : float
        giving the relative deviation of the noise when generating new problems.
    non_fixed_vars : int list
        a list containing all indices of the variables which will be affected by the noise
        when generating new problems. If not given by the user, it is determined by the program.
    """
    def __init__(self, problem: Problem, RHS_list, non_fixed_vars):
        self.problem = problem
        self.RHS_list = RHS_list
        self.generated_problems = []
        self.dev = 0
        self.non_fixed_vars = non_fixed_vars

    def set_problem(self, problem):
        self.problem = problem

    def get_problem(self):
        return self.problem

    def set_RHS_list(self, RHS_list):
        self.RHS_list = RHS_list

    def get_RHS_list(self):
        return self.RHS_list

    def set_deviation(self, dev):
        self.dev = dev

    def get_deviation(self):
        return self.dev

    def set_non_fixed_vars(self, non_fixed_vars):
        self.non_fixed_vars = non_fixed_vars

    def get_non_fixed_vars(self):
        return self.non_fixed_vars

    def clear_generated_problems(self):
        self.generated_problems = []

    def calculate_solutions(self):
        """
        The method calculate_solutions determines the exact solutions (objective value)
        of the problems in an instance of lin_opt_pbs and returns them in a list.

        Arguments
        ---------
        No arguments

        Return
        ------
        a list of solutions : float list
        """
        nb_pb = len(self.generated_problems)
        new_list = nb_pb * [None]
        counter = 1
        for pb in range(nb_pb):
            if pb == counter:
                print(pb)
                counter = 2*counter
            RHS = self.generated_problems[pb]
            self.problem.set_RHS(RHS)
            self.problem.solve()
            new_list[pb] = self.problem.get_objective_value()
        return new_list

    def extract_RHS(self):
        """
        The method extract_RHS transforms a list of RHS in the
        format used by cplex into a simple list of float lists.

        Arguments
        ---------
        No arguments

        Return
        ------
        a list of RHS : float list list
        """
        nb_pb = len(self.generated_problems)
        nb_con = len(self.generated_problems[0])
        new_list = nb_pb * [None]
        for i in range(nb_pb):
            constraints = self.generated_problems[i]
            rhs = nb_con * [None]
            for j in range(nb_con):
                rhs[j] = constraints[j][1]
            new_list[i] = rhs
        return new_list

    def generate_random_prob(self, k):
        """
        Generates single random new problem.

        More precisely, the method generate_random_prob generates a single new random problem
        by adding a gaussian noise to some coefficients of the RHS (= right hand side)
        of a chosen problem in an instance of lin_opt_pbs. The chosen coefficients are
        given by self.non_fixed_vars

        The standard deviation of that noise in each variable is computed by multiplying the
        value that variable takes by the factor dev.
        Thus the standard deviation is always chosen relative to the variable's value.

        Arguments
        ---------
        k : int
            giving the index of the chosen optimization problem in RHS_list

        Return
        ------
        No return (the new RHS has been added to self.RHS_list)
        """
        rhs = self.RHS_list[k]
        nb = len(rhs)
        new_list = nb * [None]
        for i in range(nb):
            val = rhs[i][1]
            new_val = val + (np.random.normal(0, abs(val) * self.dev, 1))[0]  # add gaussian noise to the RHS
            new_list[i] = (rhs[i][0], new_val)
        self.generated_problems.append(new_list)


def problem_generator(prob_list, N, dev, factory: Problem_factory = Cplex_Problem_Factory()):
    """
    The function problem_generator generates an instance of dataset
    with N random RHS, based on a chosen linear optimization problem,
    and their N associated solutions

    The RHS are truncated : only the non fixed coefficients are kept

    Arguments
    ---------
    prob_list :
        a list of linear optimization problems. Should be either a single file name (string),
        a list of file-names (string list) or a list of objects of the class cplex.Cplex (cplex.Cplex list)
    N : int
        giving the number of new problems to generate
    dev : float
        giving the relative deviation of the noise when generating new problems

    Return
    ------
    data : dataset instance
        containing the N generated RHS and their N associated solutions
    """
    cont = extract(prob_list, factory)
    prob_root = lin_opt_pbs(cont[0], cont[1], cont[2])
    prob_root.set_deviation(dev)
    K = len(prob_root.RHS_list)

    for i in range(N):
        ind = np.random.randint(K)
        prob_root.generate_random_prob(ind)

    rhs_list = prob_root.extract_RHS()
    sol_list = prob_root.calculate_solutions()

    return rhs_list, sol_list


def problem_generator_y(prob_list, N, dev, factory: Problem_factory = Cplex_Problem_Factory()):
    """
    The function problem_generator_y is an adapted version of problem_generator
    which can be used as callback function while training a neural network
    in order to generate new training data at the beginning of each epoch.

    For a more detailed description, see problem_generator.
    """
    cont = extract(prob_list, factory)
    prob_root = lin_opt_pbs(cont[0], cont[1], cont[2])
    prob_root.set_deviation(dev)
    K = len(prob_root.RHS_list)

    while True:
        prob_root.clear_generated_problems()

        for i in range(N):
            ind = np.random.randint(K)
            prob_root.generate_random_prob(ind)

        rhs_list = prob_root.extract_RHS()
        sol_list = prob_root.calculate_solutions()

        yield rhs_list, sol_list


if __name__ == '__main__':
    # Testing the problem_generator function

    problem_file = 'petit_probleme.lp'
    N = 10000
    dev = 0.1

    rhs_List, sol_List = problem_generator(problem_file, N, dev)
    print(rhs_List)
    print(sol_List)

    # with open('petit_probleme.csv', 'w') as csv_file:
    #     for v in sol:
    #         csv_file.write('{};'.format(v))
    #     csv_file.write('\n')


