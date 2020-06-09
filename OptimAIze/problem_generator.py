# The purpose of this code is to provide a user-friendly function
# generating a dataset
# with N random new RHS from a list of chosen linear optimization problems
# and their N associated solutions.
# The name of this function is "problem_generator"
# For more information see its description at the end of the code

import numpy as np
from optimAIze.dataset import dataset
from optimAIze.problem_selector import Selector, extract

# lin_opt_pbs is a class representing linear optimization problems.
# It will be used to generate new linear optimization problems
# by adding some noise (gaussian) to some chosen coefficients of a given problem.

# Parameters of an instance of lin_opt_pbs :
#           problem: a cplex.Cplex instance
#           RHS_list: a list of RHS in the format used by cplex (list of lists of tuples (int, float)).
#                     The first elements of the tuples represent indices of constraints, so they should
#                     all be different and never exceed the number of constraints of self.problem
#           generated_problems: a list of newly created RHS in the format used by cplex
#           dev : a float setting the relative deviation of the variables when generating new problems
#           non_fixed_vars : a list containing all indices of the variables which will be affected by the noise
#               when generating new problems. If not given by the user, it is determined by the program.

class lin_opt_pbs:
    def __init__(self, problem, RHS_list, non_fixed_vars):
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

    # The method calculate_solutions determines the exact solutions
    # of the problems in an instance of lin_opt_pbs
    # and returns them in a list

    # Arguments taken: a lin_opt_pbs instance
    # Output: a list of solutions (float list)

    def calculate_solutions(self):
        new_list = []
        nb_pb = len(self.generated_problems)
        for pb in range(nb_pb):
            if pb > 0 and pb % 100 == 0:
                print(pb)
            RHS = self.generated_problems[pb]
            self.problem.linear_constraints.set_rhs(RHS)
            self.problem.solve()
            new_list.append(self.problem.solution.get_objective_value())
        return new_list

    # The method extract_RHS transforms a list of RHS
    # in the format used by cplex into a simple list of
    # float lists

    # Arguments taken: a lin_opt_pbs instance
    # Output: a list of RHS (i.e. a list of list)

    def extract_RHS(self):
        new_list = []
        nb_pb = len(self.generated_problems)
        for i in range(nb_pb):
            constraints = self.generated_problems[i]
            rhs = []
            for elem in constraints:
                rhs.append(elem[1])
            new_list.append(rhs)
        return new_list

    # The method generate_random_prob generates a single new random problem
    # by adding a gaussian noise to some coefficients of the RHS (= right hand side)
    # of a chosen problem in an instance of lin_opt_pbs. The chosen coefficients are
    # given by self.non_fixed_vars

    # The standard deviation of that noise in each variable is computed by multiplying the
    # value that variable takes by the factor dev.
    # Thus the standard deviation is always chosen relative to the variable's value.

    # Arguments taken: an int k giving the index of the chosen optimization problem in RHS_list
    # Output: None (the new RHS has been added to self.RHS_list)

    def generate_random_prob(self, k):
        rhs = self.RHS_list[k]
        new_list = []
        for ind in self.non_fixed_vars:
            val = rhs[ind][1]
            new_val = val + (np.random.normal(0, abs(val) * self.dev, 1))[0]  # add gaussian noise to the RHS
            new_list.append((ind, new_val))
        self.generated_problems.append(new_list)

# The function problem_generator generates an instance of dataset
# with N random RHS based on a chosen linear optimization problem
# and their N associated solutions
# The RHS are truncated : only the non fixed coefficients are kept

# Parameters of problem generator :
#           problem : a cplex.Cplex linear optimization problem
#           RHS_list: a list of RHS in the right format (list of lists of tuples (int, float))
#           N : an int giving the number of RHS to generate
#           dev : a float setting the relative deviation of the variables when generating new problems
#           non_fixed_vars : a list containing all variables which will be affected by the noise
#               when generating new problems. If not given, calculated by the program.
# Output:  a dataset instance containing N RHS and their N associated solutions

def problem_generator(prob_list, N, dev):
    cont = extract(prob_list)
    prob_root = lin_opt_pbs(cont[0], cont[1], cont[2])
    prob_root.set_deviation(dev)
    K = len(prob_root.RHS_list)
    prob_root.problem.set_log_stream(None)
    prob_root.problem.set_error_stream(None)
    prob_root.problem.set_warning_stream(None)
    prob_root.problem.set_results_stream(None)

    for i in range(N):
        ind = np.random.randint(K)
        prob_root.generate_random_prob(ind)

    rhs_list = prob_root.extract_RHS()
    sol_list = prob_root.calculate_solutions()
    data = dataset(rhs_list, sol_list)  # write either dataset or dataset.dataset to create a new instance
    return data

# The function problem_generator_y is an adapted version of problem_generator
# which can be used as callback function while training a neural network
# in order to generate new training data at the beginning of each epoch

def problem_generator_y(prob_list, N, dev):
    cont = extract(prob_list)
    prob_root = lin_opt_pbs(cont[0], cont[1], cont[2])
    prob_root.set_deviation(dev)
    K = len(prob_root.RHS_list)
    prob_root.problem.set_log_stream(None)
    prob_root.problem.set_error_stream(None)
    prob_root.problem.set_warning_stream(None)
    prob_root.problem.set_results_stream(None)

    while True:
        prob_root.clear_generated_problems()

        for i in range(N):
            ind = np.random.randint(K)
            prob_root.generate_random_prob(ind)

        rhs_list = prob_root.extract_RHS()
        sol_list = prob_root.calculate_solutions()
        data = dataset(rhs_list, sol_list)
        yield data.get_RHS(), data.get_solutions()


# Testing the problem_generator function

problem_file = 'petit_probleme.lp'
N = 100000
dev = 0.1

data = problem_generator(problem_file, N, dev)
print(data.get_RHS())
print(data.get_solutions())



