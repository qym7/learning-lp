"""
The purpose of this code is to provide a user-friendly function
generating a dataset with N random new RHS from a list of chosen
linear optimization problems and their N associated solutions.
The name of this function is "problem_generator".

For more information see its description of problem_generator.
"""

import sys
import os
import numpy as np
from problem_selector import extract
from problem_interface import Problem, Problem_factory
from problem_cplex import Cplex_Problem_Factory
from Problem_xpress import Xpress_Problem_Factory
from dataset import dataset, load_csv


class lin_opt_pbs:
    """
    lin_opt_pbs is a class representing linear optimization problems.
    It will be used to generate new linear optimization problems
    by adding some noise (gaussian) to some chosen coefficients of a given problem.

    Attributes
    ----------
    problem : Problem instance (see problem_interface.py)
        a linear optimisation problem
    RHS_list : (int, float) list list
        a list of RHS in the format used by cplex.
        The first elements of the tuples represent indices of constraints, so they should
        all be different and never exceed the number of constraints of self.problem.
    generated_RHS : ((int, float) list, (int, float) list) list
        a list of newly created RHS in a particular format.
    dev : float
        giving the relative deviation of the noise when generating new problems.
    cons_to_vary : int list
        a list of indices of the constraints of the linear optimisation problems that should be affected
        by the noise when generating new problems.
    vars_to_vary : int list
        a list of indices of the variables of the linear optimisation problems that should be fixed randomly
        when generating new problems.
    """
    def __init__(self, problem: Problem, RHS_list, cons_to_vary=None, vars_to_vary=None):
        self.problem = problem
        self.RHS_list = RHS_list
        self.generated_RHS = []
        self.dev = 0
        self.cons_to_vary = cons_to_vary
        self.vars_to_vary = vars_to_vary

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

    def set_cons_to_vary(self, cons_to_vary):
        self.cons_to_vary = cons_to_vary

    def get_cons_to_vary(self):
        return self.cons_to_vary

    def set_generated_RHS(self, RHS_list):
        self.generated_RHS = RHS_list

    def add_generated_RHS(self, RHS, ind=None):
        if ind is None:
            self.generated_RHS.append(RHS)
        else:
            self.generated_RHS[ind] = RHS

    def clear_generated_RHS(self):
        self.generated_RHS = []

    def calculate_solution(self, RHS):
        """
        The method calculate_solutions determines the exact solution (objective value)
        of self.problem with the RHS given as an argument (if that linear optimisation
        problem is feasible).

        Arguments
        ---------
        RHS : [(int, float) list, (int, float) list]

        Return
        ------
        solution : float
            solution of linear optimisation problem (if feasible)
        is_feasible : bool
            states whether linear optimisation problem is feasible
        """
        self.problem.set_RHS(RHS[0])
        self.set_vars(RHS[1])
        self.problem.solve()
        solution = self.problem.get_objective_value()
        return solution, self.problem.is_feasible()

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
        nb_pb = len(self.generated_RHS)
        nb_con = len(self.generated_RHS[0][0])
        nb_vars = len(self.generated_RHS[0][1])
        new_rhs = nb_pb * [None]
        for i in range(nb_pb):
            constraints = self.generated_RHS[i][0]
            variables = self.generated_RHS[i][1]
            rhs = (nb_vars + nb_con) * [None]
            for j in range(nb_con):
                rhs[j] = constraints[j][1]
            if nb_vars > 0:
                for j in range(nb_vars):
                    rhs[j + nb_con] = variables[j][1]
            new_rhs[i] = rhs
        return new_rhs

    def generate_random_RHS(self, k):
        """
        Generates single random new problem.

        More precisely, the method generate_random_RHS generates a single new random problem
        by adding a gaussian noise to some coefficients of the RHS (= right hand side)
        of a chosen problem in an instance of lin_opt_pbs. The chosen coefficients are
        given by self.cons_to_vary.

        In addition, the method fixes some given variables of the linear optimisation
        problem randomly in a given interval. The chose variables are given by
        self.vars_to_vary.

        The standard deviation of that noise in each variable is computed by multiplying the
        value that variable takes by the factor dev.
        Thus the standard deviation is always chosen relative to the variable's value.

        Arguments
        ---------
        k : int
            giving the index of the chosen optimization problem in RHS_list

        Return
        ------
        RHS : [(int, float) list, (int, float) list]
            the generated RHS
        """
        rhs = self.RHS_list[k]
        nb = len(rhs)
        new_rhs = nb * [None]
        for i in range(nb):
            val = rhs[i][1]
            new_val = val + (np.random.normal(0, abs(val) * self.dev, 1))[0]  # add gaussian noise to the RHS
            new_rhs[i] = (rhs[i][0], new_val)
        values = self.choose_vars_random()
        return [new_rhs, values]

    def choose_vars_random(self):
        """
        Fixes the variables listed in self.vars_to_fix to random values inside
        their previous bounds.

        Return
        ------
        values : (int, float) list
            the indices of the fixed variables and their new values.
        """
        if self.vars_to_vary is None:
            return[]
        else:
            nb = len(self.vars_to_vary)
            values = nb * [None]
            for i in range(nb):
                lw_bnd, up_bnd = self.problem.var_get_bounds(self.vars_to_vary[i])
                val = np.random.uniform(lw_bnd, up_bnd)
                values[i] = (self.vars_to_vary[i], val)
            return values

    def set_vars(self, values):
        """
        Sets the vars to the given values.

        Arguments
        ---------
        values : (int, float) list
            a list of tuples (int, float) where the int represents the index of a
            variable of the linear optimisation problem and the float the value it
            should be set to.
        """
        nb = len(values)
        for i in range(nb):
            self.problem.var_set_bounds(values[i][0], values[i][1], values[i][1])

    def generate_and_solve(self, N):
        """
        Auxiliary function to problem_generator (see problem_generator.problem_generator).

        Generates a random RHS based on a randomly chosen RHS in self.RHS_list, solves
        self.problem with the newly generated RHS and repeats that process until the
        obtained linear optimisation problem is feasible. If the generated problem is feasible,
        the generated RHS is stocked in self.generated_RHS and the solution value is stocked
        in a list.

        Arguments
        ---------
        N : int
            number of problems to be generated

        Returns
        -------
        sol_list : float list
            list of solutions associated to the newly generated RHS
        """
        K = len(self.RHS_list)
        sol_list = N * [None]
        self.set_generated_RHS(N * [None])

        counter = 1
        for i in range(N):
            if i == counter:
                print(i)
                counter = counter + 100
            is_feasible = False
            ind = np.random.randint(K)
            stopper = 0
            while is_feasible is False:
                rhs = self.generate_random_RHS(ind)
                solution, is_feasible = self.calculate_solution(rhs)
                if stopper == 100:
                    raise Exception("Problem {} seems to be infeasible or dev is way too high".format(i))
                else:
                    stopper += 1
            self.add_generated_RHS(rhs, i)
            sol_list[i] = solution

        return sol_list


def give_name(N, dev, cons_to_vary=None, vars_to_vary=None):
    """
    Creates a file name for a set of generated problems.

    The file name contains som practical information (number of generated problems,
    deviation used while generation, names of varied constraints and variables).

    Arguments
    ---------
    N : int
        number of generated problems
    dev : float
        deviation used while generation
    cons_to_vary : string list
        list of names of the constraints that varied while generation
    vars_to_vary : string list
        list of names of the variables that were fixed randomly while generation
    """
    name = "Nb=" + str(N) + "_dev=" + str(dev)
    if cons_to_vary is not None:
        for elem in cons_to_vary:
            name = name + "_" + elem
    if vars_to_vary is not None:
        for elem in vars_to_vary:
            name = name + "_" + elem
    return name


def problem_generator(prob_list, N, dev, cons_to_vary, vars_to_vary, factory: Problem_factory = Cplex_Problem_Factory(),
                      save=False, single_file=False, path=None, name=None):
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
    cons_to_vary : string list
        a list of names of the constraints of the linear optimisation problems that should be affected when
        generating new problems. (All linear optimisation problems in prob_list should have an equal
        amount of constraints with equal names and in the same order.)
    vars_to_vary : string list
        a list of names of the variables of the linear optimisation problems that should be fixed randomly
        when generating new problems. (All linear optimisation problems in prob_list should have an equal
        amount of variables with equal names and in the same order.)
    factory : Problem_factory instance (see in problem_interface.py)
    save : bool
        set True if generated problems should be saved in csv file (see dataset.to_csv)
    single_file : bool
        states whether self.RHS and self.solutions are saved in a single file
        or two separate files (see dataset.to_csv)
    path :
        indicates where csv should be saved if save is True
    name : str
        gives name of file where data is stocked if save is True. If None, name is generated automatically.

    Return
    ------
    data : dataset instance
        containing the N generated RHS and their N associated solutions
    """
    cont = extract(prob_list, cons_to_vary, vars_to_vary, factory)
    prob_root = lin_opt_pbs(cont[0], cont[1], cont[2], cont[3])
    prob_root.set_deviation(dev)

    sol_list = prob_root.generate_and_solve(N)
    rhs_list = prob_root.extract_RHS()
    data = dataset(rhs_list, sol_list)

    if save is True:
        if name is None:
            name = give_name(N, dev, cons_to_vary, vars_to_vary)
        new_path = os.path.join("." if path is None else path, "Generated_problems", prob_list[0])
        data.to_csv(name, new_path, single_file)

    return data


def problem_generator_y(prob_list, N, dev, cons_to_vary, vars_to_vary, factory: Problem_factory = Cplex_Problem_Factory()):
    """
    The function problem_generator_y is an adapted version of problem_generator
    which can be used as callback function while training a neural network
    in order to generate new training data at the beginning of each epoch.

    For a more detailed description, see problem_generator.
    """
    cont = extract(prob_list, cons_to_vary, vars_to_vary, factory)
    prob_root = lin_opt_pbs(cont[0], cont[1], cont[2], cont[3])
    prob_root.set_deviation(dev)

    while True:
        prob_root.clear_generated_RHS()
        sol_list = prob_root.generate_and_solve(N)
        rhs_list = prob_root.extract_RHS()
        data = dataset(rhs_list, sol_list)

        yield data


