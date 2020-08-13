"""
Module generates a certain number of linear optimisation problems randomly out of
a single problem or a list of problems.

It contains the class lin_opt_pbs which implements all methods used for generation.
When a lin_opt_pbs instance is created, an additional GenerationMode instance must be provided
to specify the form of the base problems and what methods should be used to generate new
problems out of them.

Please read the description of the module GenerationMode.py to get a detailed description
of how the new problems are generated depending on the generation mode you choose to use.
"""

import sys
import os
import numpy as np
from problem_selector import extract
from ProblemTypeInterface import ProblemFactory
from Problem import Problem
from dataset import dataset, load_csv
from GenerationMode import GenerationModeClassic, GenerationMode, \
    GenerationModeMasterSlaveContinuous, GenerationModeMasterSlaveDiscreet
from MathFunctions import get_weights_for_convex_comb, convex_comb
import random
from time import time


class lin_opt_pbs:
    """
    Implements methods to generate new linear optimisation problems out of a single or
    a list of given problems. For further explanations read description of this module and
    the module GenerationMode.py.

    Attributes
    ----------
    generation_mode : GenerationMode instance
        see GenerationMode.py for a detailed description of the different modes available
    problem : Problem instance (see Problem.py)
        a linear optimisation problem
    master : Problem instance (see Problem.py)
        None if generation_mode is not GenerationModeMasterSlaveDiscreet or GenerationModeMasterSlaveContinuous
    RHS_list : (int, float) list list
        a list of RHS in the format used by most lp solvers.
        The first elements of the tuples represent indices of constraints, so they should
        all be different and never exceed the number of constraints of self.problem.
    generated_RHS : ((int, float) list, (int, float) list) list
        a list of newly created RHS in a particular format.
    dev : float
        giving the RELATIVE deviation of the noise when generating new problems with some generation modes.
    cons_to_vary : int list
        a list of indices of the constraints of self.problem that should vary from one generated
        problem to another.
    vars_to_vary : int list
        a list of indices of the variables of the linear optimisation problems that should be fixed randomly
        when generating new problems.
    """
    def __init__(self, problem: Problem, master: Problem, RHS_list, cons_to_vary=None, vars_to_vary=None,
                 mode: GenerationMode = None):
        self.problem = problem
        self.master = master
        self.RHS_list = RHS_list
        self.generated_RHS = []
        self.dev = 0
        self.cons_to_vary = cons_to_vary
        self.vars_to_vary = vars_to_vary
        self.generation_mode = mode

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
        by copying one RHS (= right hand side) in self.RHS_list and modifying some of its coefficients.
        The chosen coefficients are given by self.cons_to_vary.

        In addition, the method fixes some given variables of the linear optimisation
        problem randomly such that they verify the problem's constraints. The chosen variables are given by
        self.vars_to_vary.

        The way in which the constraints and variables are varied is determined by self.generation_mode.

        To set the variables, self.generation_mode will choose between choose_vars_random_trivial,
        choose_vars_random_interior_point_method, choose_vars_random_vertices_method and
        choose_vars_random_convex_comb.

        To modify the constraints, self.generation_mode will choose between choose_constraints_random_discreet,
        choose_constraints_random_perturbation and choose_constraints_random_continuous.


        Arguments
        ---------
        k : int
            giving the index of the chosen optimization problem in RHS_list

        Return
        ------
        RHS : [(int, float) list, (int, float) list]
            the generated RHS
        """
        new_rhs = self.generation_mode.choose_constraints_random(self, k)
        values = self.generation_mode.choose_vars_random(self)
        return [new_rhs, values]

    def choose_vars_random_trivial(self):
        """
        Fixes the variables listed in self.vars_to_fix to random values inside
        their bounds. Method should only be used if the variables to vary appear
        in trivial constraints only.

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

    def choose_vars_random_interior_point_method(self):
        """
        Implements the inner point method described at the beginning of GenerationMode.py.

        Method should be used when the base problem is not simple, which means that it is so large
        that no exhaustive list of its domain's border can be given. If the problem is simple
        use choose_vars_random_convex_comb instead.

        Return
        ------
        values : (int, float) list
            the indices of the fixed variables and their new values.
        """
        matrix, rhs = self.master.get_matrix()
        bounds = self.master.get_domain_border(all_vars=True, modules=self.generation_mode.modules)
        inner_points = self.master.get_inner_points()
        max_dimension = self.master.get_max_dimension()

        if self.vars_to_vary is None:
            return []
        else:
            is_feasible = False
            nb = self.master.get_number_vars()
            values = np.array(nb * [None])
            variables = np.array(nb * [None])
            rand = np.random.randint(nb)
            inner_point = inner_points[rand]
            for i in range(nb):
                val = np.random.uniform(bounds[i][0], bounds[i][1])
                values[i] = val
            if np.less_equal(np.dot(matrix, values), rhs).all():
                is_feasible = True
            initial_values = values.copy()
            exp = 1
            pond = 0
            counter = 0
            while not is_feasible:
                counter += 1
                if counter == 100:
                    print("There seems to be a point in the border of the domain in self.inner_points.")
                #bool = np.less_equal(np.dot(matrix, values), rhs)
                #dot = np.dot(matrix, values)
                values = inner_point * (1 - 1 / np.exp2(exp)) + initial_values / np.exp2(exp)
                if np.less_equal(np.dot(matrix, values), rhs).all():
                    is_feasible = True
                    exp += 1
                else:
                    exp += 1
            last_verified = values.copy()
            for i in range(20 + int(np.log10(max_dimension))):
                if np.less_equal(np.dot(matrix, values), rhs).all():
                    last_verified = values
                    pond = pond + 1 / np.exp2(exp)
                    exp += 1
                    values = inner_point * (1 - pond) + initial_values * pond
                else:
                    pond = pond - 1 / np.exp2(exp)
                    exp += 1
                    values = inner_point * (1 - pond) + initial_values * pond
            factor = random.random()
            values = last_verified * factor + (1 - factor) * inner_point
            assert np.less_equal(np.dot(matrix, values), rhs).all(), "something went wrong..."
            for i in range(nb):
                variables[i] = (self.vars_to_vary[i], values[self.vars_to_vary[i]])
            return variables

    def choose_vars_random_vertices_method(self):
        """
        Implements the random vertices method described at the beginning of GenerationMode.py.

        Method should be used when the base problem is not simple, which means that it is so large
        that no exhaustive list of its domain's border can be given. If the problem is simple
        use choose_vars_random_convex_comb instead.

        Return
        ------
        values : (int, float) list
           the indices of the fixed variables and their new values.
        """
        if self.vars_to_vary is None:
            return []
        else:
            vertices = self.master.get_some_vertices()
            nb_ver = len(vertices)
            nb_vars = len(self.vars_to_vary)
            nb_conv_comb = nb_vars * 2

            chosen_nb = max(np.random.randint(nb_conv_comb), 2)
            chosen_ind = random.sample(range(nb_ver), chosen_nb)
            weights = get_weights_for_convex_comb(chosen_nb)

            comb_ver = chosen_nb * [None]

            for i in range(chosen_nb):
                comb_ver[i] = vertices[chosen_ind[i]]

            var_values = convex_comb(weights, comb_ver)
            values = nb_vars * [None]

            for i in range(nb_vars):
                values[i] = (self.vars_to_vary[i], var_values[self.vars_to_vary[i]])

            return values

    def choose_vars_random_convex_comb(self):
        """
        Implements the convex combination method used for simple problems,
        described at the beginning of GenerationMode.py.

        Fixes the variables listed in self.vars_to_fix to random values inside
        the domain defined by matrix * X <= rhs, where matrix and rhs describe the constraints
        of self.master.

        Variables are set to random values by convex combinations of vectors
        in self.master.domain.domain_vertices.

        Return
        ------
        values : (int, float) list
            the indices of the fixed variables and their new values.
        """
        vertices = self.generation_mode.vertices
        if vertices is None:
            vertices = self.master.get_domain_border()

        if vertices is None:
            return []
        else:
            nb = len(vertices)

            weights = get_weights_for_convex_comb(nb)

            var_values = convex_comb(weights, vertices)

            nb_vars = len(self.vars_to_vary)
            values = nb_vars * [None]

            for i in range(nb_vars):
                values[i] = (self.vars_to_vary[i], var_values[self.vars_to_vary[i]])

            return values

    def choose_constraints_random_discreet(self):
        """
        Method used by GenerationModeMasterSlaveDiscreet to randomly generate the RHS
        of a single new problem and returns that RHS. (See documentation of GenerationMode.py
        for more detailed information).

        Return
        ------
        rhs : (int, float) list
           the indices of the fixed constraints and their new values.
        """
        if self.cons_to_vary is None:
            return []
        else:
            nb_cons = len(self.cons_to_vary)
            new_rhs = nb_cons * [None]

            for i in range(nb_cons):
                weights = self.RHS_list[i][1]
                values = self.RHS_list[i][0]
                val = random.choices(values, weights=weights, k=1)
                new_rhs[i] = (self.cons_to_vary[i], val[0])

            return new_rhs

    def choose_constraints_random_perturbation(self, k):
        """
        Method used by GenerationModeClassic to randomly generate the RHS
        of a single new problem and returns that RHS. (See documentation of GenerationMode.py
        for more detailed information).

        Return
        ------
        rhs : (int, float) list
           the indices of the fixed constraints and their new values.
        """
        if self.cons_to_vary is None:
            return []
        else:
            rhs = self.RHS_list[k]
            nb = len(rhs)
            new_rhs = nb * [None]
            for i in range(nb):
                val = rhs[i][1]
                new_val = val + (np.random.normal(0, abs(val) * self.dev, 1))[0]  # add gaussian noise to the RHS
                new_rhs[i] = (rhs[i][0], new_val)
        return new_rhs

    def choose_constraints_random_continuous(self):
        """
        Method used by GenerationModeMasterSlaveContinuous to randomly generate the RHS
        of a single new problem and returns that RHS. (See documentation of GenerationMode.py
        for more detailed information).

        Return
        ------
        rhs : (int, float) list
           the indices of the fixed constraints and their new values.
        """
        if self.cons_to_vary is None:
            return []
        else:
            nb_cons = len(self.cons_to_vary)
            new_rhs = nb_cons * [None]

            for i in range(nb_cons):
                values = self.RHS_list[i][0]
                m = np.min(values)
                M = np.max(values)
                ran = M - m
                val = random.uniform(m, M)
                new_val = val + (np.random.normal(0, ran * self.dev, 1))[0]
                new_rhs[i] = (self.cons_to_vary[i], new_val)

            return new_rhs

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

    def compute_random_vertices_of_master(self, nb):
        """
        Computes nb random vertices of self.master and stocks them in
        self.master.domain.some_vertices. Some vertices can possibly occur more
        then once.
        """
        vertices = nb * [None]

        print("Start computing vertices:")

        begin = time()
        for i in range(nb):
            present = time()
            if (present - begin) > 60:
                print(i)
                begin = time()
            vertices[i] = self.master.compute_random_vertex()

        self.master.set_some_vertices(vertices)

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
        self.generation_mode.prepare_vertices(self, N)

        K = len(self.RHS_list)
        sol_list = N * [None]
        self.set_generated_RHS(N * [None])

        print("Start generating problems:")

        begin = time()
        for i in range(N):
            present = time()
            if (present - begin) > 60:
                print(i)
                begin = time()
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


def problem_generator(N, dev, mode: GenerationMode, factory: ProblemFactory, save=False, single_file=False,
                      find_path=None, save_path=None, name=None):
    """
    The function problem_generator generates an instance of dataset
    with N randomly generated problems and their N associated solutions. Those problems
    are generated out of a single problem or a list of problems stocked in mode.

    Actually, the dataset instance does not really contain complete problems, but only their RHS, which are
    truncated to the constraints that vary from one problem to another, and the values the variables to vary have
    taken. That is the only information needed to train our neural networks.

    Arguments
    ---------
    N : int
        giving the number of new problems to generate
    dev : float
        giving the relative deviation of the gaussian noise applied to the constraints
        to vary when generating new problems
    mode : GenerationMode instance
        see GenerationMode.py for a detailed description of the different modes available
    factory : Problem_factory instance
        choose the problem factory that matches the lp solver you are using
    save : bool
        set True if generated problems should be saved in csv file (see dataset.to_csv)
    single_file : bool
        states whether self.RHS and self.solutions are saved in a single file
        or two separate files (see dataset.to_csv)
    find_path : str
        indicates where problems for generation are saved
    save_path : str
        indicates where csv should be saved if save is True
    name : str
        gives name of file where data is stocked if save is True. If None, name is generated automatically.


    Return
    ------
    data : dataset instance
        containing the N generated RHS and their N associated solutions
    """
    cont = extract(mode, factory, path=find_path)
    prob_root = lin_opt_pbs(cont[0], cont[1], cont[2], cont[3], cont[4], mode=mode)
    prob_root.set_deviation(dev)

    input_names = cont[5] + cont[6]
    sol_list = prob_root.generate_and_solve(N)
    rhs_list = prob_root.extract_RHS()
    data = dataset(rhs_list, sol_list, input_names=input_names)

    if save is True:
        if name is None:
            name = give_name(N, dev)
        new_path = os.path.join("." if save_path is None else save_path, "Generated_problems", cont[7][0])
        data.to_csv(name, new_path, single_file)

    return data


def problem_generator_y(N, dev, mode: GenerationMode, factory: ProblemFactory, path=None):
    """
    The function problem_generator_y is an adapted version of problem_generator
    which can be used as callback function while training a neural network
    in order to generate new training data at the beginning of each epoch.

    For a more detailed description, see problem_generator.
    """
    cont = extract(mode, factory, path=path)
    prob_root = lin_opt_pbs(cont[0], cont[1], cont[2], cont[3], cont[4], mode=mode)
    prob_root.set_deviation(dev)

    while True:
        prob_root.clear_generated_RHS()
        sol_list = prob_root.generate_and_solve(N)
        rhs_list = prob_root.extract_RHS()
        data = dataset(rhs_list, sol_list)

        yield data
