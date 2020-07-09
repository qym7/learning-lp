import pickle
import numpy as np
from problem_interface import Problem, Problem_factory
from problem_cplex import Cplex_Problem_Factory
from Problem_xpress import Xpress_Problem_Factory


class Selector:
    """
    The class Selector is designed to provide a user-friendly interface for selecting files
    containing linear optimization problems. Furthermore it reads the content of the selected
    files and converts it to the format required by the class lin_opt_pbs in problem_generator.py

    Attributes
    ----------
    prob_list : string list
        a list of linear optimization problems. Should be a list of file-names (string list).
    content :
        the content of prob_list in the format required by the class lin_opt_pbs in problem_generator.py.
    cons_to_vary : string list
        a list of names of the constraints of the linear optimisation problems that should be affected when
        generating new problems. (All linear optimisation problems in prob_list should have an equal
        amount of constraints with equal names and in the same order.)
    vars_to_vary : string list
        a list of names of the variables of the linear optimisation problems that should be fixed randomly
        when generating new problems. (All linear optimisation problems in prob_list should have an equal
        amount of variables with equal names and in the same order.)
    """
    def __init__(self, prob_list, cons_to_vary=None, vars_to_vary=None,
                 factory: Problem_factory = Cplex_Problem_Factory(), determine_cons_to_vary = False):
        self.prob_list = prob_list
        self.factory = factory
        self.content = None
        self.cons_to_vary = cons_to_vary
        self.vars_to_vary = vars_to_vary
        self.determine_cons_to_vary = determine_cons_to_vary

    def get_prob_list(self):
        return self.prob_list

    def set_prob_list(self, prob_list):
        self.prob_list = prob_list

    def get_content(self):
        return self.content

    def set_content(self, problem, RHS_list, cons_to_vary, vars_to_vary, name_list):
        self.content = problem, RHS_list, cons_to_vary, vars_to_vary, name_list

    def give_names(self):
        n = len(self.prob_list)
        name_list = n * [None]
        for i in range(n):
            name_list[i] = ("problem_(%d)", i)
        return name_list

    def fill_content(self):
        """
        The method fill_content converts the content of the linear optimization problems in
        self.prob_list to the format required by the class lin_opt_pbs ans saves it in self.content.

        More precisely the method checks first of all if the content of prob_list is a string list.
        If that is not the case, an error message will occur. If the content of prob_list is a string
        list, but cannot be read by the method (if prob_list contains a list of file names, but some
        of those files do not contain linear optimization problems for example) an exception will be raised.

        If no exception occurs, the content of prob_list will be converted into a tuple
        containing:
           - one complete linear optimization problem (see class Problem in problem_interface.py),
           - the list of RHS of all problems given in the format required by
             Cplex (list of list of tuples (int, float)),
           - the indices of the coefficients of the RHS that should be affected
             when generating new problems (int list)
           - the indices of the variables that should be fixed randomly while
             when generating new problems
           - the list of names of the problems saved in prob_list (string list)

        This tuple is saved in self.content

        Arguments
        ---------
        No arguments

        Returns
        --------
        No output
        """
        assert isinstance(self.prob_list[0], str), "Files do not contain linear optimisation problems."
        try:
            problem = self.factory.read_problem_from_file(self.prob_list[0])
            if self.cons_to_vary is None and self.determine_cons_to_vary:
                cons_to_vary = self.calculate_cons_to_vary(problem)
            elif self.cons_to_vary is not None:
                cons_to_vary = self.read_cons_to_vary(problem)
            else:
                cons_to_vary = self.cons_to_vary
            vars_to_vary = self.read_vars_to_vary(problem)
            nb = len(self.prob_list)
            RHS_list = nb * [None]
            name_list = self.prob_list
            for i in range(nb):
                problem.read(self.prob_list[i])
                constraints = problem.get_RHS(cons_to_vary)
                rhs = format_RHS(constraints, cons_to_vary)
                RHS_list[i] = rhs
        except:
            raise Exception("Files do not contain linear optimisation problems.")

        self.set_content(problem, RHS_list, cons_to_vary, vars_to_vary, name_list)

    def calculate_cons_to_vary(self, problem):
        nb = len(self.prob_list)
        if nb == 1:
            return None
        problem.read(self.prob_list[0])
        rhs = problem.get_RHS(all_cons=True)
        nb_constraints = len(rhs)
        constraints_max = rhs
        constraints_min = rhs
        for j in range(1, nb):
            problem.read(self.prob_list[j])
            rhs = problem.get_RHS(all_cons=True)
            for i in range(nb_constraints):
                if rhs[i] > constraints_max[i]:
                    constraints_max[i] = rhs[i]
                if rhs[i] < constraints_min[i]:
                    constraints_min[i] = rhs[i]
        constraints_max = np.array(constraints_max)
        constraints_min = np.array(constraints_min)
        constraints_range = constraints_max - constraints_min
        cons_to_vary = []
        for i in range(len(constraints_range)):
            if constraints_range[i] != 0:
                cons_to_vary.append(i)
        return cons_to_vary

    def read_cons_to_vary(self, problem):
        cons_names = problem.get_constraint_names()
        nb_cons = len(cons_names)

        if self.cons_to_vary is None:
            return None
        else:
            nb = len(self.cons_to_vary)
            counter = 0
            cons_to_vary = nb * [None]
            for i in range(nb_cons):
                if counter < nb and cons_names[i] == self.cons_to_vary[counter]:
                    cons_to_vary[counter] = i
                    counter += 1
            if cons_to_vary[-1] is None:
                raise Exception("Either the optimisation problems have no constraints with some of that names or "
                                "the constraint names were given in the wrong order.")
        return cons_to_vary

    def read_vars_to_vary(self, problem):
        var_names = problem.get_variable_names()
        nb_vars = len(var_names)
        if self.vars_to_vary is None:
            return None
        else:
            nb = len(self.vars_to_vary)
            counter = 0
            vars_to_vary = nb * [None]
            for i in range(nb_vars):
                if counter < nb and var_names[i] == self.vars_to_vary[counter]:
                    vars_to_vary[counter] = i
                    counter += 1
            if vars_to_vary[-1] is None:
                raise Exception("Either the optimisation problems have no variables with some of that names or "
                                "the variable names were given in the wrong order.")
        return vars_to_vary


def extract(prob_list, cons_to_vary=None, vars_to_vary=None, factory: Problem_factory = Cplex_Problem_Factory(),
            determine_cons_to_vary=False):
    """
    The function extract converts the content of a list of problems given as an argument to
    the format required by the class lin_opt_pbs in problem_generator.py and returns the result.

    (See description of fill_content, method of Selector)

    Arguments
    ---------
    prob_list : string list
        a list of linear optimization problems. Should be a list of file-names.
    cons_to_vary : string list
        a list of names of the constraints of the linear optimisation problems that should be affected when
        generating new problems. (All linear optimisation problems in prob_list should have an equal
        amount of constraints with equal names and in the same order.)
    vars_to_vary : string list
        a list of names of the variables of the linear optimisation problems that should be fixed randomly
        when generating new problems. (All linear optimisation problems in prob_list should have an equal
        amount of variables with equal names and in the same order.)
    factory : Problem_factory instance (see in problem_interface.py)
    determine_cons_to_vary : bool
        if True cons_to_vary are determined automatically

    returns
    -------
    content :
        the content of prob_list in the format required by the class lin_opt_pbs in problem_generator.py.
    """
    selector = Selector(prob_list, cons_to_vary, vars_to_vary, factory, determine_cons_to_vary=determine_cons_to_vary)
    selector.fill_content()
    content = selector.get_content()
    return content


def format_RHS(constraints, cons_to_vary):
    """Creating an RHS in the format required by the class lin_opt_pbs out of a list of constraints."""
    if cons_to_vary is None:
        return []
    else:
        nb = len(cons_to_vary)
        rhs = nb * [None]
        for i in range(nb):
            rhs[i] = (cons_to_vary[i], constraints[i])
        return rhs


if __name__ == '__main__':
    # Testing the extract function on an lp-file

    problem_file = 'petit_probleme.lp'
    Content = extract(problem_file)
    print(Content)
