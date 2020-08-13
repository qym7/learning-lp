import pickle
import numpy as np
from ProblemTypeInterface import ProblemFactory
from GenerationMode import GenerationModeClassic, GenerationMode, \
    GenerationModeMasterSlaveContinuous, GenerationModeMasterSlaveDiscreet
import os


class SelectorContent:
    """
    Stocks data. Auxiliary class of Selector.

    For further information about stocked data, see documentation of Selector.fill_content.
    """
    def __init__(self):
        self.problem = None
        self.master = None
        self.RHS_list = None
        self.cons_to_vary = None
        self.vars_to_vary = None
        self.cons_names = None
        self.vars_names = None
        self.name_list = None

    def get_content(self):
        return self.problem, self.master, self.RHS_list, self.cons_to_vary, self.vars_to_vary, self.cons_names, \
               self.vars_names, self.name_list

    def fill(self, problem, master, RHS_list, cons_to_vary, vars_to_vary, cons_names, vars_names, name_list):
        self.problem = problem
        self.master = master
        self.RHS_list = RHS_list
        self.cons_to_vary = cons_to_vary
        self.vars_to_vary = vars_to_vary
        self.cons_names = cons_names
        self.vars_names = vars_names
        self.name_list = name_list


class Selector:
    """
    The class Selector is designed to provide a user-friendly interface for selecting files
    containing linear optimization problems. Furthermore it reads the content of the selected
    files and converts it to the data format required by the class lin_opt_pbs in problem_generator.py

    Attributes
    ----------
    prob_list : string list
        a list of names of files containing linear optimisation problems and possibly additional information.
        content depends on the chose generation mode (see GenerationMode.py for more detailed information).
    path : str
        path to files
    generation_mode : GenerationMode instance
        (see GenerationMode.py for a detailed description)
    factory : ProblemFactory instance
        choose the problem factory that matches the lp solver you are using
    content : SelectorContent instance
        the content of prob_list in the format required by the class lin_opt_pbs in problem_generator.py.
    """
    def __init__(self, mode: GenerationMode, factory: ProblemFactory, path=None):
        self.prob_list = mode.get_prob_list()
        self.path = path
        self.generation_mode = mode
        self.factory = factory
        self.content = SelectorContent()
        self.cons_names = None
        self.vars_names = None

    def get_prob_list(self):
        return self.prob_list

    def set_prob_list(self, prob_list):
        self.prob_list = prob_list

    def get_content(self):
        return self.content.get_content()

    def set_content(self, problem, master, RHS_list, cons_to_vary, vars_to_vary, cons_names, vars_names, name_list):
        self.content.fill(problem, master, RHS_list, cons_to_vary, vars_to_vary, cons_names, vars_names, name_list)

    def give_names(self):
        n = len(self.prob_list)
        name_list = n * [None]
        for i in range(n):
            name_list[i] = ("problem_(%d)", i)
        return name_list

    def fill_content(self):
        """
        The method fill_content converts the content of the linear optimization problems in
        self.prob_list to the data format required by the class lin_opt_pbs and saves it in self.content.

        More precisely the method checks first of all if the content of prob_list is a string list.
        If that is not the case, an error message will occur. If the content of prob_list is a string
        list, but cannot be read by the method (if prob_list contains a list of file names, but some
        of those files do not contain linear optimization problems for example) an exception will be raised.

        If no exception occurs, the content of prob_list will be stocked in SelectorContent instance.
        That SelectorContent should contain:
           - one complete linear optimization problem (see class Problem in Problem.py),
           - a master problem if GenerationMode is GenerationModeMasterSlaveDiscreet or
             GenerationModeMasterSlaveContinuous
           - the list of RHS of all problems given in the format required by
             the problem_generator.py module (list of list of tuples (int, float)),
           - the indices of the coefficients of the RHS that should be affected
             when generating new problems (int list)
           - the indices of the variables that should be fixed randomly
             when generating new problems
           - the names of those constraints in the linear optimisation problems
           - the names of those variables in the linear optimisation problems
           - the list of names of the problems saved in prob_list (string list)

        Arguments
        ---------
        No arguments

        Returns
        --------
        No output
        """
        assert isinstance(self.prob_list[0], str), "Files do not contain linear optimisation problems."
        try:
            problem = self.factory.read_problem_from_file(os.path.join("." if self.path is None else self.path,
                                                                       self.prob_list[0]))
            self.content.problem = problem
            self.generation_mode.generate_information(self, problem, self.path)
            self.content.name_list = self.prob_list
            self.content.cons_names = self.cons_names
            self.content.vars_names = self.vars_names
        except:
            raise Exception("Files do not contain linear optimisation problems.")

    def calculate_cons_to_vary(self, problem):
        """
        Determines cons_to_vary if not given manually if more then one problem was given
        as an input. Method compares the rhs of the given problems and memorizes only the
        constraints that don't have the same value in each rhs.

        Arguments
        ---------
        problem : Problem instance

        Returns
        -------
        cons_to_vary : int list
        """
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
        self.cons_names = problem.get_constraint_names(cons_to_vary)

        return cons_to_vary

    def calculate_vertices(self, problem):
        """
        Returns the vertices of the domain of a linear optimisation problem.
        """
        if self.generation_mode.vertices is None:
            return problem.compute_vertices_poly()

    def read_cons_to_vary(self, problem, cons_to_vary_names):
        """
        Returns the indices of the constraints of a given problem whose names match those
        in a given list of constraint names.

        Raises an exception when some names are not found.

        Arguments
        ---------
        problem : Problem instance
        cons_to_vary_names : str list

        Returns
        -------
        vars_to_vary : int list
        """
        self.cons_names = cons_to_vary_names

        cons_names = problem.get_constraint_names()
        nb_cons = len(cons_names)

        if cons_to_vary_names is None:
            return None
        else:
            nb = len(cons_to_vary_names)
            counter = 0
            cons_to_vary = nb * [None]
            for i in range(nb_cons):
                if counter < nb and cons_names[i] == cons_to_vary_names[counter]:
                    cons_to_vary[counter] = i
                    counter += 1
            if cons_to_vary[-1] is None:
                raise Exception("Either the optimisation problems have no constraints with some of that names or "
                                "the constraint names were given in the wrong order.")
        return cons_to_vary

    def read_vars_to_vary(self, problem, vars_to_vary_names):
        """
        Returns the indices of the variables of a given problem whose names match those
        in a given list of variable names.

        Raises an exception when some names are not found.

        Arguments
        ---------
        problem : Problem instance
        vars_to_vary_names : str list

        Returns
        -------
        vars_to_vary : int list
        """
        self.vars_names = vars_to_vary_names

        var_names = problem.get_variable_names()
        nb_vars = len(var_names)
        if vars_to_vary_names is None:
            return None
        else:
            nb = len(vars_to_vary_names)
            counter = 0
            vars_to_vary = nb * [None]
            for i in range(nb_vars):
                if counter < nb and var_names[i] == vars_to_vary_names[counter]:
                    vars_to_vary[counter] = i
                    counter += 1
            if vars_to_vary[-1] is None:
                raise Exception("Either the optimisation problems have no variables with some of that names or "
                                "the variable names were given in the wrong order.")
        return vars_to_vary

    def generate_information_classic(self, problem, path=None):
        """
        Auxiliary method of fill_content.

        Is called when self.generation_mode is GenerationModeClassic.
        (See documentation of GenerationMode.py for more detailed information).
        """
        cons_to_vary = self.generation_mode.cons_to_vary
        if cons_to_vary is None and self.generation_mode.determine_cons_to_vary:
            cons_to_vary = self.calculate_cons_to_vary(problem)
        elif cons_to_vary is not None:
            cons_to_vary = self.read_cons_to_vary(problem, cons_to_vary)

        nb = len(self.prob_list)
        RHS_list = nb * [None]

        for i in range(nb):
            name = self.prob_list[i]
            if path is not None:
                name = os.path.join(path, name)
            problem.read(name)
            constraints = problem.get_RHS(cons_to_vary)
            rhs = format_RHS(constraints, cons_to_vary)
            RHS_list[i] = rhs

        vars_to_vary = self.read_vars_to_vary(problem, self.generation_mode.vars_to_vary)

        self.content.master = problem
        self.content.RHS_list = RHS_list
        self.content.cons_to_vary = cons_to_vary
        self.content.vars_to_vary = vars_to_vary

    def generate_information_possible_vals(self, problem, path=None):
        """
        Auxiliary method of fill_content.

        Is called when self.generation_mode is GenerationModeMasterSlaveDiscreet
        or GenerationModeMasterSlaveContinuous.
        (See documentation of GenerationMode.py for more detailed information).
        """
        name = self.prob_list[2]
        new_path = os.path.join("." if path is None else path, name)
        sto = open(new_path, "r")

        cons_to_vary = None
        RHS_list = []
        index = 0

        counter = 0
        nb_lines = 0
        for line in sto:
            nb_lines += 1
        var_name = None
        possible_vals = []
        weights = []

        sto = open(new_path, "r")

        for line in sto:
            if 1 < counter < nb_lines - 1:
                line_cont = line.split()
                index = len(line_cont) - 1
                if var_name is None:
                    var_name = line_cont[1]
                    cons_to_vary = [var_name]
                    possible_vals.append(float(line_cont[2]))
                    weights.append(float(line_cont[index]))
                elif line_cont[1] == var_name:
                    possible_vals.append(float(line_cont[2]))
                    weights.append(float(line_cont[index]))
                else:
                    var_name = line_cont[1]
                    cons_to_vary.append(var_name)
                    RHS_list.append([possible_vals.copy(), weights.copy()])
                    possible_vals = [float(line_cont[2])]
                    weights = [float(line_cont[index])]

            counter += 1

        RHS_list.append([possible_vals.copy(), weights.copy()])

        cons_to_vary = self.read_cons_to_vary(problem, cons_to_vary)

        master = self.factory.read_problem_from_file(os.path.join("." if path is None else path,
                                                                  self.prob_list[1]),
                                                     simple_problem=self.generation_mode.is_simple)
        vars_to_vary = self.read_vars_to_vary(master, master.get_variable_names())

        master.get_domain_border(all_vars=True)

        self.content.master = master
        self.content.RHS_list = RHS_list
        self.content.cons_to_vary = cons_to_vary
        self.content.vars_to_vary = vars_to_vary


def extract(mode: GenerationMode, factory: ProblemFactory, path=None):
    """
    The function extract and converts the content stocked in the given GenerationMode instance to
    the format required by the class lin_opt_pbs in problem_generator.py and returns the result.
    (See description of fill_content, method of Selector)

    Arguments
    ---------

    mode : GenerationMode instance
        see GenerationMode.py for a detailed description of the different modes availablr
    factory : Problem_factory instance (see in problem_interface.py)
        choose the problem factory that matches the lp solver you are using
    path : str
        path to problem files

    returns
    -------
    content :
        the data required by the class lin_opt_pbs (see problem_generator.py)
    """
    selector = Selector(mode=mode, factory=factory, path=path)
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



