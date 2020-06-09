import cplex
import pickle

# The class Selector is designed to provide a user-friendly interface for selecting files
# containing linear optimization problems. Furthermore it reads the content of the selected
# files and converts it to the format required by the class lin_opt_pbs in problem_generator.py

# Parameters of an instance of Selector:
#       prob_list: a list of linear optimization problems. Should be either a single file name (string),
#              a list of file-names (string list) or a list of objects of the class cplex.Cplex (cplex.Cplex list)
#       content: the content of prob_list in the format required by the class lin_opt_pbs in problem_generator.py.
#


class Selector:
    def __init__(self, prob_list):
        self.prob_list = prob_list
        self.content = None

    def get_prob_list(self):
        return self.prob_list

    def set_prob_list(self, prob_list):
        self.prob_list = prob_list

    def get_content(self):
        return self.content

    def set_content(self, problem, RHS_list, non_fixed_vars, name_list):
        self.content = problem, RHS_list, non_fixed_vars, name_list

    # The method fill_content converts the content of the linear optimization problems in
    # self.prob_list to the format required by the class lin_opt_pbs ans saves it in self.content.

    # More precisely the method checks first of all if the content of prob_list is either a string,
    # a string list, or a cplex.Cplex list. If that is not the case, an error message will occur. If
    # the content of prob_list has one of the types listed above, but cannot be read by the method
    # (if prob_list contains a list of file names, but some of those files do not contain linear
    # optimization problems for example) an exception will be raised.

    # If no exception occurs, the content of prob_list will be converted into a tuple
    # containing:
    #       - one complete linear optimization problem (cplex.Cplex instance),
    #       - the list of RHS of all problems given in the format required by
    #         Cplex (list of list of tuples (int, float)),
    #       - the indices of the coefficients of the RHS that should be affected
    #         when generating new problems (int list)
    #       - the list of names of the problems saved in prob_list (string list)

    # This tuple is saved in self.content

    def fill_content(self):
        if isinstance(self.prob_list, str):  # if prob_list contains a single file name
            try:
                file_content = pickle.load(open(self.prob_list, "rb"))
                if isinstance(file_content[0], cplex.Cplex):
                    problem = file_content[0]
                    RHS_list = []
                    name_list = []
                    for i in range(len(file_content[1])):
                        name_list.append(("problem_(%d)", i))
                    for elem in file_content[1]:
                        constraints = elem.linear_constraints.get_rhs()
                        n = len(constraints)
                        rhs = []
                        for i in range(n):
                            rhs.append((i, constraints[i]))
                        RHS_list.append(rhs)
                else:
                    raise Exception("File does not contain linear optimization problems")
            except pickle.UnpicklingError:
                problem = cplex.Cplex()
                RHS_list = []
                name_list = self.prob_list
                problem.read(self.prob_list)
                constraints = problem.linear_constraints.get_rhs()
                n = len(constraints)
                rhs = []
                for i in range(n):
                    rhs.append((i, constraints[i]))
                RHS_list.append(rhs)

        elif isinstance(self.prob_list[0], str):  # checking if prob_list contains a list of files
            problem = cplex.Cplex()
            RHS_list = []
            name_list = self.prob_list
            for elem in self.prob_list:
                problem.read(elem)
                constraints = problem.linear_constraints.get_rhs()
                n = len(constraints)
                rhs = []
                for i in range(n):
                    rhs.append((i, constraints[i]))
                RHS_list.append(rhs)
        else:  # checking finally if prob_list contains a cplex.Cplex list, throws error message otherwise
            assert isinstance(self.prob_list[0], cplex.Cplex), "This file does not contain " \
                                                         "linear optimisation problems."
            problem = self.prob_list[0]
            RHS_list = []
            name_list = []
            for i in range(len(self.prob_list)):
                name_list.append(("problem_(%d)", i))
            for elem in self.prob_list:
                constraints = elem.linear_constraints.get_rhs()
                n = len(constraints)
                rhs = []
                for i in range(n):
                    rhs.append((i, constraints[i]))
                RHS_list.append(rhs)

        non_fixed_vars = read_non_fixed_vars(problem)
        self.set_content(problem, RHS_list, non_fixed_vars, name_list)


def read_non_fixed_vars(problem):
    constraint_names = problem.linear_constraints.get_names()
    non_fixed_vars = []
    for i in range(len(constraint_names)):
        this_name = constraint_names[i]
        if this_name.startswith("demand"):
            non_fixed_vars.append(i)
    print("non fixed vars ", non_fixed_vars)
    return non_fixed_vars


# The function extract converts the content of a list of problems given as an argument to
# the format required by the class lin_opt_pbs in problem_generator.py and returns the result.
# (See description of fill_content, method of Selector)

def extract(prob_list):
    selector = Selector(prob_list)
    selector.fill_content()
    content = selector.get_content()
    return content


# Testing the extract function on an lp-file

problem_file = 'petit_probleme.lp'
content = extract(problem_file)
print(content)