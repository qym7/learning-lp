"""
Implementation for FICO Xpress of the ProblemType interface in ProblemTypeInterface.py.

See ProblemTypeInterface.py for further information.
"""

from ProblemTypeInterface import ProblemFactory, ProblemType
from Problem import Problem
import xpress as xp
import numpy as np


class XpressType(ProblemType):
    def create(self):
        """
        Returns a new empty optimisation problem.

        Returns
        -------
        problem : xpress.problem
        """
        return xp.problem()

    def read(self, problem, filename):
        """
        Loads problem from file into problem.

        Arguments
        ---------
        problem : problem instance
        filename : str
        """
        problem.content.read(filename)

    def get_RHS(self, problem, cons_to_vary=None, all_cons=False):
        """
        Returns the RHS of the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance
        cons_to_vary : int list
            indices of RHS to be returned
        all_cons : bool
            when True, all constraints are returned

        Returns
        -------
        rhs : float list (IMPORTANT)
        """
        rhs = []
        if all_cons:
            problem.content.getrhs(rhs, 0, self.get_number_vars(problem) - 1)
        elif cons_to_vary is not None:
            for elem in cons_to_vary:
                aux = []
                problem.content.getrhs(aux, elem, elem)
                rhs.append(aux[0])
        return rhs

    def get_number_vars(self, problem):
        """
        Returns the dimension of the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        nb_vars : int
        """
        return problem.content.attributes.cols

    def get_number_cons(self, problem):
        """
        Returns the number of constraints of the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        nb_cols : int
        """
        return problem.content.attributes.rows

    def set_objective(self, problem, obj):
        """
        Sets objective function of linear optimisation problem to obj.

        Arguments
        ---------
        problem : problem instance
        obj : xpress objective function
            an express objective function can for example be built by applying
            basic operations on xpress variable objects (ex. var1 + 2 * var2).
        """
        problem.content.setObjective(obj)

    def set_constraints(self, problem):
        """
        Sets problem.constraints to (A, b), where A is the constraint matrix of the linear optimisation problem and
        b the rhs such that Ax <= b.

        The purpose of that method is to create a matrix representation of the problems constraints,
        nothing is modified.

        Arguments
        ---------
        problem : problem instance
        """
        nb_rows = self.get_number_cons(problem)
        nb_cols = self.get_number_vars(problem)
        nb_e = 0

        names = self.get_variable_names(problem)

        indices = []
        values = []
        row_types = []
        problem.content.getrowtype(row_types, 0, nb_rows - 1)

        for elem in row_types:
            if elem == 'E':
                nb_e += 1

        matrix = (nb_rows + nb_e) * [None]
        rhs = (nb_rows + nb_e) * [None]

        index = 0

        for i in range(nb_rows):
            row = np.array(nb_cols * [None])
            counter = 0
            problem.content.getrows(mstart=None, mclind=indices, dmatval=values, size=nb_cols, first=i, last=i)
            non_zeros = len(values)
            elem = []
            problem.content.getrhs(elem, i, i)

            for j in range(nb_cols):
                if counter < non_zeros and names[j] == indices[counter].name.strip():
                    row[j] = values[counter]
                    counter += 1
                else:
                    row[j] = 0

            if row_types[i] == 'L':
                matrix[index] = row
                rhs[index] = elem[0]
                index += 1
            elif row_types[i] == 'G':
                matrix[index] = -row
                rhs[index] = -elem[0]
                index += 1
            else:
                matrix[index] = row
                matrix[index + 1] = -row
                rhs[index] = elem[0]
                rhs[index + 1] = -elem[0]
                index += 2

        problem.constraints = np.array(matrix), np.array(rhs)

    def get_matrix(self, problem):
        """
        Returns constraint matrix of the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        matrix : float list list
        """
        if problem.constraints is None:
            self.set_constraints(problem)
            return problem.constraints
        else:
            return problem.constraints

    def set_RHS(self, problem, rhs):
        """
        Changes the RHS of the linear optimisation problem to the input rhs.

        Arguments
        ---------
        problem : problem instance
        rhs : (int, float) list
            the format required by cplex (IMPORTANT)
        """
        problem.content.chgrhs(mindex=[i[0] for i in rhs], rhs=[i[1] for i in rhs])

    def get_constraint_names(self, problem, cons_to_vary=None):
        """
        Returns list of names of constraints of the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance
        cons_to_vary : int list
            indices of constraints whose names should be returned.

        Returns
        -------
        name list : string list
        """
        if cons_to_vary is None:
            constraints = problem.content.getConstraint()
            nb_cons = len(constraints)
            name_list = nb_cons * [None]
            for i in range(nb_cons):
                name_list[i] = constraints[i].name.strip()
        else:
            constraints = problem.content.getConstraint()
            nb_cons = len(cons_to_vary)
            name_list = nb_cons * [None]
            for i in range(nb_cons):
                name_list[i] = constraints[cons_to_vary[i]].name.strip()

        return name_list

    def get_variable(self, problem, ind):
        """
        Returns variable with index ind.

        Arguments
        ---------
        problem : problem instance
        ind : int
            index of variable to be returned

        Returns
        -------
        var : xpress variable object
        """
        return problem.content.getVariable(ind)

    def get_variable_names(self, problem):
        """
        Returns list of names of variables of the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        name list : string list
        """
        variables = problem.content.getVariable()
        nb_vars = len(variables)
        name_list = nb_vars * [None]
        for i in range(nb_vars):
            name_list[i] = variables[i].name.strip()
        return name_list

    def solve(self, problem):
        """
        Solves the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance
        """
        problem.content.solve()

    def get_objective_value(self, problem):
        """
        Returns the solution of the linear optimisation problem (objective value).

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        val : float
        """
        return problem.content.getObjVal()

    def get_sol(self, problem):
        """
        Returns the solution of the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        sol : float list
        """
        return problem.content.getSolution()

    def mute_solver(self, problem):
        """
        Disables all messages generated by the solver while solving the optimisation problem.

        Arguments
        ---------
        problem : problem instance
        """
        problem.content.setControl("outputlog", 0)

    def var_get_bounds(self, problem, ind):
        """
        Returns the bounds of the variable with index ind.

        Arguments
        ---------
        problem : problem instance
        ind : int
            variable whose bounds are to be returned

        Returns
        -------
        bounds : (float, float)
        """
        var = problem.content.getVariable(ind)
        return var.lb, var.ub

    def var_set_bounds(self, problem, ind, lw_bnd, up_bnd):
        """
        Sets the bounds of the variable with index ind to lw_bnd and up_bnd.

        Arguments
        ---------
        problem : problem instance
        ind : int
            index of variable whose bounds should be modified
        lw_bnd : float
            new lower bound
        up_bnd : float
            new upper bound
        """
        problem.content.chgbounds([ind, ind], ["L", "U"], [lw_bnd, up_bnd])

    def get_status(self, problem):
        """
        Returns the status of the solution.

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        status : type depends on implementation
        """
        return problem.content.getProbStatusString()

    def is_feasible(self, problem):
        """
        True if problem is feasible.

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        is_feasible : bool
        """
        if self.get_status(problem) == "lp_infeas":
            return False
        else:
            return True


class XpressProblemFactory(ProblemFactory):

    def get_problem_instance(self) -> Problem:
        return Problem(prob_type=XpressType())

    def read_problem_from_file(self, filename: str, simple_problem=False) -> Problem:
        p = Problem(simple_problem=simple_problem, prob_type=XpressType())
        p.read(filename)
        return p