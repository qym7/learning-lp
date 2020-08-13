"""
Class does not implement linear optimisation problems - which is the purpose of Problem
in Problem.py - but objects that are given as attributes to Problem instances and provide
methods for modifying and reading features of the optimisation problem stocked in it.

Two implementations exist:
- FIXO Xpress (see ProblemTypeXpress.py)
- Cplex (see ProblemTypeCplex.py)

All methods take a Problem instance as their first argument, that will usually be
the Problem in which the ProblemType instance is stocked.
"""


class ProblemType:

    def create(self):
        """
        Returns a new empty optimisation problem.

        Returns
        -------
        problem : optimisation problem (exact object depends on implementation of interface)
        """
        pass

    def read(self, problem, filename):
        """
        Loads problem from file into problem.

        Arguments
        ---------
        problem : problem instance
        filename : str
        """
        pass

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
        pass

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
        pass

    def get_number_cols(self, problem):
        """
        Returns the number of constraints of the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        nb_cols : int
        """
        pass

    def set_objective(self, problem, obj):
        """
        Sets objective function of linear optimisation problem to obj.

        Arguments
        ---------
        problem : problem instance
        obj : objective function (type of object depends on implementation)
        """
        pass

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
        pass

    def get_matrix(self, problem):
        """
        Returns constraint matrix A of the linear optimisation problem and
        rhs b such that Ax <= b.

        Arguments
        ---------
        problem : problem instance

        Returns
        -------
        matrix : float list list
        rhs : float list
        """
        pass

    def set_RHS(self, problem, rhs):
        """
        Changes the RHS of the linear optimisation problem to the input rhs.

        Arguments
        ---------
        problem : problem instance
        rhs : (int, float) list
            the format required by cplex (IMPORTANT)
        """
        pass

    def get_constraint_names(self, problem, cons_to_vary=None):
        """
        Returns list of names of constraints of the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance
        cons_to_vary : int list
            indices of constraints whose names should be returned

        Returns
        -------
        name list : string list
        """
        pass

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
        pass

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
        var : variable (exact object depends on implementation of interface)
        """
        pass

    def solve(self, problem):
        """
        Solves the linear optimisation problem.

        Arguments
        ---------
        problem : problem instance
        """
        pass

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
        pass

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
        pass

    def mute_solver(self, problem):
        """
        Disables all messages generated by the solver while solving the optimisation problem.

        Arguments
        ---------
        problem : problem instance
        """
        pass

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
        pass

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
        pass

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
        pass

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
        pass


class ProblemFactory:

    def get_problem_instance(self):
        pass

    def read_problem_from_file(self, filename: str, simple_problem=False):
        pass
