"""
Module implements single linear optimisation problems.

Contains two classes:
- Problem : objects represent linear optimisation problems
- Domain : objects represent domains on which linear optimisation problems are defined.
           is essentially used to compute and stock vertices of the domain of the linear optimisation problem
           or compute and stock the bounds of the smallest box containing the domain

Every Problem instance has a Domain instance as attribute.
"""

from MathFunctions import convex_comb, get_weights_for_convex_comb
import numpy as np
import pypoman
import random
import os


class Domain:
    """
    The purpose of Domain is to get and stock useful information of the domain of a linear optimisation problem.

    Usually the class will not be used independently, but as an auxiliary class of Problem (see class Problem).

    Attributes
    ----------
    domain_vertices : numpy array (2 dimensional)
        exhaustive list of vertices of the domain of a linear optimisation problem. Can be given as an
        argument when creating a new object
    inner points : numpy array (2 dimensional)
        list of one or more inner points of the domain of a linear optimisation problem. Is None by
        default and filled by the method compute_inner_points
    some_vertices : numpy array (2 dimensional)
        list of some vertices of the domain of a linear optimisation problem. Useful when the exhaustive
        list of vertices of that domain would be too big to be computed. For some applications, like
        computing some inner points of the domain, no exhaustive list is needed
    approx_domain_box : list of (float, float)
        bounds of the smallest rectangular box containing the domain of a linear optimisation problem
    max_dimension : float
        maximal extension of the smallest rectangular box containing the domain of a linear optimisation
        problem.
    """
    def __init__(self, vertices=None):
        self.domain_vertices = vertices
        self.inner_points = None
        self.some_vertices = None
        self.approx_domain_box = None
        self.max_dimension = None

    def set_domain_vertices(self, vertices):
        self.domain_vertices = vertices

    def set_approx_domain_box(self, bounds):
        self.domain_vertices = bounds

    def set_some_vertices(self, vertices):
        self.some_vertices = vertices

    def add_some_vertices(self, vertices):
        self.some_vertices.extend(vertices)

    def get_domain_vertices(self):
        return self.domain_vertices

    def get_approx_domain_box(self):
        return self.domain_vertices

    def get_some_vertices(self):
        return self.some_vertices

    def get_inner_points(self):
        return self.inner_points

    def add_inner_points(self, points):
        self.inner_points.extend(points)

    def compute_max_dim(self):
        max_dim = 0

        if self.approx_domain_box is not None:
            for elem in self.approx_domain_box:
                if elem[1] >= max_dim:
                    max_dim = elem[1]
            self.max_dimension = max_dim

    def compute_inner_points(self, nb_inner_points, modules=None):
        """
        Computes some points that possibly lie in the interior of the
        domain of definition of a linear optimisation problem.

        Method needs self.some_vertices to contain at least two vectors
        representing vertices of the domain. Computes candidates for inner
        points by making convex combinations of the vectors in self.some_vertices.
        Nothing guarantees that the results of those convex combinations actually
        are in the interior of the domain. In general, the higher the amount of
        points in self.some_vertices is, the more probable it will be that the
        results are effectively in the interior.

        By default, all points in some_vertices are involved in the convex combinations,
        the weights for every combination are chosen randomly and no point is favoured.

        This behaviour can by modified by providing one or more GenerationModule objects.

        Arguments
        ---------
        modules : list of GenerationModule instances
        nb_inner_points : int
            number of inner points to be computed
        """
        if self.some_vertices is None:
            self.inner_points = None
        else:
            nb = len(self.some_vertices)
            inner_points = nb_inner_points * [None]

            if modules is None:
                for j in range(nb_inner_points):
                    vertices = self.some_vertices.copy()
                    weights = get_weights_for_convex_comb(nb)

                    var_values = np.dot(np.transpose(vertices), weights)
                    inner_points[j] = var_values

                self.inner_points = inner_points

            else:
                self.inner_points = []
                for module in modules:
                    module.compute_inner_points(self, nb_inner_points)


class Problem:
    """
    Class implements linear optimisation problems.

    Attributes
    ----------
    type : ProblemType instance (see ProblemTypeInterface.py, ProblemTypeXpress.py and ProblemTypeCplex.py
           for further and more precise information)
        must be given as an attribute when creating an object
    content : linear optimisation problem
        type is xpress.problem if self.type is a XpressType instance and cplex.Cplex if self.type is
        a CplexType instance
    domain : Domain instance (see class Domain)
        computes and stocks useful information about the domain of the linear optimisation problem
    constraints : (numpy.array (2 dimensional), numpy array (1 dimensional))
        matrix of constraints A and rhs b such that constraints of linear optimisation problem can
        written as A * X <= b. Set to None when problem is created and filled by the method
        set_constraints.
    is_simple : bool
        states whether the problem is simple enough that an exhaustive list of the vertices of its
        domain can be computed or not. Will determine which methods will be used by the problem_generator
        and problem_selector modules (see problem_generator.py and problem_selector.py) to generate
        new random problems out of this Problem instance.
    file_name : str
        if problem was loaded from file, the file's name is stocked in self.file_name
    """
    def __init__(self, prob_type, simple_problem=False):
        self.type = prob_type
        self.content = prob_type.create()
        self.mute_solver()
        self.domain = Domain()
        self.constraints = None
        self.is_simple = simple_problem
        self.file_name = None

    def read(self, filename):
        """
        Loads problem from file.

        Arguments
        ---------
        filename : str
        """
        self.type.read(self, filename)
        self.file_name = filename

    def get_RHS(self, cons_to_vary=None, all_cons=False):
        """
        Returns the RHS of the linear optimisation problem.

        Arguments
        ---------
        cons_to_vary : int list
            indices of RHS to be returned
        all_cons : bool
            when True, all constraints are returned

        Returns
        -------
        rhs : float list (IMPORTANT)
        """
        return self.type.get_RHS(self, cons_to_vary, all_cons)

    def get_number_vars(self):
        """
        Returns the dimension of the linear optimisation problem.

        Returns
        -------
        nb_vars : int
        """
        return self.type.get_number_vars(self)

    def get_number_cols(self):
        """
        Returns the number of constraints of the linear optimisation problem.

        Returns
        -------
        nb_cols : int
        """
        return self.type.get_number_cols(self)

    def get_inner_points(self):
        """Returns self.inner_points."""
        return self.domain.get_inner_points()

    def get_max_dimension(self):
        return self.domain.max_dimension

    def get_some_vertices(self):
        return self.domain.some_vertices

    def set_some_vertices(self, vertices):
        self.domain.set_some_vertices(vertices)

    def set_objective(self, obj):
        """
        Sets objective function of linear optimisation problem to obj.

        Arguments
        ---------
        obj : objective function (type of object depends on implementation)
        """
        self.type.set_objective(self, obj)

    def set_constraints(self):
        """
        Sets problem.constraints to (A, b), where A is the constraint matrix of the linear optimisation problem and
        b the rhs such that Ax <= b.

        The purpose of that method is to create a matrix representation of the problems constraints,
        nothing is modified.
        """
        self.type.set_constraints(self)

    def get_matrix(self):
        """
        Returns constraint matrix A of the linear optimisation problem and
        rhs b such that Ax <= b.

        Returns
        -------
        matrix : float list list
        rhs : float list
        """
        return self.type.get_matrix(self)

    def set_RHS(self, rhs):
        """
        Changes the RHS of the linear optimisation problem to the input rhs.

        Arguments
        ---------
        rhs : (int, float) list
            the format required by cplex (IMPORTANT)
        """
        self.type.set_RHS(self, rhs)

    def get_constraint_names(self, cons_to_vary=None):
        """
        Returns list of names of constraints of the linear optimisation problem.

        Arguments
        ---------
        cons_to_vary : int list
            indices of constraints whose names should be returned

        Returns
        -------
        name list : string list
        """
        return self.type.get_constraint_names(self, cons_to_vary)

    def get_variable(self, ind):
        """
        Returns variable with index ind.

        Arguments
        ---------
        ind : int
            index of variable to be returned

        Returns
        -------
        var : variable (exact object depends on class of self.type)
        """
        return self.type.get_variable(self, ind)

    def get_variable_names(self):
        """
        Returns list of names of variables of the linear optimisation problem.

        Returns
        -------
        name list : string list
        """
        return self.type.get_variable_names(self)

    def solve(self):
        """Solves the linear optimisation problem."""
        self.type.solve(self)

    def get_objective_value(self):
        """Returns the solution of the linear optimisation problem (objective value)."""
        return self.type.get_objective_value(self)

    def get_sol(self):
        """Returns the solution of the linear optimisation problem."""
        return self.type.get_sol(self)

    def mute_solver(self):
        """Disables all messages generated by the solver while solving the optimisation problem."""
        self.type.mute_solver(self)

    def var_get_bounds(self, ind):
        """
        Returns the bounds of the variable with index ind.

        Arguments
        ---------
        ind : int
            variable whose bounds are to be returned

        Returns
        -------
        bounds : (float, float)
        """
        return self.type.var_get_bounds(self, ind)

    def var_set_bounds(self, ind, lw_bnd, up_bnd):
        """
        Sets the bounds of the variable with index ind to lw_bnd and up_bnd.

        Arguments
        ---------
        ind : int
            index of variable whose bounds should be modified
        lw_bnd : float
            new lower bound
        up_bnd : float
            new upper bound
        """
        self.type.var_set_bounds(self, ind, lw_bnd, up_bnd)

    def get_status(self):
        """
        Returns the status of the solution.

        Returns
        -------
        status : type depends on implementation
        """
        return self.type.get_status(self)

    def is_feasible(self):
        """
        True if problem is feasible.

        Returns
        -------
        is_feasible : bool
        """
        return self.type.is_feasible(self)

    def compute_random_vertex(self, factory=None):
        """
        Computes a single random vertex of the domain of the linear optimisation problem
        and returns it.

        Arguments
        ---------
        factory : ProblemFactory instance
            needed because the method creates an auxiliary Problem instance
        """
        if factory is None:
            aux_problem = self
        else:
            aux_problem = factory.read_problem_from_file(self.file_name)

        nb_vars = self.get_number_vars()
        random_vect = nb_vars * [None]

        for i in range(nb_vars):
            val = random.random()
            random_vect[i] = val

        objective = random_vect[0] * self.get_variable(0)

        for i in range(1, nb_vars):
            objective = objective + random_vect[i] * self.get_variable(i)

        aux_problem.set_objective(objective)
        aux_problem.solve()

        return aux_problem.get_sol()

    def compute_vertices_poly(self):
        """
        Computes the vertices of the domain of the linear
        optimisation problem and saves them in self.domain.

        Method will only be used when self.is_simple is True. If that is the case, the
        computed vertices will be used by the module problem_generator.py to generate new problems
        out of this Problem instance.
        """
        if self.domain.domain_vertices is None:
            matrix, rhs = self.get_matrix()
            index = len(matrix[0])

            identity = - np.identity(index)
            zeros = np.zeros(index)

            matrix = np.concatenate((matrix, identity))
            rhs = np.concatenate((rhs, zeros))

            vertices = pypoman.compute_polytope_vertices(matrix, rhs)

            self.domain.set_domain_vertices(vertices)

    def approx_var_bounds(self, vars_to_vary=None, all_vars=True, modules=None):
        """
        Computes the bounds of the smallest rectangular box that contains the domain of the linear
        optimisation problem and stocks them in self.domain.

        These bounds will be used by some methods in problem_generator.py in order to generate
        new problems out of this Problem instance. Method is only used when self.is_simple is False,
        which means that it will be too difficult to compute an exhaustive list of its domain's vertices
        with the method compute_vertices_poly (see compute_vertices_poly).

        Method also computes some vertices of the problem's domain while computing the bounds. Those
        vertices are the stocked in self.domain and can be used to compute some point that possibly lie in the
        interior of that domain.

        Additional vertices can be computed by GenerationModules if given.

        Arguments
        ---------
        vars_to_vary : str list
            list of names of variables whose approximate bounds are to be computed
        all_vars : bool
            states whether the approximate bounds of all vars are to be computed
        modules : list og GenerationModule instances
        """
        if all_vars:
            nb_vars = self.get_number_vars()
            bounds = nb_vars * [None]
            some_vertices = 2 * nb_vars * [None]

            for i in range(nb_vars):
                var = self.get_variable(i)
                self.set_objective(var)
                self.solve()
                lb = self.get_objective_value()
                lp = np.array(self.get_sol())
                self.set_objective(-var)
                self.solve()
                ub = - self.get_objective_value()
                up = np.array(self.get_sol())
                bounds[i] = (lb, ub)
                some_vertices[2*i] = lp
                some_vertices[2*i + 1] = up

        else:
            if vars_to_vary is None:
                bounds = None
                some_vertices = None
                nb_vars = 0
            else:
                nb_vars = len(vars_to_vary)
                bounds = nb_vars * [None]
                some_vertices = 2 * nb_vars * [None]

                for i in range(nb_vars):
                    var = self.get_variable(vars_to_vary[i])
                    self.set_objective(var)
                    self.solve()
                    lb = self.get_objective_value()
                    lp = np.array(self.get_sol())
                    self.set_objective(-var)
                    self.solve()
                    ub = - self.get_objective_value()
                    up = np.array(self.get_sol())
                    bounds[i] = (lb, ub)
                    some_vertices[2 * i] = lp
                    some_vertices[2 * i + 1] = up

        self.domain.set_some_vertices(some_vertices)

        if modules is not None:
            for module in modules:
                module.compute_additional_vertices(self)

        self.domain.set_approx_domain_box(bounds)
        self.domain.compute_inner_points(nb_inner_points=2 * nb_vars, modules=modules)
        self.domain.compute_max_dim()

    def get_domain_border(self, vars_to_vary=None, all_vars=True, modules=None):
        """
        Returns either the exact vertices of the domain of the linear optimisation problem
        if the problem is simple (self.is_simple is True) or the bounds of an approximate box containing the
        domain else.

        Arguments
        ---------
        vars_to_vary : str list
            list of names of variables whose approximate bounds are to be computed
            (only used when problem is not simple)
        all_vars : bool
            states whether the approximate bounds of all vars are to be computed
        modules : GenerationModule instance
            only use is to modify the behaviour of approx_var_bounds, which will only
            be called if problem is not simple

        Returns
        -------
        vertices or bounds : (float, float) list
            list of vertices or approximate bounds of domain
        """
        if self.domain.domain_vertices is not None:
            return self.domain.get_domain_vertices()
        if self.domain.approx_domain_box is not None:
            return self.domain.get_approx_domain_box()
        elif self.is_simple:
            self.compute_vertices_poly()
            return self.domain.get_domain_vertices()
        else:
            self.approx_var_bounds(vars_to_vary, all_vars, modules)
            return self.domain.get_approx_domain_box()


if __name__ == '__main__':

    if False:
        petit_probleme = Xpress_Problem_Factory().read_problem_from_file("petit_probleme.lp")
        petit_probleme.set_RHS([(25, -12)])
        petit_probleme.solve()
        print(petit_probleme.get_objective_value())
        print(petit_probleme.get_status())

    if False:

        master = Xpress_Problem_Factory().read_problem_from_file(
            "D:\\repository\\learning-lp\\data\\Original_problems\\storm\\stormmaster.mps")

        matrix, rhs = master.get_matrix()
        line = 63
        nb = len(matrix[0])
        variables = []

        for i in range(nb):
            if matrix[line][i] != 0:
                variables.append((i, matrix[line][i]))

        nb_vars = len(variables)

        objective = master.get_variable(variables[0][0]) * variables[0][1]

        for i in range(1, nb_vars):
            objective = objective + master.get_variable(variables[i][0]) * variables[i][1]

        master.content.setObjective(objective)
        master.solve()
        print(master.get_objective_value())

        master.content.setObjective(-objective)
        master.solve()
        print(-master.get_objective_value())
