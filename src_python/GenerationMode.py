"""
GenerationMode objects state how the module problem_generator.py should generate
new problems out of a given linear optimisation problem or a list of linear optimisation problems.

Three generation modes, which apply in different situations, are implemented so far:

GenerationModeClassic
---------------------
This mode should be used when:
- you have a single linear optimisation problem or a list of linear optimisation
  problems that differ by some constraints in their rhs only
- a list of variables to set randomly such that they verify the problems constraints
  (note that the constraints that affect the variables values should not vary from
  one problem to another)
- a list of constraints to vary randomly (this list can by given by the user, if it is not
  and you have more then one base problem, you can specify determine_cons_to_vary = True such
  that the class will determine the constraints that vary from one problem to another by
  itself.

For each newly generated problem, the GenerationMode object then chooses a random base problem,
copies that problem and modifies the copy by adding a gaussian noise to the constraints meant to
be varied and sets the variables randomly such that they verify the problem's constraints.

(*)

There are different methods to set the variables, which apply in different situations:
- if the problem is simple (attribute is_simple of GenerationMode object is set to True),
  an exhaustive list of the vertices of the problem's domain is computed and the variables
  are set by convex combinations of these vertices
- if the problem is not simple, the variables are either set by the inner point method or
  the random vertices method:
    + The inner point method determines the smallest box containing
      the problem's domain and some interior points of that domain. Then it chooses
      a random point x in that box, links it to a random interior point y, computes the nearest
      point z on the line containing x and y such that z lies in the problem's domain (so z is
      almost a border point) and eventually shifts it towards y, with a random factor between 0 and 1.
    + The random vertices method computes a few random vertices of the problem's domain and sets
      the variables by making convex combinations of these points.


GenerationModeMasterSlaveDiscreet
---------------------------------
This mode should be used to generate new optimisation problems out of base problems
that were obtained by doing a Bender's decomposition of a large optimisation problem.

In that case, you should have:
- a master problem
- a slave problem
- a stochastic file containing the names of the constraints to vary, a discrete set of possible
  values for each constraint and probability weights for those values

The slave problem is larger then the master and contains variables whose names are identical to the
master's variables. These variables are fixed to random values within the master's domain of definition
(which is automatically contained in the domain of definition of the slave problem). Eventually, the constraints
listed in the stochastic file are fixed randomly to values inside the respective discreet sets, respecting
of course the associated probability weights.

There are different methods to set the variables, which apply in different situations (see above
for a more detailed description (*).

GenerationModeMasterSlaveContinuous
-----------------------------------
Similar to GenerationModeMasterSlaveDiscreet, with the exception that the sets of discreet values
in the stochastic file are more or less ignored:

Each constraint is fixed randomly the smallest interval containing the associated discreet set.
Then a gaussian noise is added.
"""

from GenerationModules import GenerationModule


class GenerationMode:

    def get_prob_list(self):
        pass

    def prepare_vertices(self, lps, nb):
        pass

    def generate_information(self, selector, problem, path=None):
        pass

    def choose_vars_random(self, lps):
        pass

    def choose_constraints_random(self, lps, k):
        pass


class GenerationModeClassic(GenerationMode):
    """
    For precise description look at the beginning of the module (Generationmode.py).

    Attributes
    ----------
    prob_list : string list
        a list of linear optimization problems. Should be a list of file-names containing linear
        optimisation problems that differ only by some values in their rhs.
    cons_to_vary : string list
        a list of names of the constraints of the linear optimisation problems that should be affected when
        generating new problems. (All linear optimisation problems in prob_list should have an equal
        amount of constraints with equal names and in the same order.)
    vars_to_vary : string list
        a list of names of the variables of the linear optimisation problems that should be fixed randomly
        when generating new problems. (All linear optimisation problems in prob_list should have an equal
        amount of variables with equal names and in the same order.)
    vertices : float list list
        user can provide an exhaustive list of the vertices of one of the problems' domains if such a list
        has already been computed.
    determine_cons_to_vary : bool
    simple_problem : bool
    trivial_constraints_only : bool
        states whether the variables to vary appear only in trivial constraints. Process of setting them is
        much faster in that case
    use_random_vertices_method : bool
        provided that the base problems are not simple, states whether the random vertices method should
        be used to set the variables. If False, the interior point method is used.
    """
    def __init__(self, prob_list, cons_to_vary=None, vars_to_vary=None, determine_cons_to_vary=False,
                 simple_problem=False, trivial_constraints_only=False, vertices=None,
                 use_random_vertices_method=False):
        self.prob_list = prob_list
        self.cons_to_vary = cons_to_vary
        self.vars_to_vary = vars_to_vary
        self.vertices = vertices
        self.determine_cons_to_vary = determine_cons_to_vary
        self.is_simple = simple_problem
        self.is_trivial = trivial_constraints_only
        self.use_rand_vert = use_random_vertices_method

    def warning(self):
        if self.cons_to_vary is None and self.vars_to_vary is None and not self.determine_cons_to_vary:
            print("WARNING: You stated that all constraints are fixed and no variables will be set."
                  "Thus the same linear optimisation problem will be generated over and over again.")

    def get_prob_list(self):
        return self.prob_list

    def prepare_vertices(self, lps, nb):
        if not self.is_simple:
            if self.use_rand_vert:
                lps.compute_random_vertices_of_master(nb)

    def generate_information(self, selector, problem, path=None):
        return selector.generate_information_classic(problem, path)

    def choose_vars_random(self, lps):
        if self.is_trivial:
            return lps.choose_vars_random_trivial()
        elif self.is_simple:
            return lps.choose_vars_random_convex_comb()
        else:
            if self.use_rand_vert:
                return lps.choose_vars_random_vertices_method()
            else:
                return lps.choose_vars_random_interior_point_method()

    def choose_constraints_random(self, lps, k):
        return lps.choose_constraints_random_perturbation(k)


class GenerationModeMasterSlaveDiscreet(GenerationMode):
    """
    For precise description look at the beginning of the module (Generationmode.py).

    Attributes
    ----------
    master : str
        name of the file containing the master problem
    slave : str
        name of the file containing the slave problem
    sto : str
        name of the stochastic file
    vertices : float list list
        user can provide an exhaustive list of the vertices of the master's domain if it has already been
        computed.
    is_simple : bool
    modules : list of GenerationModule instances
        (see GenerationModules.py for a precise description)
    use_random_vertices_method : bool
        provided that the base problems are not simple, states whether the random vertices method should
        be used to set the variables. If False, the interior point method is used.
    """
    def __init__(self, master, slave, sto, vertices=None, simple_problem=False, modules=None,
                 use_random_vertices_method=False):
        self.master = master
        self.slave = slave
        self.sto = sto
        self.vertices = vertices
        self.is_simple = simple_problem
        self.modules = modules
        self.use_rand_vert = use_random_vertices_method

    def get_prob_list(self):
        prob_list = [self.slave, self.master, self.sto]
        return prob_list

    def prepare_vertices(self, lps, nb):
        if not self.is_simple:
            if self.use_rand_vert:
                lps.compute_random_vertices_of_master(nb)

    def generate_information(self, selector, problem, path=None):
        return selector.generate_information_possible_vals(problem, path)

    def choose_vars_random(self, lps):
        if self.is_simple:
            return lps.choose_vars_random_convex_comb()
        else:
            if self.use_rand_vert:
                return lps.choose_vars_random_vertices_method()
            else:
                return lps.choose_vars_random_interior_point_method()

    def choose_constraints_random(self, lps, k):
        return lps.choose_constraints_random_discreet()


class GenerationModeMasterSlaveContinuous(GenerationMode):
    """
    For precise description look at the beginning of the module (Generationmode.py).

    Attributes
    ----------
    master : str
        name of the file containing the master problem
    slave : str
        name of the file containing the slave problem
    sto : str
        name of the stochastic file
    vertices : float list list
        user can provide an exhaustive list of the vertices of the master's domain if it has already been
        computed.
    is_simple : bool
    modules : list of GenerationModule instances
        (see GenerationModules.py for a precise description)
    use_random_vertices_method : bool
        provided that the base problems are not simple, states whether the random vertices method should
        be used to set the variables. If False, the interior point method is used.
    """
    def __init__(self, master, slave, sto, vertices=None, simple_problem=False, modules=None,
                 use_random_vertices_method=False):
        self.master = master
        self.slave = slave
        self.sto = sto
        self.vertices = vertices
        self.is_simple = simple_problem
        self.modules = modules
        self.use_rand_vert = use_random_vertices_method

    def get_prob_list(self):
        prob_list = [self.slave, self.master, self.sto]
        return prob_list

    def prepare_vertices(self, lps, nb):
        if not self.is_simple:
            if self.use_rand_vert:
                lps.compute_random_vertices_of_master(nb)

    def generate_information(self, selector, problem, path=None):
        return selector.generate_information_possible_vals(problem, path)

    def choose_vars_random(self, lps):
        if self.is_simple:
            return lps.choose_vars_random_convex_comb()
        else:
            if self.use_rand_vert:
                return lps.choose_vars_random_vertices_method()
            else:
                return lps.choose_vars_random_interior_point_method()

    def choose_constraints_random(self, lps, k):
        return lps.choose_constraints_random_continuous()
