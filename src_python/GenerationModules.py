"""
Generation modules are objects that can be given to a GenerationMode object (see GenerationMode.py)
in order to slightly modify how some methods involved in the process of generating new problems
behave.

Should only be used when the base problem is not simple and thus either the inner point method or
the random vertices method are used to generate new problems.
"""

from MathFunctions import convex_comb, get_weights_for_convex_comb
import numpy as np
import random


class GenerationModule:
    """Interface for generation modules."""

    def compute_additional_vertices(self, problem):
        pass

    def compute_inner_points_non_standard(self, problem, nb_inner_point):
        pass


class GenerationModuleStormInnerPoint(GenerationModule):
    """
    GenerationModule to be used when using storm to generate new problems with the inner
    point method.

    storm is a non simple problem, so the exhaustive list of its domain's vertices can not be computed.
    Thus only the random vertices and the inner point method can be used by problem_generator.py to generate
    storm-based problems.

    When using the inner point method, candidates for inner points of storms domain ar computed by making convex
    combinations of random vertices. However experience has shown that almost all vertices lie on a single
    (n-1) dimensional side of that domain, such that just by choosing random vertices, it is very unlikely
    to actually get a point in the interior. Thus, this GenerationModule adds some well chosen vertices to the
    list of random vertices used for the convex combination, with high weights. That way, the inner point
    candidates usually lie effectively in the interior of the domain.
    """

    def compute_additional_vertices(self, problem):
        """
        Method that modifies the behaviour of Problem.approx_var_bounds (see Problem.py).
        """
        matrix, rhs = problem.get_matrix()
        lines = [57, 58, 59, 60, 61, 62, 63]
        nb = len(matrix[0])
        variables = []

        additional_vertices = [None] * len(lines)
        counter = 0

        for elem in lines:
            for i in range(nb):
                if matrix[elem][i] != 0:
                    variables.append((i, matrix[elem][i]))

            nb_vars = len(variables)

            objective = problem.get_variable(variables[0][0]) * variables[0][1]

            for i in range(1, nb_vars):
                objective = objective + problem.get_variable(variables[i][0]) * variables[i][1]

            problem.content.setObjective(objective)
            problem.solve()
            additional_vertices[counter] = problem.get_sol()
            counter += 1

        problem.domain.add_some_vertices(additional_vertices)

    def compute_inner_points_non_standard(self, domain, nb_inner_points):
        """
        Method that modifies the behaviour of Domain.compute_inner_points (see Problem.py).
        """
        nb = len(domain.some_vertices) - 7

        inner_points = nb_inner_points * [None]

        for j in range(nb_inner_points):
            weights = get_weights_for_convex_comb(nb)

            vertices = domain.some_vertices.copy()[:-7]
            var_values = np.dot(np.transpose(vertices), weights)

            coeff57 = random.random() * 0.2 + 0.001
            coeff58 = random.random() * 0.1
            coeff59 = random.random() * 0.1
            coeff60 = random.random() * 0.025
            coeff61 = random.random() * 0.025
            coeff62 = random.random() * 0.025
            coeff63 = random.random() * 0.025

            sum = coeff57 + coeff58 + coeff59 + coeff60 + coeff61 + coeff62 + coeff63

            coeff57 = coeff57 / (sum * 2)
            coeff58 = coeff58 / (sum * 2)
            coeff59 = coeff59 / (sum * 2)
            coeff60 = coeff60 / (sum * 2)
            coeff61 = coeff61 / (sum * 2)
            coeff62 = coeff62 / (sum * 2)
            coeff63 = coeff63 / (sum * 2)

            var_values = coeff57 * np.array(domain.some_vertices[-1]) + \
                         coeff58 * np.array(domain.some_vertices[-7]) + \
                         coeff59 * np.array(domain.some_vertices[-6]) + \
                         coeff60 * np.array(domain.some_vertices[-5]) + \
                         coeff61 * np.array(domain.some_vertices[-6]) + \
                         coeff62 * np.array(domain.some_vertices[-6]) + \
                         coeff63 * np.array(domain.some_vertices[-6]) + 0.5 * var_values

            inner_points[j] = var_values

        domain.add_inner_points(inner_points)

