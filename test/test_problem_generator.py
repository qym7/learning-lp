import os
from Problem_xpress import Xpress_Problem_Factory
from problem_generator import problem_generator
from dataset import dataset


def test_every_thing_should_be_ok_on_petit_probleme_lp():
    """Testing the problem_generator function"""
    prob_list = [os.path.join('..', 'data', 'petit_probleme.lp')]
    cons_to_vary = ['demand_0', 'demand_7', 'demand_8']
    vars_to_vary = None
    Number = 10
    Deviation = 0.1

    data = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary, Xpress_Problem_Factory())
    rhs_list = data.get_RHS()
    solutions = data.get_solutions()
    assert len(rhs_list) == len(solutions) == Number and len(rhs_list[0]) == len(cons_to_vary), "" \
        "Error in problem generator function."
