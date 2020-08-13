import os
from Problem import Xpress_Problem_Factory
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


def test_generation():
    """
    Test generation on petit_probleme.lp, varying constraints demand_0, demand_7 and demand_8
    one by one.
    """
    cons_list = ["demand_0", "demand_7", "demand_8"]
    Number = 10000
    Deviation = 1.5
    prob_list = ["petit_probleme.lp"]
    vars_to_vary = None
    for elem in cons_list:
        cons_to_vary = [elem]
        data = problem_generator(prob_list, Number, Deviation, cons_to_vary, vars_to_vary, Xpress_Problem_Factory(),
                                 save=True, single_file=True)
    assert True

