import os
import pytest
from Problem import Xpress_problem, Xpress_Problem_Factory
import xpress as xp


#class Xpress_problem_test:

def test_read_problem_from_file_fails():
    with pytest.raises(xp.SolverError):
        prob = Xpress_Problem_Factory().read_problem_from_file("petit_probleme.lp")


def test_read_problem_from_file():
    prob = Xpress_Problem_Factory().read_problem_from_file(os.path.join("..", "data", "petit_probleme.lp"))
    assert isinstance(prob, Xpress_problem), "problem creation from file failed."


def test_get_constraints_names():
    prob = Xpress_Problem_Factory().read_problem_from_file(os.path.join("..", "data", "petit_probleme.lp"))
    name_list = prob.get_constraint_names()
    assert (name_list[0], name_list[23], name_list[32]) == ("FlotMax_0,6", "demand_0", "flot_9"), "" \
        "Constraint name extraction failed, verify if solver has added spaces at the end of the variable names."


