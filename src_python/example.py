# Testing the problem_generator function
import sys

if len(sys.argv) == 1:
    problem_file = 'petit_probleme.lp'
else:
    problem_file = []
    for i in range(1, len(sys.argv)):
        problem_file.append(sys.argv[i])
print('problem_file : ', problem_file)
sys.exit()