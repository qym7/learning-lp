import sys
from dataset import load_csv, load_csv_single_file
from DataAnalyser import DatasetAnalyser

if __name__ == '__main__':

    if True:

        path = sys.argv[1]
        bound_file_name = sys.argv[2]
        sol_file_name = sys.argv[3]

        data = load_csv(bound_file_name, sol_file_name, path=path)

        Analyser = DatasetAnalyser(data)
        Analyser.plot2D_sol_fct_of_RHS(save=True, name="test_plot")

    if False:
        path = sys.argv[1]
        file_name = sys.argv[2]

        data = load_csv_single_file(file_name, path=path)

        Analyser = DatasetAnalyser(data)
        Analyser.plot2D_sol_fct_of_RHS()

    if False:
        path = 'D:\\repository\learning-lp\data\Generated_problems\problem_rte_1.lp'

        bound_file_name1 = "Nb=100_dev=0_C1_RHS.csv"
        sol_file_name1 = "Nb=100_dev=0_C1_sol.csv"

        data1 = load_csv(bound_file_name1, sol_file_name1, path)

        bound_file_name2 = "the_same_again_RHS.csv"
        sol_file_name2 = "the_same_again_sol.csv"

        data2 = load_csv(bound_file_name1, sol_file_name1, path)

        data1.merge(data2)

        print(data1.size())

        Analyser = DatasetAnalyser(data1)
        Analyser.plot2D_sol_fct_of_RHS()
