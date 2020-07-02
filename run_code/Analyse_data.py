import sys
from dataset import load_csv, load_csv_single_file
from DataAnalyser import DatasetAnalyser, OutputDataAnalyser, DataAnalyser
from DataProcessor import BoundProcessorSecondDeg
from OutputData import OutputData

if __name__ == '__main__':

    if False:

        path = sys.argv[1]
        bound_file_name = sys.argv[2]
        sol_file_name = sys.argv[3]

        data = load_csv(bound_file_name, sol_file_name, path=path)

        Analyser = DatasetAnalyser(data)
        Analyser.plot2D_sol_fct_of_RHS(save=True, name="test_plot")

    if True:
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

    if False:

        a = 4.9682e-06
        b = -0.00822419
        c = 2.47186
        dev = 39152400
        avg = 7538330000

        path = sys.argv[1]
        bound_file_name = sys.argv[2]
        sol_file_name = sys.argv[3]

        data = load_csv(bound_file_name, sol_file_name, path=path)
        new_data = data.copy()

        processor = BoundProcessorSecondDeg(a, b, c, dev, avg)
        processor.pre_process(data)

        for_plot = OutputData(data.get_solutions(), data.get_RHS())

        analyser = DataAnalyser(new_data, for_plot)
        analyser.performance_plot_2D(save=True, name="quadratic_interpolation.pdf")
