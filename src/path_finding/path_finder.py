from src.path_finding.problem import Level1ProblemBuilder
from src.path_finding.search import aStarSearch
from src.path_finding.csv_creator import CsvCreatorBuilder
import time


def find_path():
    before = time.time()
    problem = Level1ProblemBuilder().build()
    solution = aStarSearch(problem)
    print(solution)
    print("Time: ", time.time() - before)
    creator = CsvCreatorBuilder(result_path='data/results/test_result.csv', solution=solution).build()
    creator.create_csv()


if __name__ == "__main__":
    find_path()