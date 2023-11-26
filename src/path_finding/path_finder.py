from src.path_finding.problem import Level1ProblemBuilder
from src.path_finding.search import aStarSearch
import time


def find_path():
    before = time.time()
    problem = Level1ProblemBuilder().build()
    result = aStarSearch(problem)
    print(result)
    print("Time: ", time.time() - before)


if __name__ == "__main__":
    find_path()