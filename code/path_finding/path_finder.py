from code.path_finding.problem import DummyProblem
from code.path_finding.search import aStarSearch


def find_path():
    problem = DummyProblem()
    result = aStarSearch(problem)
    print(result)


if __name__ == "__main__":
    find_path()