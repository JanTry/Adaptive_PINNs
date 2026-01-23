from src.enums.problems import Problems1D, Problems2D
from src.problems.problems_1D.diffusion import DiffusionProblem1D
from src.problems.problems_1D.p07_001 import P07001Problem1D
from src.problems.problems_1D.p07_01 import P0701Problem1D
from src.problems.problems_1D.tan_01 import Tan01Problem1D
from src.problems.problems_1D.tan_03 import Tan03Problem1D
from src.problems.problems_2D.p07_01_2D import P0701Problem2D
from src.problems.problems_2D.tan_05_2D import Tan05Problem2D
from src.problems.problems_2D.allen_cahn_2D import AllenCahnProblem2D
from src.problems.problems_2D.convection_2D import ConvectionProblem2D


def problem_factory_1D(problem: Problems1D):
    problem_classes = {
        Problems1D.DIFFUSION: DiffusionProblem1D,
        Problems1D.TAN_01: Tan01Problem1D,
        Problems1D.TAN_03: Tan03Problem1D,
        Problems1D.P07_01: P0701Problem1D,
        Problems1D.P07_001: P07001Problem1D,
    }

    return problem_classes[problem]()


def problem_factory_2D(problem: Problems2D):
    problem_classes = {
        Problems2D.TAN_05: Tan05Problem2D,
        Problems2D.P07_01: P0701Problem2D,
        Problems2D.ALLEN_CAHN: AllenCahnProblem2D,
        Problems2D.CONVECTION: ConvectionProblem2D,
    }

    return problem_classes[problem]()
