import sys
import torch
import logging

from src.enums.adaptations import EAdaptations
from src.enums.problems import EProblems

# Used only if not specified in the PINN training run
PROBLEM = EProblems.DIFFUSION
ADAPTATION = EAdaptations.NO_ADAPTATION

# Collocation points limits
NUM_MAX_POINTS = 200  # DEF 200
NUM_TEST_POINTS = 20  # DEF 20

# PINN settings
LAYERS = 3  # DEF 3
NEURONS = 15  # DEF 15
LEARNING_RATE = 0.005  # DEF 0.005

# Problem/adaptation specific values
EPSILON = 0.1  # DEF 0.1, required for the Advection Diffusion problem
MAX_DEPTH = 10  # DEF 10, required for mesh adaptation
NUM_BASE_MESH_POINTS = 20  # DEF 20, required for Middle points adaptation

# RUN settings
TOLERANCE = 1e-4  # DEF 1e-4, error tolerance
NUMBER_EPOCHS = 1000  # DEF 1000, number of EPOCHS per 1 adaptation iteration
MAX_ITERS = 100  # DEF 1000, maximum number of iterations for a run
LOG_LEVEL = logging.INFO  # DEF logging.INFO

# logger config
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s[%(levelname)s] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")
logging.log(logging.INFO, f"Using device: {DEVICE}")

# R3
DEFAULT_R3_MAX_ITERATIONS = 5