from .solver import BaseSolver
from .det_solver import DetSolver
from .utils import output_to_smiles
from .chemistry import _verify_chirality
from typing import Dict 
TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
}