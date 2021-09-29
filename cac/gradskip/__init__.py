# TODO: Review and modify comments
"""
:mod:`torch.cac.gradskip` is a package implementing optimization algorithm with cac-gradient-skip.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from .cac_sgd import SGD
from torch.optim.optimizer import Optimizer, required
