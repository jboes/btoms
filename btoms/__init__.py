# Copyright 2019
# (see accompanying license files for details).
"""Bayesian Optimization for atoms."""

from .opt import NEB
from .gp import GPCalculator, GaussianProcess
from .kernel import GRBF, GConstant, GWhiteNoise
from . import utils

__all__ = ['GaussianProcess', 'GPCalculator', 'NEB',
           'GRBF', 'GConstant', 'GWhiteNoise']
__version__ = '0.0.1'
