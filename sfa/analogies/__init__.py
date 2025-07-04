"""Theoretical analogy modules for semantic flow analysis."""

from .base_analogy import BaseAnalogy
from .epidemic import EpidemicAnalogy
from .ferromagnetic import FerromagneticAnalogy
from .evolutionary import EvolutionaryAnalogy
from .bounded_confidence import BoundedConfidenceAnalogy
from .ensemble import AnalogyEnsemble

__all__ = [
    'BaseAnalogy',
    'EpidemicAnalogy',
    'FerromagneticAnalogy', 
    'EvolutionaryAnalogy',
    'BoundedConfidenceAnalogy',
    'AnalogyEnsemble'
]