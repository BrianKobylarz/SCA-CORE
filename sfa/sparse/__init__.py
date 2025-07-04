"""Sparse data handling modules for Reddit-style event-driven data."""

from .interpolation import (
    TemporalInterpolation, AdaptiveInterpolation, SemanticConsistencyInterpolation,
    interpolate_missing_data, create_uniform_timeline
)
from .confidence_bands import (
    ConfidenceBandEstimator, estimate_trajectory_uncertainty, 
    compare_uncertainty_methods, detect_high_uncertainty_periods
)

# Import other modules if they exist
try:
    from .adaptive_windows import AdaptiveTemporalWindows
except ImportError:
    AdaptiveTemporalWindows = None

try:
    from .neighbor_expansion import AdaptiveNeighborExpansion
except ImportError:
    AdaptiveNeighborExpansion = None

__all__ = [
    'TemporalInterpolation', 'AdaptiveInterpolation', 'SemanticConsistencyInterpolation',
    'interpolate_missing_data', 'create_uniform_timeline',
    'ConfidenceBandEstimator', 'estimate_trajectory_uncertainty', 
    'compare_uncertainty_methods', 'detect_high_uncertainty_periods'
]

# Add optional imports to __all__ if they exist
if AdaptiveTemporalWindows is not None:
    __all__.append('AdaptiveTemporalWindows')
if AdaptiveNeighborExpansion is not None:
    __all__.append('AdaptiveNeighborExpansion')