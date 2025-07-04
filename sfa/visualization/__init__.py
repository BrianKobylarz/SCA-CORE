"""Visualization modules for semantic flow analysis."""

from .layer_manager import VisualizationLayerManager, VisualizationLayer

# Import other modules if they exist
try:
    from .flow_network import FlowNetworkVisualizer
except ImportError:
    FlowNetworkVisualizer = None

try:
    from .umap_projector import UMAPProjector
except ImportError:
    UMAPProjector = None

try:
    from .dashboard import StreamlitDashboard
except ImportError:
    StreamlitDashboard = None

__all__ = [
    'VisualizationLayerManager', 'VisualizationLayer'
]

# Add optional imports to __all__ if they exist
if FlowNetworkVisualizer is not None:
    __all__.append('FlowNetworkVisualizer')
if UMAPProjector is not None:
    __all__.append('UMAPProjector')
if StreamlitDashboard is not None:
    __all__.append('StreamlitDashboard')