"""Semantic Flow Analyzer - A comprehensive system for analyzing semantic dynamics in textual data.

This package provides tools for:
- Tracking semantic flows and changes over time
- Analyzing complex dynamics using theoretical analogies
- Detecting critical transitions and phase changes
- Visualizing semantic evolution patterns
- Generating comprehensive reports and insights

Main modules:
- core: Core data structures and base classes
- dynamics: Flow tracking and burst detection
- metrics: Complexity and flow metrics
- analogies: Theoretical analogies (epidemic, ferromagnetic, evolutionary, etc.)
- sparse: Sparse data handling for event-driven data
- visualization: Interactive visualizations and dashboards
- analysis: Main orchestrators and report generators
- io: Data loading, export, and alignment utilities
- config: Configuration management
"""

# Core imports (always available)
from .core import (
    Word, Timestamp, Embedding, SemanticFlow, FlowEvent, CommunityLineage, PhaseTransition,
    BaseAnalyzer, TemporalEmbeddingStore, FlowManager
)

# Configuration imports (always available)
from .config import FlowConfig, ComplexityConfig, VisualizationConfig, Settings

# I/O imports (available)
from .io import RedditDataLoader, CacheManager

# Sparse data imports (available)  
from .sparse import TemporalInterpolation, ConfidenceBandEstimator

# Visualization imports (available)
from .visualization import VisualizationLayerManager

# Optional imports with fallbacks
try:
    from .analysis import SemanticFlowAnalyzer
except ImportError:
    import warnings
    warnings.warn("SemanticFlowAnalyzer could not be imported", ImportWarning)
    SemanticFlowAnalyzer = None

try:
    from .analysis import ReportGenerator, InterventionAdvisor, EventCorrelator
except ImportError:
    ReportGenerator = InterventionAdvisor = EventCorrelator = None

try:
    from .dynamics import SemanticFlowTracker, BurstDetector, CascadeAnalyzer
except ImportError:
    SemanticFlowTracker = BurstDetector = CascadeAnalyzer = None

try:
    from .dynamics import CommunityEvolutionTracker, PhaseTransitionDetector
except ImportError:
    CommunityEvolutionTracker = PhaseTransitionDetector = None

try:
    from .analogies import BaseAnalogy, EpidemicAnalogy, FerromagneticAnalogy
except ImportError:
    BaseAnalogy = EpidemicAnalogy = FerromagneticAnalogy = None

try:
    from .analogies import EvolutionaryAnalogy, BoundedConfidenceAnalogy, AnalogyEnsemble
except ImportError:
    EvolutionaryAnalogy = BoundedConfidenceAnalogy = AnalogyEnsemble = None

try:
    from .visualization import FlowNetworkVisualizer, UMAPProjector, StreamlitDashboard
except ImportError:
    FlowNetworkVisualizer = UMAPProjector = StreamlitDashboard = None

try:
    from .io import ExportManager, EmbeddingAlignment
except ImportError:
    ExportManager = EmbeddingAlignment = None

__version__ = "0.1.0"
__author__ = "Semantic Flow Analyzer Team"
__description__ = "Comprehensive semantic flow analysis system bridging NLP with complexity science"

# Build __all__ list dynamically
__all__ = [
    # Core types and classes (always available)
    'Word', 'Timestamp', 'Embedding', 'SemanticFlow', 'FlowEvent', 'CommunityLineage', 'PhaseTransition',
    'BaseAnalyzer', 'TemporalEmbeddingStore', 'FlowManager',
    
    # Configuration (always available)
    'FlowConfig', 'ComplexityConfig', 'VisualizationConfig', 'Settings',
    
    # I/O (available)
    'RedditDataLoader', 'CacheManager',
    
    # Sparse data (available)
    'TemporalInterpolation', 'ConfidenceBandEstimator',
    
    # Visualization (available)
    'VisualizationLayerManager'
]

# Add optional imports to __all__ if they exist
optional_imports = [
    'SemanticFlowAnalyzer', 'ReportGenerator', 'InterventionAdvisor', 'EventCorrelator',
    'SemanticFlowTracker', 'BurstDetector', 'CascadeAnalyzer', 'CommunityEvolutionTracker', 'PhaseTransitionDetector',
    'BaseAnalogy', 'EpidemicAnalogy', 'FerromagneticAnalogy', 'EvolutionaryAnalogy', 
    'BoundedConfidenceAnalogy', 'AnalogyEnsemble',
    'FlowNetworkVisualizer', 'UMAPProjector', 'StreamlitDashboard',
    'ExportManager', 'EmbeddingAlignment'
]

for import_name in optional_imports:
    if globals().get(import_name) is not None:
        __all__.append(import_name)