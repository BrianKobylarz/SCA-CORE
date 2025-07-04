"""Flow-specific configuration for semantic analysis."""

from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class FlowConfig:
    """Configuration for semantic flow analysis"""
    
    # Flow tracking
    default_k_neighbors: int = 25
    flow_similarity_threshold: float = 0.3
    flow_trail_length: int = 5
    
    # Burst detection
    burst_z_threshold: float = 3.0
    burst_min_magnitude: float = 0.1
    burst_temporal_window: int = 7
    
    # Cascade analysis
    cascade_risk_threshold: float = 1.0
    contagion_min_neighbors: int = 3
    
    # Adaptive parameters
    min_neighbors_sparse: int = 5
    max_similarity_decay: float = 0.5
    activity_threshold: float = 0.2
    
    # Visualization
    flow_particle_count: int = 100
    flow_animation_speed: float = 1.0
    trail_fade_rate: float = 0.9
    
    # Reddit-specific
    subreddit_clustering: bool = True
    cross_subreddit_bridges: bool = True
    event_correlation: bool = True
    
    # Community detection
    min_community_size: int = 3
    
    # Additional parameters
    cascade_risk_threshold: float = 1.0

@dataclass
class ComplexityConfig:
    """Configuration for complexity science metrics"""
    
    # Stochasticity
    stochasticity_window: int = 5
    stochasticity_scaling: float = 4.0
    
    # Path dependency
    path_dependency_permutations: int = 100
    path_dependency_min_length: int = 3
    
    # Network contagion
    contagion_distance_threshold: float = 0.7
    contagion_min_affected: int = 2
    
    # Phase transitions
    phase_detection_window: int = 10
    phase_indicators: List[str] = None
    
    def __post_init__(self):
        if self.phase_indicators is None:
            self.phase_indicators = [
                'modularity', 'cascade_risk', 'graph_entropy', 
                'algebraic_connectivity', 'community_volatility'
            ]

@dataclass
class AnalogiesConfig:
    """Configuration for scientific analogies"""
    
    # Epidemic model
    epidemic_beta: float = 0.5  # Infection rate
    epidemic_gamma: float = 0.1  # Recovery rate
    epidemic_mu: float = 0.01   # Birth/death rate
    
    # Ferromagnetic model
    ferromagnetic_temperature: float = 1.0
    ferromagnetic_coupling: float = 1.0
    ferromagnetic_field: float = 0.0
    
    # Evolutionary model
    evolutionary_mutation_rate: float = 0.01
    evolutionary_selection_strength: float = 1.0
    evolutionary_drift_coefficient: float = 0.1
    
    # Bounded confidence model
    bounded_confidence_threshold: float = 0.3
    bounded_confidence_convergence: float = 0.1
    bounded_confidence_noise: float = 0.05
    
    # Integration settings
    ensemble_weights: Dict[str, float] = None
    ensemble_method: str = 'weighted_average'
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'epidemic': 0.3,
                'ferromagnetic': 0.3, 
                'evolutionary': 0.2,
                'bounded_confidence': 0.2
            }

@dataclass
class SparseDataConfig:
    """Configuration for sparse data handling"""
    
    # Adaptive windows
    base_window_days: int = 7
    min_window_days: int = 1
    max_window_days: int = 30
    activity_threshold: float = 0.2
    
    # Neighbor expansion
    neighbor_expansion_factor: float = 1.5
    max_neighbors: int = 100
    min_similarity: float = 0.1
    
    # Interpolation
    interpolation_method: str = 'linear'
    extrapolation_limit: int = 3
    confidence_threshold: float = 0.5
    
    # Uncertainty quantification
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    band_smoothing: float = 0.1

@dataclass
class VisualizationFlowConfig:
    """Flow-specific visualization configuration"""
    
    # 3D visualization
    umap_n_components: int = 3
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = 'cosine'
    
    # Flow animation
    animation_duration: int = 500
    transition_duration: int = 300
    particle_lifetime: int = 10
    
    # Network layout
    network_layout: str = 'spring'
    edge_bundling: bool = True
    node_sizing: str = 'centrality'
    
    # Color schemes
    flow_colorscale: str = 'Blues'
    burst_colorscale: str = 'Reds'
    cascade_colorscale: str = 'Viridis'
    
    # Interactive features
    hover_info: bool = True
    zoom_enabled: bool = True
    pan_enabled: bool = True
    selection_enabled: bool = True
    
    # Export settings
    export_format: str = 'html'
    export_width: int = 1200
    export_height: int = 800
    export_dpi: int = 300

# Default configurations
default_flow_config = FlowConfig()
default_complexity_config = ComplexityConfig()
default_analogies_config = AnalogiesConfig()
default_sparse_config = SparseDataConfig()
default_viz_flow_config = VisualizationFlowConfig()