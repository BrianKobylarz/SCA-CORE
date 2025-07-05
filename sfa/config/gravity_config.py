"""Configuration for gravitational field semantic analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

@dataclass
class GravitationalFieldConfig:
    """Configuration for gravitational field computation."""
    
    # Gravitational physics constants
    semantic_G: float = 1.0  # Gravitational constant for semantic space
    mass_scaling: str = "embedding_norm"  # How to compute semantic mass
    force_law: str = "inverse_square"  # Gravitational force law
    
    # Orbital mechanics
    min_orbit_radius: float = 0.01  # Minimum orbital distance
    max_orbit_radius: float = 10.0  # Maximum orbital distance
    stability_threshold: float = 0.5  # Orbital stability threshold
    
    # Neighbor selection
    k_neighbors: int = 50  # Number of orbiting neighbors
    neighbor_selection: str = "cosine_similarity"  # Method for finding neighbors
    neighbor_cache_size: int = 1000  # Cache size for neighbor computation
    
    # Burst detection
    burst_threshold: float = 2.0  # Threshold for detecting semantic bursts
    burst_window: int = 2  # Temporal window for burst detection
    cascade_detection: bool = True  # Enable cascade burst detection
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 10000
    parallel_computation: bool = True
    batch_size: int = 100

@dataclass
class UMAPVisualizationConfig:
    """Configuration for UMAP-based visualization."""
    
    # UMAP parameters
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 3
    metric: str = "cosine"
    random_state: int = 42
    
    # Vocabulary sampling
    max_vocabulary_size: int = 500  # Limit vocabulary for efficiency
    vocabulary_sampling: str = "random"  # How to sample large vocabularies
    
    # Animation settings
    animation_fps: int = 30  # Frames per second for temporal animation
    interpolation_method: str = "linear"  # Interpolation between timestamps
    smooth_transitions: bool = True
    
    # Visual styling
    focal_word_size: int = 20
    neighbor_size_range: tuple = (5, 15)
    arrow_width_range: tuple = (1, 5)
    opacity_settings: Dict[str, float] = field(default_factory=lambda: {
        'focal_word': 1.0,
        'neighbors': 0.8,
        'arrows': 0.7,
        'orbital_paths': 0.3
    })
    
    # Color schemes
    color_schemes: Dict[str, str] = field(default_factory=lambda: {
        'gravitational_strength': 'Viridis',
        'orbital_stability': 'RdYlBu',
        'semantic_drift': 'Plasma',
        'burst_intensity': 'Hot',
        'community_membership': 'Set3'
    })

@dataclass
class HierarchicalVisualizationConfig:
    """Configuration for multi-layer hierarchical visualization."""
    
    # Layer configuration
    enabled_layers: List[str] = field(default_factory=lambda: [
        'density_field',
        'motion_trails', 
        'orbital_mechanics',
        'communities',
        'burst_events',
        'flow_field'
    ])
    
    # Density field layer
    density_resolution: int = 50  # Grid resolution for density field
    density_smoothing: float = 0.1  # Gaussian smoothing parameter
    density_threshold: float = 0.01  # Minimum density to display
    
    # Motion trails layer
    trail_length: int = 10  # Number of historical positions to show
    trail_decay: float = 0.8  # Opacity decay for trail segments
    trail_interpolation: bool = True  # Smooth trail interpolation
    
    # Orbital mechanics layer
    show_orbital_paths: bool = True
    show_gravitational_arrows: bool = True
    show_escape_velocities: bool = False
    orbital_path_segments: int = 50  # Resolution of orbital paths
    
    # Community layer
    community_detection_method: str = "gravitational_clustering"
    min_community_size: int = 3
    show_convex_hulls: bool = True
    hull_alpha: float = 0.2  # Transparency of community boundaries
    
    # Burst events layer
    burst_visualization_style: str = "explosion"  # Style of burst visualization
    burst_duration_frames: int = 30  # How long to show burst effects
    show_cascade_effects: bool = True
    
    # Flow field layer
    flow_arrow_density: int = 20  # Number of flow arrows per dimension
    flow_arrow_scale: float = 1.0  # Scale factor for flow arrows
    show_flow_streamlines: bool = False

@dataclass
class TemporalNavigationConfig:
    """Configuration for temporal navigation and interpolation."""
    
    # Time navigation
    time_slider_steps: int = 100  # Number of steps in time slider
    auto_play_speed: float = 1.0  # Playback speed multiplier
    loop_animation: bool = True
    
    # Interpolation
    temporal_interpolation: bool = True
    interpolation_frames: int = 10  # Frames between actual timestamps
    interpolation_method: str = "cubic_spline"  # Interpolation algorithm
    extrapolation_enabled: bool = False  # Allow extrapolation beyond data
    
    # Caching
    cache_interpolated_frames: bool = True
    max_cached_frames: int = 1000
    precompute_interpolation: bool = False  # Precompute all frames

@dataclass
class StorageConfig:
    """Configuration for HDF5-based storage system."""
    
    # File settings
    compression: str = "gzip"  # HDF5 compression algorithm
    compression_level: int = 9  # Compression level (1-9)
    chunk_size: Optional[int] = None  # HDF5 chunk size
    
    # Caching
    cache_size: int = 10000  # Number of embeddings to cache
    cache_policy: str = "lru"  # Cache eviction policy
    preload_vocabulary: bool = True  # Preload vocabulary metadata
    
    # Performance
    parallel_io: bool = True  # Enable parallel I/O operations
    io_buffer_size: int = 1024 * 1024  # I/O buffer size in bytes
    lazy_loading: bool = True  # Enable lazy loading of embeddings
    
    # Backup and recovery
    backup_enabled: bool = False
    backup_interval: int = 3600  # Backup interval in seconds
    max_backup_files: int = 5

@dataclass
class GravityAnalysisConfig:
    """Master configuration combining all subsystem configurations."""
    
    # Subsystem configurations
    gravitational_field: GravitationalFieldConfig = field(default_factory=GravitationalFieldConfig)
    umap_visualization: UMAPVisualizationConfig = field(default_factory=UMAPVisualizationConfig)
    hierarchical_viz: HierarchicalVisualizationConfig = field(default_factory=HierarchicalVisualizationConfig)
    temporal_navigation: TemporalNavigationConfig = field(default_factory=TemporalNavigationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Global settings
    enable_logging: bool = True
    log_level: str = "INFO"
    performance_monitoring: bool = True
    
    # Experimental features
    enable_experimental: bool = False
    experimental_features: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any warnings."""
        warnings = []
        
        # Check gravitational field settings
        if self.gravitational_field.k_neighbors > 200:
            warnings.append("Large k_neighbors may impact performance")
        
        if self.gravitational_field.burst_threshold < 0.5:
            warnings.append("Low burst threshold may produce many false positives")
        
        # Check UMAP settings
        if self.umap_visualization.max_vocabulary_size < self.gravitational_field.k_neighbors:
            warnings.append("UMAP vocabulary size smaller than k_neighbors")
        
        # Check storage settings
        if self.storage.cache_size < self.gravitational_field.k_neighbors * 2:
            warnings.append("Storage cache may be too small for efficient operation")
        
        # Check temporal settings
        if (self.temporal_navigation.interpolation_frames > 50 and 
            self.temporal_navigation.cache_interpolated_frames):
            warnings.append("High interpolation frames with caching may use excessive memory")
        
        return warnings
    
    def get_performance_profile(self) -> str:
        """Get recommended performance profile based on settings."""
        if (self.gravitational_field.k_neighbors <= 25 and 
            self.umap_visualization.max_vocabulary_size <= 200):
            return "fast"
        elif (self.gravitational_field.k_neighbors <= 100 and 
              self.umap_visualization.max_vocabulary_size <= 1000):
            return "balanced"
        else:
            return "comprehensive"
    
    def optimize_for_performance(self) -> None:
        """Automatically optimize settings for performance."""
        profile = self.get_performance_profile()
        
        if profile == "fast":
            self.gravitational_field.k_neighbors = min(25, self.gravitational_field.k_neighbors)
            self.umap_visualization.max_vocabulary_size = min(200, self.umap_visualization.max_vocabulary_size)
            self.hierarchical_viz.density_resolution = min(30, self.hierarchical_viz.density_resolution)
            self.temporal_navigation.interpolation_frames = min(5, self.temporal_navigation.interpolation_frames)
        elif profile == "balanced":
            self.gravitational_field.k_neighbors = min(50, self.gravitational_field.k_neighbors)
            self.umap_visualization.max_vocabulary_size = min(500, self.umap_visualization.max_vocabulary_size)
            self.hierarchical_viz.density_resolution = min(50, self.hierarchical_viz.density_resolution)
            self.temporal_navigation.interpolation_frames = min(10, self.temporal_navigation.interpolation_frames)
        
        logger.info(f"Optimized configuration for {profile} performance profile")
    
    def optimize_for_quality(self) -> None:
        """Automatically optimize settings for maximum quality."""
        self.gravitational_field.k_neighbors = max(100, self.gravitational_field.k_neighbors)
        self.umap_visualization.max_vocabulary_size = max(1000, self.umap_visualization.max_vocabulary_size)
        self.hierarchical_viz.density_resolution = max(100, self.hierarchical_viz.density_resolution)
        self.temporal_navigation.interpolation_frames = max(20, self.temporal_navigation.interpolation_frames)
        self.temporal_navigation.interpolation_method = "cubic_spline"
        
        logger.info("Optimized configuration for maximum quality")

# Default configurations for common use cases
def create_fast_config() -> GravityAnalysisConfig:
    """Create configuration optimized for speed."""
    config = GravityAnalysisConfig()
    config.optimize_for_performance()
    return config

def create_quality_config() -> GravityAnalysisConfig:
    """Create configuration optimized for quality."""
    config = GravityAnalysisConfig()
    config.optimize_for_quality()
    return config

def create_demo_config() -> GravityAnalysisConfig:
    """Create configuration suitable for demonstrations."""
    config = GravityAnalysisConfig()
    
    # Moderate settings for demos
    config.gravitational_field.k_neighbors = 30
    config.umap_visualization.max_vocabulary_size = 200
    config.umap_visualization.animation_fps = 15
    config.hierarchical_viz.enabled_layers = [
        'orbital_mechanics', 'communities', 'burst_events'
    ]
    config.temporal_navigation.interpolation_frames = 5
    
    return config

def create_research_config() -> GravityAnalysisConfig:
    """Create configuration suitable for research analysis."""
    config = GravityAnalysisConfig()
    
    # Comprehensive settings for research
    config.gravitational_field.k_neighbors = 100
    config.gravitational_field.cascade_detection = True
    config.umap_visualization.max_vocabulary_size = 1000
    config.hierarchical_viz.enabled_layers = [
        'density_field', 'motion_trails', 'orbital_mechanics',
        'communities', 'burst_events', 'flow_field'
    ]
    config.temporal_navigation.interpolation_method = "cubic_spline"
    config.storage.backup_enabled = True
    config.performance_monitoring = True
    
    return config