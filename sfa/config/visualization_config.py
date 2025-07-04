"""Visualization configuration for Semantic Flow Analyzer."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import plotly.colors as pcolors

@dataclass
class VisualizationConfig:
    """Main visualization configuration class."""
    
    # General settings
    theme: str = 'plotly_white'
    color_palette: str = 'viridis'
    figure_width: int = 1200
    figure_height: int = 800
    dpi: int = 300
    
    # Animation settings
    animation_enabled: bool = True
    animation_duration: int = 500
    transition_duration: int = 300
    frame_duration: int = 100
    
    # Interactive settings
    interactive_mode: bool = True
    hover_enabled: bool = True
    zoom_enabled: bool = True
    pan_enabled: bool = True
    selection_enabled: bool = True
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ['html', 'png', 'svg'])
    export_quality: str = 'high'
    export_transparent: bool = False
    
    # Layer-specific configurations
    word_layer_config: 'WordLayerConfig' = field(default_factory=lambda: WordLayerConfig())
    community_layer_config: 'CommunityLayerConfig' = field(default_factory=lambda: CommunityLayerConfig())
    system_layer_config: 'SystemLayerConfig' = field(default_factory=lambda: SystemLayerConfig())
    meta_layer_config: 'MetaLayerConfig' = field(default_factory=lambda: MetaLayerConfig())
    
    # Dashboard settings
    dashboard_config: 'DashboardConfig' = field(default_factory=lambda: DashboardConfig())

@dataclass
class WordLayerConfig:
    """Configuration for word-level visualizations."""
    
    # Trajectory settings
    trajectory_3d: bool = True
    trajectory_line_width: int = 3
    trajectory_marker_size: int = 5
    trajectory_opacity: float = 0.8
    trajectory_colorscale: str = 'Viridis'
    
    # Flow field settings
    flow_vector_scale: float = 0.3
    flow_vector_color: str = 'red'
    flow_vector_width: int = 2
    flow_grid_resolution: int = 10
    
    # Evolution heatmap settings
    heatmap_colorscale: str = 'RdYlBu_r'
    heatmap_aspect_ratio: str = 'auto'
    
    # Burst timeline settings
    burst_line_color: str = 'red'
    burst_line_width: int = 3
    burst_threshold_color: str = 'orange'
    burst_marker_size: int = 8

@dataclass
class CommunityLayerConfig:
    """Configuration for community-level visualizations."""
    
    # Network settings
    node_size_range: Tuple[int, int] = (10, 30)
    edge_width_range: Tuple[int, int] = (1, 5)
    node_color_attribute: str = 'community'
    edge_color: str = 'gray'
    network_layout: str = 'spring'
    
    # Evolution settings
    evolution_line_width: int = 3
    evolution_marker_size: int = 8
    evolution_opacity: float = 0.8
    
    # Overlap matrix settings
    overlap_colorscale: str = 'Blues'
    overlap_show_values: bool = True
    
    # Health metrics settings
    health_fill_opacity: float = 0.6
    health_line_width: int = 2

@dataclass
class SystemLayerConfig:
    """Configuration for system-level visualizations."""
    
    # Phase space settings
    phase_line_width: int = 3
    phase_marker_size: int = 4
    phase_colorscale: str = 'Viridis'
    phase_trajectory_opacity: float = 0.8
    
    # Cascade risk settings
    cascade_risk_color: str = 'red'
    cascade_r0_color: str = 'blue'
    cascade_threshold_color: str = 'orange'
    cascade_line_width: int = 3
    
    # Complexity metrics settings
    complexity_bar_color: str = 'lightblue'
    complexity_show_values: bool = True
    
    # Global flow settings
    inflow_color: str = 'green'
    outflow_color: str = 'red'
    flow_fill: bool = True
    flow_line_width: int = 3

@dataclass
class MetaLayerConfig:
    """Configuration for meta-system level visualizations."""
    
    # Long-term trends settings
    trend_line_width: int = 3
    trend_marker_size: int = 6
    trend_opacity: float = 0.8
    
    # Regime detection settings
    regime_stable_color: str = 'green'
    regime_transitional_color: str = 'orange'
    regime_critical_color: str = 'red'
    regime_show_thresholds: bool = True
    
    # Correlation matrix settings
    correlation_colorscale: str = 'RdBu'
    correlation_center_zero: bool = True
    
    # Predictive indicators settings
    prediction_current_color: str = 'blue'
    prediction_forecast_color: str = 'red'
    prediction_uncertainty_opacity: float = 0.3

@dataclass
class DashboardConfig:
    """Configuration for dashboard and multi-layer views."""
    
    # Layout settings
    dashboard_layout: str = 'grid'  # 'grid', 'tabs', 'accordion'
    grid_rows: int = 2
    grid_cols: int = 2
    subplot_spacing: float = 0.1
    
    # Title and labeling
    show_main_title: bool = True
    show_subplot_titles: bool = True
    title_font_size: int = 16
    subplot_title_font_size: int = 14
    
    # Legend settings
    show_legend: bool = True
    legend_position: str = 'right'
    legend_orientation: str = 'vertical'
    
    # Responsive settings
    responsive: bool = True
    mobile_friendly: bool = True
    auto_resize: bool = True

@dataclass
class FlowVisualizationConfig:
    """Specific configuration for flow visualizations."""
    
    # Flow particle settings
    particle_count: int = 100
    particle_size: int = 3
    particle_opacity: float = 0.7
    particle_trail_length: int = 10
    
    # Flow animation settings
    flow_speed: float = 1.0
    flow_direction_arrows: bool = True
    flow_magnitude_scaling: float = 1.0
    
    # Flow colorization
    flow_colorscale: str = 'Blues'
    flow_color_by: str = 'magnitude'  # 'magnitude', 'direction', 'coherence'
    
    # Vector field settings
    vector_density: int = 20
    vector_scale: float = 0.5
    vector_normalize: bool = True

@dataclass
class NetworkVisualizationConfig:
    """Configuration for network visualizations."""
    
    # Node settings
    node_size_metric: str = 'degree'  # 'degree', 'betweenness', 'closeness', 'pagerank'
    node_color_metric: str = 'community'
    node_opacity: float = 0.8
    node_border_width: int = 1
    
    # Edge settings
    edge_width_metric: str = 'weight'
    edge_opacity: float = 0.5
    edge_curve: float = 0.1
    show_edge_labels: bool = False
    
    # Layout algorithms
    layout_algorithm: str = 'force_atlas'  # 'spring', 'circular', 'force_atlas', 'fruchterman'
    layout_iterations: int = 50
    layout_seed: Optional[int] = None
    
    # Community visualization
    highlight_communities: bool = True
    community_colors: List[str] = field(default_factory=lambda: pcolors.qualitative.Set3)
    community_opacity: float = 0.3

@dataclass
class TimeSeriesVisualizationConfig:
    """Configuration for time series visualizations."""
    
    # Line settings
    line_width: int = 2
    marker_size: int = 6
    fill_area: bool = False
    
    # Trend analysis
    show_trend_line: bool = True
    trend_line_style: str = 'dash'
    trend_line_color: str = 'red'
    
    # Anomaly detection
    highlight_anomalies: bool = True
    anomaly_marker_size: int = 10
    anomaly_color: str = 'red'
    
    # Uncertainty bands
    show_confidence_bands: bool = True
    confidence_level: float = 0.95
    band_opacity: float = 0.3

@dataclass
class HeatmapVisualizationConfig:
    """Configuration for heatmap visualizations."""
    
    # Color settings
    colorscale: str = 'RdBu'
    reverse_colorscale: bool = False
    center_colorscale: bool = True
    
    # Display settings
    show_values: bool = True
    value_format: str = '.3f'
    font_size: int = 10
    
    # Clustering
    cluster_rows: bool = False
    cluster_cols: bool = False
    clustering_method: str = 'ward'

@dataclass
class UMAPVisualizationConfig:
    """Configuration for UMAP projections."""
    
    # UMAP parameters
    n_components: int = 3
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = 'cosine'
    
    # Visualization settings
    point_size: int = 5
    point_opacity: float = 0.7
    color_by: str = 'timestamp'  # 'timestamp', 'cluster', 'metric'
    
    # Animation settings
    animate_evolution: bool = True
    animation_frames: int = 50
    transition_smoothness: float = 0.8

# Color scheme definitions
COLOR_SCHEMES = {
    'default': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'accent': '#2ca02c',
        'warning': '#d62728',
        'info': '#9467bd'
    },
    'semantic_flow': {
        'flow_positive': '#2ca02c',
        'flow_negative': '#d62728',
        'flow_neutral': '#1f77b4',
        'burst_event': '#ff7f0e',
        'phase_transition': '#9467bd'
    },
    'complexity': {
        'stable': '#2ca02c',
        'transitional': '#ff7f0e',
        'critical': '#d62728',
        'chaotic': '#9467bd',
        'ordered': '#1f77b4'
    }
}

# Default color scales for different visualization types
DEFAULT_COLORSCALES = {
    'flow': 'Blues',
    'burst': 'Reds',
    'cascade': 'Oranges',
    'community': 'Set3',
    'phase': 'Viridis',
    'correlation': 'RdBu',
    'evolution': 'Plasma'
}

# Pre-defined themes
THEMES = {
    'light': {
        'template': 'plotly_white',
        'background_color': 'white',
        'grid_color': 'lightgray',
        'text_color': 'black'
    },
    'dark': {
        'template': 'plotly_dark',
        'background_color': '#2f2f2f',
        'grid_color': '#444444',
        'text_color': 'white'
    },
    'minimal': {
        'template': 'simple_white',
        'background_color': 'white',
        'grid_color': 'lightgray',
        'text_color': 'black'
    },
    'scientific': {
        'template': 'plotly_white',
        'background_color': 'white',
        'grid_color': '#e6e6e6',
        'text_color': '#333333'
    }
}

# Utility functions for configuration
def create_default_config() -> VisualizationConfig:
    """Create default visualization configuration."""
    return VisualizationConfig()

def create_publication_config() -> VisualizationConfig:
    """Create configuration optimized for publications."""
    config = VisualizationConfig()
    config.theme = 'scientific'
    config.figure_width = 800
    config.figure_height = 600
    config.dpi = 300
    config.export_formats = ['pdf', 'svg', 'png']
    config.export_quality = 'high'
    config.animation_enabled = False
    return config

def create_interactive_config() -> VisualizationConfig:
    """Create configuration optimized for interactive exploration."""
    config = VisualizationConfig()
    config.animation_enabled = True
    config.interactive_mode = True
    config.hover_enabled = True
    config.zoom_enabled = True
    config.pan_enabled = True
    config.selection_enabled = True
    return config

def create_dashboard_config() -> VisualizationConfig:
    """Create configuration optimized for dashboards."""
    config = VisualizationConfig()
    config.figure_width = 400
    config.figure_height = 300
    config.dashboard_config.dashboard_layout = 'grid'
    config.dashboard_config.responsive = True
    config.dashboard_config.mobile_friendly = True
    return config

def get_color_scheme(scheme_name: str) -> Dict[str, str]:
    """Get predefined color scheme."""
    return COLOR_SCHEMES.get(scheme_name, COLOR_SCHEMES['default'])

def get_colorscale(viz_type: str) -> str:
    """Get default colorscale for visualization type."""
    return DEFAULT_COLORSCALES.get(viz_type, 'Viridis')

def apply_theme(config: VisualizationConfig, theme_name: str) -> VisualizationConfig:
    """Apply a predefined theme to configuration."""
    if theme_name in THEMES:
        theme = THEMES[theme_name]
        config.theme = theme['template']
        # Additional theme-specific modifications can be added here
    return config

# Configuration validation
def validate_config(config: VisualizationConfig) -> List[str]:
    """Validate visualization configuration and return list of issues."""
    issues = []
    
    # Check required fields
    if config.figure_width <= 0:
        issues.append("Figure width must be positive")
    
    if config.figure_height <= 0:
        issues.append("Figure height must be positive")
    
    if config.dpi <= 0:
        issues.append("DPI must be positive")
    
    # Check animation settings
    if config.animation_duration < 0:
        issues.append("Animation duration cannot be negative")
    
    # Check color schemes
    if config.color_palette not in pcolors.named_colorscales():
        issues.append(f"Unknown color palette: {config.color_palette}")
    
    # Check export formats
    valid_formats = ['html', 'png', 'jpg', 'jpeg', 'pdf', 'svg', 'webp']
    for fmt in config.export_formats:
        if fmt.lower() not in valid_formats:
            issues.append(f"Unsupported export format: {fmt}")
    
    return issues

# Configuration serialization
def config_to_dict(config: VisualizationConfig) -> Dict[str, Any]:
    """Convert configuration to dictionary."""
    import dataclasses
    return dataclasses.asdict(config)

def config_from_dict(config_dict: Dict[str, Any]) -> VisualizationConfig:
    """Create configuration from dictionary."""
    # Handle nested configurations
    if 'word_layer_config' in config_dict:
        config_dict['word_layer_config'] = WordLayerConfig(**config_dict['word_layer_config'])
    if 'community_layer_config' in config_dict:
        config_dict['community_layer_config'] = CommunityLayerConfig(**config_dict['community_layer_config'])
    if 'system_layer_config' in config_dict:
        config_dict['system_layer_config'] = SystemLayerConfig(**config_dict['system_layer_config'])
    if 'meta_layer_config' in config_dict:
        config_dict['meta_layer_config'] = MetaLayerConfig(**config_dict['meta_layer_config'])
    if 'dashboard_config' in config_dict:
        config_dict['dashboard_config'] = DashboardConfig(**config_dict['dashboard_config'])
    
    return VisualizationConfig(**config_dict)

# Default configuration instances
default_config = create_default_config()
publication_config = create_publication_config()
interactive_config = create_interactive_config()
dashboard_config = create_dashboard_config()