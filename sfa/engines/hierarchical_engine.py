"""Hierarchical visualization engine with 6-layer gravitational field system."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import networkx as nx

from ..core.base import BaseAnalyzer
from ..core.storage import HDF5EmbeddingStore
from ..core.gravity import HighDimensionalFlowMetrics, SemanticGravityField
from ..visualization.gravity_viz import UMAPVisualizationLayer
from ..config.gravity_config import GravityAnalysisConfig

logger = logging.getLogger(__name__)

class HierarchicalVisualizationEngine(BaseAnalyzer):
    """
    Six-layer hierarchical visualization system for gravitational semantic fields.
    
    Layers:
    1. Density Field - Background semantic density with gravitational potential
    2. Motion Trails - Historical trajectories of words through semantic space
    3. Orbital Mechanics - Current gravitational forces and orbital paths
    4. Communities - Semantic communities with convex hull boundaries
    5. Burst Events - Sudden gravitational field disruptions
    6. Flow Field - Global semantic flow patterns and currents
    """
    
    def __init__(self, storage: HDF5EmbeddingStore, flow_metrics: HighDimensionalFlowMetrics,
                 umap_viz: UMAPVisualizationLayer, config: GravityAnalysisConfig):
        super().__init__("HierarchicalVisualizationEngine")
        
        self.storage = storage
        self.flow_metrics = flow_metrics
        self.umap_viz = umap_viz
        self.config = config
        
        # Layer management
        self.layers = {
            'density_field': DensityFieldLayer(self),
            'motion_trails': MotionTrailsLayer(self),
            'orbital_mechanics': OrbitalMechanicsLayer(self),
            'communities': CommunitiesLayer(self),
            'burst_events': BurstEventsLayer(self),
            'flow_field': FlowFieldLayer(self)
        }
        
        # State management
        self.current_layers = set(config.hierarchical_viz.enabled_layers)
        self.layer_data = {}
        
    def create_complete_visualization(self, focal_word: str, timestamp: str,
                                    k_neighbors: int = 50) -> go.Figure:
        """
        Create complete hierarchical visualization with all enabled layers.
        
        Layers are composited in order from background to foreground.
        """
        logger.info(f"Creating hierarchical visualization for {focal_word}@{timestamp}")
        
        # Initialize base figure
        fig = go.Figure()
        
        # Ensure UMAP projection exists
        if timestamp not in self.umap_viz.projections:
            self.umap_viz.fit_umap_for_timestamp(timestamp)
        
        # Get gravitational field data
        gravity_field = self.flow_metrics.compute_gravitational_field(
            focal_word, timestamp, k_neighbors
        )
        
        if gravity_field is None:
            return self._create_error_figure(f"No gravitational field for {focal_word}@{timestamp}")
        
        # Generate each layer in order (background to foreground)
        layer_order = [
            'density_field',
            'flow_field', 
            'motion_trails',
            'communities',
            'orbital_mechanics',
            'burst_events'
        ]
        
        layer_traces = {}
        
        for layer_name in layer_order:
            if layer_name in self.current_layers and layer_name in self.layers:
                logger.debug(f"Generating layer: {layer_name}")
                
                try:
                    traces = self.layers[layer_name].generate_traces(
                        focal_word, timestamp, gravity_field
                    )
                    layer_traces[layer_name] = traces
                    
                    # Add traces to figure
                    for trace in traces:
                        fig.add_trace(trace)
                        
                except Exception as e:
                    logger.warning(f"Failed to generate {layer_name} layer: {e}")
        
        # Configure layout
        self._configure_layout(fig, focal_word, timestamp)
        
        # Store layer data for inspection
        self.layer_data[f"{focal_word}@{timestamp}"] = {
            'gravity_field': gravity_field,
            'layer_traces': layer_traces,
            'timestamp': timestamp
        }
        
        return fig
    
    def create_temporal_comparison(self, focal_word: str, timestamps: List[str],
                                 k_neighbors: int = 50) -> go.Figure:
        """
        Create side-by-side comparison of gravitational fields across timestamps.
        """
        n_timestamps = len(timestamps)
        cols = min(3, n_timestamps)
        rows = (n_timestamps + cols - 1) // cols
        
        subplot_titles = [f"{focal_word} @ {ts}" for ts in timestamps]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)],
            horizontal_spacing=0.02,
            vertical_spacing=0.02
        )
        
        for i, timestamp in enumerate(timestamps):
            row = i // cols + 1
            col = i % cols + 1
            
            # Create single timestamp visualization
            single_fig = self.create_complete_visualization(focal_word, timestamp, k_neighbors)
            
            # Add traces to subplot
            for trace in single_fig.data:
                trace.showlegend = False  # Avoid legend clutter
                fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(
            title=f"Temporal Evolution: {focal_word}",
            height=400 * rows,
            showlegend=True
        )
        
        return fig
    
    def toggle_layer(self, layer_name: str) -> bool:
        """Toggle layer visibility."""
        if layer_name in self.layers:
            if layer_name in self.current_layers:
                self.current_layers.remove(layer_name)
                return False
            else:
                self.current_layers.add(layer_name)
                return True
        return False
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about all layers."""
        return {
            layer_name: {
                'enabled': layer_name in self.current_layers,
                'description': layer.description,
                'settings': layer.get_settings()
            }
            for layer_name, layer in self.layers.items()
        }
    
    def _configure_layout(self, fig: go.Figure, focal_word: str, timestamp: str) -> None:
        """Configure the 3D layout for hierarchical visualization."""
        fig.update_layout(
            title=f"Gravitational Semantic Field: {focal_word} @ {timestamp}",
            scene=dict(
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                zaxis_title="UMAP Dimension 3",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(255,255,255,0.2)',
                    showbackground=True,
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                yaxis=dict(
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(255,255,255,0.2)',
                    showbackground=True,
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                zaxis=dict(
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(255,255,255,0.2)',
                    showbackground=True,
                    zerolinecolor='rgba(255,255,255,0.2)'
                )
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            height=800,
            margin=dict(l=0, r=0, t=50, b=0)
        )
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create error figure."""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error: {error_message}",
            showarrow=False,
            font=dict(size=16, color="red"),
            xref="paper", yref="paper"
        )
        fig.update_layout(title="Visualization Error")
        return fig


class BaseVisualizationLayer:
    """Base class for visualization layers."""
    
    def __init__(self, engine: HierarchicalVisualizationEngine):
        self.engine = engine
        self.description = "Base visualization layer"
        self.opacity = 1.0
        self.enabled = True
    
    def generate_traces(self, focal_word: str, timestamp: str, 
                       gravity_field: SemanticGravityField) -> List[go.Scatter3d]:
        """Generate plotly traces for this layer."""
        raise NotImplementedError
    
    def get_settings(self) -> Dict[str, Any]:
        """Get layer settings."""
        return {
            'opacity': self.opacity,
            'enabled': self.enabled
        }


class DensityFieldLayer(BaseVisualizationLayer):
    """Layer 1: Background semantic density field with gravitational potential."""
    
    def __init__(self, engine):
        super().__init__(engine)
        self.description = "Background semantic density field showing gravitational potential"
    
    def generate_traces(self, focal_word: str, timestamp: str,
                       gravity_field: SemanticGravityField) -> List[go.Scatter3d]:
        """Generate density field visualization."""
        config = self.engine.config.hierarchical_viz
        
        if timestamp not in self.engine.umap_viz.projections:
            return []
        
        projection_data = self.engine.umap_viz.projections[timestamp]
        positions = projection_data['projections']
        
        # Create 3D grid for density field
        resolution = config.density_resolution
        x_range = np.linspace(positions[:, 0].min(), positions[:, 0].max(), resolution)
        y_range = np.linspace(positions[:, 1].min(), positions[:, 1].max(), resolution)
        z_range = np.linspace(positions[:, 2].min(), positions[:, 2].max(), resolution)
        
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # Compute density at each grid point
        density = np.zeros_like(X)
        
        for i, pos in enumerate(positions):
            # Gaussian kernel around each word position
            dist_sq = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
            density += np.exp(-dist_sq / (2 * config.density_smoothing**2))
        
        # Apply smoothing
        density = gaussian_filter(density, sigma=1.0)
        
        # Create isosurface for density field
        traces = []
        
        # Multiple density levels
        density_levels = [0.1, 0.3, 0.5]
        colors = ['rgba(100,100,255,0.1)', 'rgba(150,150,255,0.15)', 'rgba(200,200,255,0.2)']
        
        for level, color in zip(density_levels, colors):
            if density.max() > level:
                # Sample points at density level
                mask = density > level
                sample_indices = np.where(mask)
                
                if len(sample_indices[0]) > 0:
                    # Subsample for performance
                    n_samples = min(1000, len(sample_indices[0]))
                    sample_idx = np.random.choice(len(sample_indices[0]), n_samples, replace=False)
                    
                    x_samples = X[sample_indices][sample_idx]
                    y_samples = Y[sample_indices][sample_idx]
                    z_samples = Z[sample_indices][sample_idx]
                    
                    trace = go.Scatter3d(
                        x=x_samples,
                        y=y_samples,
                        z=z_samples,
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=color,
                            opacity=0.1
                        ),
                        name=f'Density {level}',
                        showlegend=False,
                        hoverinfo='skip'
                    )
                    traces.append(trace)
        
        return traces


class MotionTrailsLayer(BaseVisualizationLayer):
    """Layer 2: Historical trajectories showing word movement through semantic space."""
    
    def __init__(self, engine):
        super().__init__(engine)
        self.description = "Historical trajectories of words through semantic space"
    
    def generate_traces(self, focal_word: str, timestamp: str,
                       gravity_field: SemanticGravityField) -> List[go.Scatter3d]:
        """Generate motion trails for words."""
        config = self.engine.config.hierarchical_viz
        
        # Get historical timestamps
        all_timestamps = self.engine.storage.get_timestamps()
        current_idx = all_timestamps.index(timestamp) if timestamp in all_timestamps else -1
        
        if current_idx < config.trail_length:
            return []
        
        # Get historical timestamps for trail
        trail_timestamps = all_timestamps[max(0, current_idx - config.trail_length):current_idx + 1]
        
        traces = []
        
        # Create trail for focal word
        focal_trail = self._create_word_trail(focal_word, trail_timestamps, 'gold', 'Focal Trail')
        if focal_trail:
            traces.append(focal_trail)
        
        # Create trails for neighbor words
        for i, neighbor_word in enumerate(gravity_field.neighbor_words[:10]):  # Limit for performance
            color = f'rgba({50 + i*20}, {100 + i*15}, {200 - i*10}, 0.6)'
            trail = self._create_word_trail(neighbor_word, trail_timestamps, color, f'{neighbor_word} Trail')
            if trail:
                traces.append(trail)
        
        return traces
    
    def _create_word_trail(self, word: str, timestamps: List[str], color: str, name: str) -> Optional[go.Scatter3d]:
        """Create trail for a single word."""
        positions = []
        valid_timestamps = []
        
        for ts in timestamps:
            if ts in self.engine.umap_viz.projections:
                projection_data = self.engine.umap_viz.projections[ts]
                if word in projection_data['words']:
                    idx = projection_data['words'].index(word)
                    positions.append(projection_data['projections'][idx])
                    valid_timestamps.append(ts)
        
        if len(positions) < 2:
            return None
        
        positions = np.array(positions)
        
        # Create trail with opacity decay
        config = self.engine.config.hierarchical_viz
        opacities = [config.trail_decay ** (len(positions) - i - 1) for i in range(len(positions))]
        
        return go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            line=dict(
                color=color,
                width=3
            ),
            marker=dict(
                size=[3 + 2*op for op in opacities],
                color=color,
                opacity=opacities
            ),
            name=name,
            hovertemplate=f'<b>{word}</b><br>Time: %{{text}}<br>Position: (%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<extra></extra>',
            text=valid_timestamps,
            showlegend=True
        )


class OrbitalMechanicsLayer(BaseVisualizationLayer):
    """Layer 3: Current gravitational forces and orbital paths."""
    
    def __init__(self, engine):
        super().__init__(engine)
        self.description = "Current gravitational forces and orbital mechanics"
    
    def generate_traces(self, focal_word: str, timestamp: str,
                       gravity_field: SemanticGravityField) -> List[go.Scatter3d]:
        """Generate orbital mechanics visualization."""
        config = self.engine.config.hierarchical_viz
        
        if timestamp not in self.engine.umap_viz.projections:
            return []
        
        projection_data = self.engine.umap_viz.projections[timestamp]
        
        # Find focal word position
        focal_idx = None
        for i, word in enumerate(projection_data['words']):
            if word == focal_word:
                focal_idx = i
                break
        
        if focal_idx is None:
            return []
        
        focal_pos = projection_data['projections'][focal_idx]
        traces = []
        
        # Add focal word as central body
        traces.append(go.Scatter3d(
            x=[focal_pos[0]],
            y=[focal_pos[1]],
            z=[focal_pos[2]],
            mode='markers+text',
            marker=dict(
                size=25,
                color='gold',
                symbol='diamond',
                line=dict(width=3, color='orange')
            ),
            text=[focal_word],
            textposition='top center',
            name=f'Central: {focal_word}',
            hovertemplate=f'<b>{focal_word}</b><br>Central Gravitational Body<extra></extra>'
        ))
        
        # Add orbiting neighbors
        neighbor_positions = []
        gravitational_strengths = []
        stability_colors = []
        neighbor_words = []
        
        for i, neighbor_word in enumerate(gravity_field.neighbor_words):
            if neighbor_word in projection_data['words']:
                neighbor_idx = projection_data['words'].index(neighbor_word)
                pos = projection_data['projections'][neighbor_idx]
                
                neighbor_positions.append(pos)
                gravitational_strengths.append(gravity_field.gravitational_strengths[i])
                stability_colors.append(gravity_field.stability_indices[i])
                neighbor_words.append(neighbor_word)
        
        if neighbor_positions:
            neighbor_positions = np.array(neighbor_positions)
            
            # Size proportional to gravitational strength
            sizes = np.array(gravitational_strengths)
            sizes = 5 + 15 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8)
            
            traces.append(go.Scatter3d(
                x=neighbor_positions[:, 0],
                y=neighbor_positions[:, 1],
                z=neighbor_positions[:, 2],
                mode='markers+text',
                marker=dict(
                    size=sizes,
                    color=stability_colors,
                    colorscale='RdYlBu',
                    colorbar=dict(title="Orbital Stability"),
                    line=dict(width=1, color='darkblue'),
                    opacity=0.8
                ),
                text=neighbor_words,
                textposition='top center',
                name='Orbiting Words',
                hovertemplate='<b>%{text}</b><br>' +
                            'Gravitational Strength: %{marker.size:.3f}<br>' +
                            'Stability: %{marker.color:.3f}<extra></extra>'
            ))
            
            # Add gravitational force arrows
            if config.show_gravitational_arrows:
                arrow_traces = self._create_force_arrows(focal_pos, neighbor_positions, gravitational_strengths)
                traces.extend(arrow_traces)
            
            # Add orbital paths
            if config.show_orbital_paths:
                orbital_traces = self._create_orbital_paths(focal_pos, neighbor_positions)
                traces.extend(orbital_traces)
        
        return traces
    
    def _create_force_arrows(self, focal_pos: np.ndarray, neighbor_positions: np.ndarray,
                           strengths: np.ndarray) -> List[go.Scatter3d]:
        """Create gravitational force arrows."""
        traces = []
        
        for i, (neighbor_pos, strength) in enumerate(zip(neighbor_positions, strengths)):
            # Arrow from neighbor toward focal (gravitational attraction)
            direction = focal_pos - neighbor_pos
            direction_norm = direction / np.linalg.norm(direction)
            
            # Arrow length proportional to force
            max_length = 0.3
            arrow_length = max_length * (strength / strengths.max())
            arrow_end = neighbor_pos + direction_norm * arrow_length
            
            # Arrow color based on strength
            color_intensity = strength / strengths.max()
            color = f'rgba(255, {int(255 * (1 - color_intensity))}, 0, 0.7)'
            
            # Create arrow as line
            trace = go.Scatter3d(
                x=[neighbor_pos[0], arrow_end[0], None],
                y=[neighbor_pos[1], arrow_end[1], None],
                z=[neighbor_pos[2], arrow_end[2], None],
                mode='lines',
                line=dict(
                    width=max(2, int(8 * color_intensity)),
                    color=color
                ),
                showlegend=False,
                hoverinfo='skip',
                name='Gravitational Force'
            )
            traces.append(trace)
        
        return traces
    
    def _create_orbital_paths(self, focal_pos: np.ndarray, neighbor_positions: np.ndarray) -> List[go.Scatter3d]:
        """Create orbital path traces."""
        config = self.engine.config.hierarchical_viz
        traces = []
        
        for neighbor_pos in neighbor_positions:
            # Create circular orbit approximation
            radius = np.linalg.norm(neighbor_pos - focal_pos)
            
            # Generate circle points
            theta = np.linspace(0, 2*np.pi, config.orbital_path_segments)
            circle_x = focal_pos[0] + radius * np.cos(theta)
            circle_y = focal_pos[1] + radius * np.sin(theta)
            circle_z = np.full_like(circle_x, focal_pos[2])  # Simplified to horizontal orbit
            
            trace = go.Scatter3d(
                x=circle_x,
                y=circle_y,
                z=circle_z,
                mode='lines',
                line=dict(width=1, color='rgba(128, 128, 128, 0.3)', dash='dash'),
                showlegend=False,
                hoverinfo='skip',
                name='Orbital Path'
            )
            traces.append(trace)
        
        return traces


class CommunitiesLayer(BaseVisualizationLayer):
    """Layer 4: Semantic communities with convex hull boundaries."""
    
    def __init__(self, engine):
        super().__init__(engine)
        self.description = "Semantic communities with gravitational clustering"
    
    def generate_traces(self, focal_word: str, timestamp: str,
                       gravity_field: SemanticGravityField) -> List[go.Scatter3d]:
        """Generate community visualization."""
        config = self.engine.config.hierarchical_viz
        
        if timestamp not in self.engine.umap_viz.projections:
            return []
        
        projection_data = self.engine.umap_viz.projections[timestamp]
        
        # Detect communities using gravitational clustering
        communities = self._detect_gravitational_communities(gravity_field, projection_data)
        
        traces = []
        colors = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.3)', 'rgba(0,0,255,0.3)', 
                 'rgba(255,255,0,0.3)', 'rgba(255,0,255,0.3)', 'rgba(0,255,255,0.3)']
        
        for i, community in enumerate(communities):
            if len(community['words']) >= config.min_community_size:
                # Get community positions
                positions = []
                for word in community['words']:
                    if word in projection_data['words']:
                        idx = projection_data['words'].index(word)
                        positions.append(projection_data['projections'][idx])
                
                if len(positions) >= 4 and config.show_convex_hulls:
                    # Create convex hull
                    hull_trace = self._create_convex_hull(positions, colors[i % len(colors)])
                    if hull_trace:
                        traces.append(hull_trace)
                
                # Add community center marker
                if positions:
                    center = np.mean(positions, axis=0)
                    traces.append(go.Scatter3d(
                        x=[center[0]],
                        y=[center[1]],
                        z=[center[2]],
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color=colors[i % len(colors)],
                            symbol='circle',
                            opacity=0.8
                        ),
                        text=[f'Community {i+1}'],
                        textposition='top center',
                        name=f'Community {i+1} ({len(community["words"])} words)',
                        hovertemplate=f'<b>Community {i+1}</b><br>Size: {len(community["words"])}<br>Cohesion: {community["cohesion"]:.3f}<extra></extra>'
                    ))
        
        return traces
    
    def _detect_gravitational_communities(self, gravity_field: SemanticGravityField,
                                        projection_data: Dict) -> List[Dict]:
        """Detect communities based on gravitational clustering."""
        # Simple clustering based on gravitational strength and stability
        communities = []
        
        # Group words by similar gravitational properties
        strengths = gravity_field.gravitational_strengths
        stabilities = gravity_field.stability_indices
        
        # Use DBSCAN clustering on gravitational features
        features = np.column_stack([strengths, stabilities])
        
        try:
            clustering = DBSCAN(eps=0.3, min_samples=3).fit(features)
            labels = clustering.labels_
            
            for label in set(labels):
                if label != -1:  # Ignore noise points
                    mask = labels == label
                    community_words = [gravity_field.neighbor_words[i] for i in range(len(mask)) if mask[i]]
                    community_strengths = strengths[mask]
                    
                    community = {
                        'words': community_words,
                        'cohesion': np.mean(community_strengths),
                        'stability': np.mean(stabilities[mask]),
                        'size': len(community_words)
                    }
                    communities.append(community)
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
        
        return communities
    
    def _create_convex_hull(self, positions: List[np.ndarray], color: str) -> Optional[go.Mesh3d]:
        """Create convex hull mesh for community boundary."""
        try:
            positions = np.array(positions)
            hull = ConvexHull(positions)
            
            return go.Mesh3d(
                x=positions[hull.vertices, 0],
                y=positions[hull.vertices, 1],
                z=positions[hull.vertices, 2],
                i=hull.simplices[:, 0],
                j=hull.simplices[:, 1],
                k=hull.simplices[:, 2],
                color=color,
                opacity=self.engine.config.hierarchical_viz.hull_alpha,
                showlegend=False,
                hoverinfo='skip',
                name='Community Boundary'
            )
        except Exception as e:
            logger.warning(f"Failed to create convex hull: {e}")
            return None


class BurstEventsLayer(BaseVisualizationLayer):
    """Layer 5: Sudden gravitational field disruptions (bursts)."""
    
    def __init__(self, engine):
        super().__init__(engine)
        self.description = "Burst events and gravitational field disruptions"
    
    def generate_traces(self, focal_word: str, timestamp: str,
                       gravity_field: SemanticGravityField) -> List[go.Scatter3d]:
        """Generate burst event visualization."""
        # For current timestamp, we don't show bursts
        # Bursts are detected between timestamps
        return []


class FlowFieldLayer(BaseVisualizationLayer):
    """Layer 6: Global semantic flow patterns and currents."""
    
    def __init__(self, engine):
        super().__init__(engine)
        self.description = "Global semantic flow patterns and vector fields"
    
    def generate_traces(self, focal_word: str, timestamp: str,
                       gravity_field: SemanticGravityField) -> List[go.Scatter3d]:
        """Generate flow field visualization."""
        config = self.engine.config.hierarchical_viz
        
        if timestamp not in self.engine.umap_viz.projections:
            return []
        
        projection_data = self.engine.umap_viz.projections[timestamp]
        positions = projection_data['projections']
        
        # Create flow field grid
        x_range = np.linspace(positions[:, 0].min(), positions[:, 0].max(), config.flow_arrow_density)
        y_range = np.linspace(positions[:, 1].min(), positions[:, 1].max(), config.flow_arrow_density)
        z_range = np.linspace(positions[:, 2].min(), positions[:, 2].max(), config.flow_arrow_density)
        
        traces = []
        
        # Sample flow field at grid points
        for x in x_range[::2]:  # Subsample for performance
            for y in y_range[::2]:
                for z in z_range[::2]:
                    # Compute local flow direction (simplified)
                    flow_vector = self._compute_local_flow(np.array([x, y, z]), gravity_field, projection_data)
                    
                    if np.linalg.norm(flow_vector) > 0.01:  # Only show significant flows
                        # Create flow arrow
                        arrow_end = np.array([x, y, z]) + flow_vector * config.flow_arrow_scale
                        
                        trace = go.Scatter3d(
                            x=[x, arrow_end[0], None],
                            y=[y, arrow_end[1], None],
                            z=[z, arrow_end[2], None],
                            mode='lines',
                            line=dict(
                                width=2,
                                color='rgba(100, 100, 100, 0.5)'
                            ),
                            showlegend=False,
                            hoverinfo='skip',
                            name='Flow Field'
                        )
                        traces.append(trace)
        
        return traces[:50]  # Limit number of arrows for performance
    
    def _compute_local_flow(self, position: np.ndarray, gravity_field: SemanticGravityField,
                          projection_data: Dict) -> np.ndarray:
        """Compute local flow direction at a point."""
        # Simplified: flow points toward regions of higher semantic density
        nearby_words = []
        
        for i, word in enumerate(projection_data['words']):
            word_pos = projection_data['projections'][i]
            distance = np.linalg.norm(word_pos - position)
            
            if distance < 1.0:  # Within influence radius
                nearby_words.append((word, word_pos, distance))
        
        if not nearby_words:
            return np.zeros(3)
        
        # Flow toward center of mass of nearby words
        center_of_mass = np.mean([pos for _, pos, _ in nearby_words], axis=0)
        flow_direction = center_of_mass - position
        
        # Normalize
        norm = np.linalg.norm(flow_direction)
        if norm > 0:
            return flow_direction / norm * 0.1  # Small arrow scale
        
        return np.zeros(3)