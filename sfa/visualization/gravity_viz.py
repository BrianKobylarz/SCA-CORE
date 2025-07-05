"""UMAP-based visualization for gravitational semantic fields."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import umap
import logging
from typing import Dict, List, Optional, Tuple, Any
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

from ..core.types import Word, Timestamp
from ..core.base import BaseAnalyzer
from ..core.storage import HDF5EmbeddingStore
from ..core.gravity import HighDimensionalFlowMetrics, SemanticGravityField

logger = logging.getLogger(__name__)

class UMAPVisualizationLayer(BaseAnalyzer):
    """
    UMAP-based visualization layer that operates separately from metric computation.
    
    This class takes high-dimensional gravitational field data and projects it
    to 2D/3D space for interactive visualization while preserving the semantic
    relationships computed in the full embedding space.
    """
    
    def __init__(self, embedding_store: HDF5EmbeddingStore, 
                 flow_metrics: HighDimensionalFlowMetrics):
        super().__init__("UMAPVisualizationLayer")
        self.store = embedding_store
        self.flow_metrics = flow_metrics
        
        # UMAP models for each timestamp
        self.umap_models = {}
        self.projections = {}
        
        # Visualization settings
        self.umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': 3,
            'metric': 'cosine',
            'random_state': 42
        }
        
        # Color schemes
        self.color_schemes = {
            'gravitational_strength': 'Viridis',
            'orbital_stability': 'RdYlBu',
            'semantic_drift': 'Plasma',
            'burst_intensity': 'Hot'
        }
    
    def fit_umap_for_timestamp(self, timestamp: str, n_neighbors: int = 15,
                              vocabulary_limit: int = 500) -> Dict[str, Any]:
        """
        Fit UMAP model for a specific timestamp and project embeddings.
        
        Each timestamp gets its own UMAP model to preserve temporal consistency
        while allowing the semantic space to evolve naturally.
        """
        # Get vocabulary and embeddings
        vocab = self.store.get_vocabulary(timestamp)
        if len(vocab) > vocabulary_limit:
            # Sample vocabulary for efficiency
            vocab = np.random.choice(vocab, vocabulary_limit, replace=False).tolist()
        
        embeddings = self.store.get_embeddings_batch(vocab, timestamp)
        
        # Prepare embedding matrix
        valid_words = []
        embedding_matrix = []
        
        for word in vocab:
            if word in embeddings and embeddings[word] is not None:
                valid_words.append(word)
                embedding_matrix.append(embeddings[word])
        
        if len(valid_words) < 10:
            return {'error': f'Insufficient embeddings for {timestamp}'}
        
        embedding_matrix = np.array(embedding_matrix)
        
        # Fit UMAP
        umap_params = self.umap_params.copy()
        umap_params['n_neighbors'] = min(n_neighbors, len(valid_words) - 1)
        
        umap_model = umap.UMAP(**umap_params)
        projections = umap_model.fit_transform(embedding_matrix)
        
        # Store results
        self.umap_models[timestamp] = umap_model
        self.projections[timestamp] = {
            'words': valid_words,
            'embeddings': embedding_matrix,
            'projections': projections
        }
        
        return {
            'timestamp': timestamp,
            'num_words': len(valid_words),
            'projection_shape': projections.shape,
            'umap_params': umap_params
        }
    
    def create_gravitational_visualization(self, focal_word: str, timestamp: str,
                                         k_neighbors: int = 50) -> go.Figure:
        """
        Create interactive 3D visualization of gravitational semantic field.
        
        Shows the focal word at center with orbiting neighbors, gravitational
        forces as arrows, and orbital paths as curves.
        """
        # Get gravitational field data
        gravity_field = self.flow_metrics.compute_gravitational_field(
            focal_word, timestamp, k_neighbors
        )
        
        if gravity_field is None:
            return self._create_error_figure(f"No gravitational field data for {focal_word}@{timestamp}")
        
        # Ensure UMAP projection exists
        if timestamp not in self.projections:
            self.fit_umap_for_timestamp(timestamp)
        
        if timestamp not in self.projections:
            return self._create_error_figure(f"Failed to create UMAP projection for {timestamp}")
        
        projection_data = self.projections[timestamp]
        
        # Find focal word and neighbors in projection
        focal_idx = None
        neighbor_indices = {}
        
        for i, word in enumerate(projection_data['words']):
            if word == focal_word:
                focal_idx = i
            elif word in gravity_field.neighbor_words:
                neighbor_idx = gravity_field.neighbor_words.index(word)
                neighbor_indices[word] = (i, neighbor_idx)
        
        if focal_idx is None:
            return self._create_error_figure(f"Focal word {focal_word} not found in projection")
        
        # Create figure
        fig = go.Figure()
        
        # Get focal word position
        focal_pos = projection_data['projections'][focal_idx]
        
        # Add focal word (central gravitational body)
        fig.add_trace(go.Scatter3d(
            x=[focal_pos[0]],
            y=[focal_pos[1]],
            z=[focal_pos[2]],
            mode='markers+text',
            marker=dict(
                size=20,
                color='gold',
                symbol='diamond',
                line=dict(width=3, color='orange')
            ),
            text=[focal_word],
            textposition='top center',
            name=f'Focal: {focal_word}',
            hovertemplate=f'<b>{focal_word}</b><br>Central Gravitational Body<extra></extra>'
        ))
        
        # Add orbiting neighbors
        neighbor_positions = []
        gravitational_strengths = []
        stability_colors = []
        
        for word, (proj_idx, gravity_idx) in neighbor_indices.items():
            pos = projection_data['projections'][proj_idx]
            neighbor_positions.append(pos)
            gravitational_strengths.append(gravity_field.gravitational_strengths[gravity_idx])
            stability_colors.append(gravity_field.stability_indices[gravity_idx])
        
        if neighbor_positions:
            neighbor_positions = np.array(neighbor_positions)
            neighbor_words = list(neighbor_indices.keys())
            
            # Size proportional to gravitational strength
            sizes = np.array(gravitational_strengths)
            sizes = 5 + 15 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8)
            
            fig.add_trace(go.Scatter3d(
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
                            'Stability: %{marker.color:.3f}<br>' +
                            'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
            ))
            
            # Add gravitational force arrows
            arrow_data = self._create_gravitational_arrows(
                focal_pos, neighbor_positions, gravitational_strengths
            )
            
            for i, arrow in enumerate(arrow_data):
                fig.add_trace(go.Scatter3d(
                    x=arrow['x'],
                    y=arrow['y'],
                    z=arrow['z'],
                    mode='lines',
                    line=dict(
                        width=arrow['width'],
                        color=arrow['color']
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add orbital paths (simplified circles)
        orbital_traces = self._create_orbital_paths(focal_pos, neighbor_positions)
        for trace in orbital_traces:
            fig.add_trace(trace)
        
        # Update layout
        fig.update_layout(
            title=f"Gravitational Semantic Field: {focal_word} @ {timestamp}",
            scene=dict(
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                zaxis_title="UMAP Dimension 3",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_temporal_animation(self, focal_word: str, timestamps: List[str],
                                k_neighbors: int = 50, fps: int = 10) -> go.Figure:
        """
        Create animated visualization showing gravitational field evolution.
        
        Uses smooth interpolation between timestamps for fluid animation.
        """
        # Get interpolation data
        interp_data = self.flow_metrics.compute_temporal_interpolation(
            focal_word, timestamps, target_fps=fps
        )
        
        if 'error' in interp_data:
            return self._create_error_figure(interp_data['error'])
        
        # Fit UMAP for all timestamps
        for timestamp in timestamps:
            if timestamp not in self.projections:
                self.fit_umap_for_timestamp(timestamp)
        
        # Create animation frames
        frames = []
        
        for i, frame_data in enumerate(interp_data['interpolated_frames']):
            frame = self._create_animation_frame(focal_word, frame_data, i)
            frames.append(frame)
        
        # Create base figure with first frame
        fig = self.create_gravitational_visualization(focal_word, timestamps[0], k_neighbors)
        
        # Add animation frames
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"frame": {"duration": 100, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 50}}],
                            label="▶ Play",
                            method="animate"
                        ),
                        dict(
                            args=[{"frame": {"duration": 0, "redraw": True},
                                  "mode": "immediate", "transition": {"duration": 0}}],
                            label="⏸ Pause",
                            method="animate"
                        )
                    ]),
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.011,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=20),
                        prefix="Time: ",
                        visible=True,
                        xanchor="right"
                    ),
                    transition=dict(duration=50, easing="cubic-in-out"),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[[f"frame{i}"],
                                  {"frame": {"duration": 50, "redraw": True},
                                   "mode": "immediate",
                                   "transition": {"duration": 50}}],
                            label=f"Frame {i}",
                            method="animate"
                        ) for i in range(len(frames))
                    ]
                )
            ]
        )
        
        return fig
    
    def create_burst_visualization(self, focal_word: str, timestamps: List[str],
                                 k_neighbors: int = 50) -> go.Figure:
        """
        Create visualization highlighting burst events and their effects.
        
        Shows sudden changes in gravitational field as explosive events
        with affected regions highlighted.
        """
        # Detect bursts
        bursts = self.flow_metrics.detect_semantic_bursts(focal_word, timestamps, k_neighbors)
        
        if not bursts:
            return self._create_error_figure(f"No burst events detected for {focal_word}")
        
        # Create subplot for each burst
        n_bursts = len(bursts)
        cols = min(3, n_bursts)
        rows = (n_bursts + cols - 1) // cols
        
        subplot_titles = [f"Burst @ {burst.timestamp}" for burst in bursts]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)],
            horizontal_spacing=0.05,
            vertical_spacing=0.05
        )
        
        for i, burst in enumerate(bursts):
            row = i // cols + 1
            col = i % cols + 1
            
            # Create burst visualization
            burst_traces = self._create_burst_traces(burst)
            
            for trace in burst_traces:
                fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(
            title=f"Semantic Burst Events: {focal_word}",
            height=300 * rows,
            showlegend=False
        )
        
        return fig
    
    def compute_community_convex_hull(self, community_words: List[str], 
                                    timestamp: str) -> Optional[np.ndarray]:
        """
        Compute convex hull for a community of words in UMAP space.
        
        Used to visualize semantic communities as bounded regions.
        """
        if timestamp not in self.projections:
            self.fit_umap_for_timestamp(timestamp)
        
        if timestamp not in self.projections:
            return None
        
        projection_data = self.projections[timestamp]
        
        # Find community words in projection
        community_positions = []
        for word in community_words:
            if word in projection_data['words']:
                idx = projection_data['words'].index(word)
                community_positions.append(projection_data['projections'][idx])
        
        if len(community_positions) < 4:  # Need at least 4 points for 3D hull
            return None
        
        try:
            hull = ConvexHull(community_positions)
            return hull
        except Exception as e:
            logger.warning(f"Failed to compute convex hull: {e}")
            return None
    
    def _create_gravitational_arrows(self, focal_pos: np.ndarray, 
                                   neighbor_positions: np.ndarray,
                                   strengths: np.ndarray) -> List[Dict]:
        """Create arrow traces for gravitational forces."""
        arrows = []
        
        for i, (neighbor_pos, strength) in enumerate(zip(neighbor_positions, strengths)):
            # Arrow from neighbor to focal (gravitational attraction)
            direction = focal_pos - neighbor_pos
            direction_norm = direction / np.linalg.norm(direction)
            
            # Arrow length proportional to force
            max_length = 0.3
            arrow_length = max_length * (strength / strengths.max())
            
            arrow_end = neighbor_pos + direction_norm * arrow_length
            
            # Arrow color based on strength
            color_intensity = strength / strengths.max()
            color = f'rgba(255, {int(255 * (1 - color_intensity))}, 0, 0.7)'
            
            arrow = {
                'x': [neighbor_pos[0], arrow_end[0], None],
                'y': [neighbor_pos[1], arrow_end[1], None],
                'z': [neighbor_pos[2], arrow_end[2], None],
                'width': max(2, int(5 * color_intensity)),
                'color': color
            }
            arrows.append(arrow)
        
        return arrows
    
    def _create_orbital_paths(self, focal_pos: np.ndarray, 
                            neighbor_positions: np.ndarray) -> List[go.Scatter3d]:
        """Create orbital path traces around focal word."""
        traces = []
        
        for neighbor_pos in neighbor_positions:
            # Create circular orbit approximation
            radius = np.linalg.norm(neighbor_pos - focal_pos)
            
            # Generate circle points
            theta = np.linspace(0, 2*np.pi, 50)
            circle_x = focal_pos[0] + radius * np.cos(theta)
            circle_y = focal_pos[1] + radius * np.sin(theta)
            circle_z = np.full_like(circle_x, focal_pos[2])
            
            trace = go.Scatter3d(
                x=circle_x,
                y=circle_y,
                z=circle_z,
                mode='lines',
                line=dict(width=1, color='rgba(128, 128, 128, 0.3)', dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
            traces.append(trace)
        
        return traces
    
    def _create_animation_frame(self, focal_word: str, frame_data: Dict, 
                              frame_idx: int) -> go.Frame:
        """Create single animation frame."""
        # Simplified frame creation - in practice would interpolate positions
        traces = []
        
        # Add focal word
        traces.append(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=20, color='gold'),
            name=focal_word
        ))
        
        return go.Frame(
            data=traces,
            name=f"frame{frame_idx}",
            traces=[0, 1]  # Update specific traces
        )
    
    def _create_burst_traces(self, burst) -> List[go.Scatter3d]:
        """Create visualization traces for a burst event."""
        traces = []
        
        # Simplified burst visualization
        # Would show explosion effects, disrupted orbits, etc.
        trace = go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(
                size=30,
                color='red',
                symbol='circle',
                opacity=0.7
            ),
            name=f'Burst: {burst.burst_magnitude:.2f}'
        )
        traces.append(trace)
        
        return traces
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create error figure with message."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error: {error_message}",
            showarrow=False,
            font=dict(size=16, color="red"),
            xref="paper", yref="paper"
        )
        
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        
        return fig