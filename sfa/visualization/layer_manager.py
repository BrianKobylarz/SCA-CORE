"""Visualization layer manager for multi-scale semantic flow analysis."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime
from collections import defaultdict

from ..core.types import (
    Word, Timestamp, SemanticFlow, FlowEvent, WordTrajectory, 
    CommunityLineage, PhaseTransition
)
from ..core.base import BaseAnalyzer
from ..config.flow_config import VisualizationFlowConfig

logger = logging.getLogger(__name__)

class VisualizationLayer:
    """Individual visualization layer."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.figures = {}
        self.data = {}
        self.config = {}
        self.is_active = True
    
    def add_figure(self, figure_name: str, figure: go.Figure) -> None:
        """Add a figure to this layer."""
        self.figures[figure_name] = figure
    
    def get_figure(self, figure_name: str) -> Optional[go.Figure]:
        """Get a figure from this layer."""
        return self.figures.get(figure_name)
    
    def update_data(self, data: Dict[str, Any]) -> None:
        """Update layer data."""
        self.data.update(data)
    
    def configure(self, **config) -> None:
        """Configure layer settings."""
        self.config.update(config)

class VisualizationLayerManager(BaseAnalyzer):
    """
    Manages the four-layer visualization system:
    1. Word Level - Individual word trajectories and flows
    2. Community Level - Local semantic neighborhoods
    3. System Level - Global network properties
    4. Meta-System Level - Cross-temporal patterns
    """
    
    def __init__(self, analyzer, config: VisualizationFlowConfig):
        super().__init__("VisualizationLayerManager")
        self.analyzer = analyzer
        self.config = config
        
        # Initialize layers
        self.layers = {
            'word': VisualizationLayer(
                'Word Level',
                'Individual word trajectories and semantic flows'
            ),
            'community': VisualizationLayer(
                'Community Level', 
                'Local semantic neighborhoods and community evolution'
            ),
            'system': VisualizationLayer(
                'System Level',
                'Global network properties and phase transitions'
            ),
            'meta': VisualizationLayer(
                'Meta-System Level',
                'Cross-temporal patterns and long-term dynamics'
            )
        }
        
        # Color schemes for different layers
        self.color_schemes = {
            'word': px.colors.sequential.Blues,
            'community': px.colors.sequential.Greens, 
            'system': px.colors.sequential.Reds,
            'meta': px.colors.sequential.Viridis
        }
        
        # Animation settings
        self.animation_frames = []
        self.current_frame = 0
        
    def analyze(self) -> Dict[str, Any]:
        """Generate all visualization layers."""
        logger.info("Generating visualization layers")
        
        try:
            # Generate each layer
            word_results = self._generate_word_layer()
            community_results = self._generate_community_layer()
            system_results = self._generate_system_layer()
            meta_results = self._generate_meta_layer()
            
            # Create integrated dashboard
            dashboard = self._create_integrated_dashboard()
            
            return {
                'layer_visualizations': {
                    'word': word_results,
                    'community': community_results,
                    'system': system_results,
                    'meta': meta_results
                },
                'integrated_dashboard': dashboard,
                'animation_frames': len(self.animation_frames),
                'layers_active': len([l for l in self.layers.values() if l.is_active])
            }
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_word_layer(self) -> Dict[str, Any]:
        """Generate word-level visualizations."""
        logger.info("Generating word-level visualizations")
        
        layer = self.layers['word']
        
        # 1. Word trajectory plot
        trajectory_fig = self._create_word_trajectories()
        if trajectory_fig:
            layer.add_figure('trajectories', trajectory_fig)
        
        # 2. Flow vector field
        flow_field_fig = self._create_flow_vector_field()
        if flow_field_fig:
            layer.add_figure('flow_field', flow_field_fig)
        
        # 3. Word evolution heatmap
        evolution_fig = self._create_word_evolution_heatmap()
        if evolution_fig:
            layer.add_figure('evolution_heatmap', evolution_fig)
        
        # 4. Burst events timeline
        burst_fig = self._create_burst_timeline()
        if burst_fig:
            layer.add_figure('burst_timeline', burst_fig)
        
        return {
            'figures_created': len(layer.figures),
            'figure_names': list(layer.figures.keys()),
            'layer_data': layer.data
        }
    
    def _generate_community_layer(self) -> Dict[str, Any]:
        """Generate community-level visualizations."""
        logger.info("Generating community-level visualizations")
        
        layer = self.layers['community']
        
        # 1. Community network graph
        network_fig = self._create_community_network()
        if network_fig:
            layer.add_figure('network', network_fig)
        
        # 2. Community evolution timeline
        evolution_fig = self._create_community_evolution()
        if evolution_fig:
            layer.add_figure('evolution', evolution_fig)
        
        # 3. Community overlap matrix
        overlap_fig = self._create_community_overlap()
        if overlap_fig:
            layer.add_figure('overlap_matrix', overlap_fig)
        
        # 4. Community health metrics
        health_fig = self._create_community_health()
        if health_fig:
            layer.add_figure('health_metrics', health_fig)
        
        return {
            'figures_created': len(layer.figures),
            'figure_names': list(layer.figures.keys()),
            'layer_data': layer.data
        }
    
    def _generate_system_layer(self) -> Dict[str, Any]:
        """Generate system-level visualizations."""
        logger.info("Generating system-level visualizations")
        
        layer = self.layers['system']
        
        # 1. Phase space plot
        phase_fig = self._create_phase_space_plot()
        if phase_fig:
            layer.add_figure('phase_space', phase_fig)
        
        # 2. Cascade risk evolution
        cascade_fig = self._create_cascade_risk_plot()
        if cascade_fig:
            layer.add_figure('cascade_risk', cascade_fig)
        
        # 3. System complexity metrics
        complexity_fig = self._create_complexity_metrics()
        if complexity_fig:
            layer.add_figure('complexity_metrics', complexity_fig)
        
        # 4. Global flow patterns
        global_flow_fig = self._create_global_flow_patterns()
        if global_flow_fig:
            layer.add_figure('global_flows', global_flow_fig)
        
        return {
            'figures_created': len(layer.figures),
            'figure_names': list(layer.figures.keys()),
            'layer_data': layer.data
        }
    
    def _generate_meta_layer(self) -> Dict[str, Any]:
        """Generate meta-system level visualizations."""
        logger.info("Generating meta-system level visualizations")
        
        layer = self.layers['meta']
        
        # 1. Long-term trend analysis
        trend_fig = self._create_long_term_trends()
        if trend_fig:
            layer.add_figure('long_term_trends', trend_fig)
        
        # 2. Regime detection plot
        regime_fig = self._create_regime_detection()
        if regime_fig:
            layer.add_figure('regime_detection', regime_fig)
        
        # 3. Cross-temporal correlation
        correlation_fig = self._create_cross_temporal_correlation()
        if correlation_fig:
            layer.add_figure('correlation_matrix', correlation_fig)
        
        # 4. Predictive indicators
        prediction_fig = self._create_predictive_indicators()
        if prediction_fig:
            layer.add_figure('predictive_indicators', prediction_fig)
        
        return {
            'figures_created': len(layer.figures),
            'figure_names': list(layer.figures.keys()),
            'layer_data': layer.data
        }
    
    def _create_word_trajectories(self) -> Optional[go.Figure]:
        """Create 3D word trajectory visualization."""
        try:
            if not hasattr(self.analyzer, 'analysis_results') or not self.analyzer.analysis_results:
                return self._create_sample_trajectories()
            
            # Get UMAP projections if available
            umap_data = self.analyzer.analysis_results.get('umap_analysis', {})
            if 'temporal_trajectories' not in umap_data:
                return self._create_sample_trajectories()
            
            trajectories = umap_data['temporal_trajectories']
            
            fig = go.Figure()
            
            # Plot trajectory for each word
            for word, trajectory_data in trajectories.items():
                if 'positions' in trajectory_data:
                    positions = trajectory_data['positions']
                    
                    # Extract coordinates
                    x_coords = [pos[0] for pos in positions]
                    y_coords = [pos[1] for pos in positions] 
                    z_coords = [pos[2] if len(pos) > 2 else 0 for pos in positions]
                    timestamps = trajectory_data.get('timestamps', list(range(len(positions))))
                    
                    # Add trajectory line
                    fig.add_trace(go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='lines+markers',
                        name=word,
                        line=dict(width=3),
                        marker=dict(
                            size=5,
                            color=list(range(len(positions))),
                            colorscale='Viridis',
                            showscale=False
                        ),
                        text=[f"{word}<br>Time: {t}" for t in timestamps],
                        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
                    ))
            
            fig.update_layout(
                title="Word Trajectories in Semantic Space",
                scene=dict(
                    xaxis_title="UMAP Dimension 1",
                    yaxis_title="UMAP Dimension 2", 
                    zaxis_title="UMAP Dimension 3",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                showlegend=True,
                height=800
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create word trajectories: {e}")
            return self._create_sample_trajectories()
    
    def _create_sample_trajectories(self) -> go.Figure:
        """Create sample trajectory visualization."""
        fig = go.Figure()
        
        # Create sample trajectories
        for i, word in enumerate(['technology', 'innovation', 'digital', 'future'][:4]):
            t = np.linspace(0, 2*np.pi, 20)
            x = np.cos(t) + 0.2 * i
            y = np.sin(t) + 0.2 * i  
            z = t / (2*np.pi) + 0.1 * i
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name=word,
                line=dict(width=3),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Sample Word Trajectories",
            scene=dict(
                xaxis_title="Semantic Dimension 1",
                yaxis_title="Semantic Dimension 2",
                zaxis_title="Time",
            ),
            height=600
        )
        
        return fig
    
    def _create_flow_vector_field(self) -> Optional[go.Figure]:
        """Create flow vector field visualization."""
        try:
            # Create a grid for vector field
            x = np.linspace(-2, 2, 10)
            y = np.linspace(-2, 2, 10)
            X, Y = np.meshgrid(x, y)
            
            # Sample flow vectors (in practice, computed from actual flows)
            U = np.sin(X) * np.cos(Y)
            V = np.cos(X) * np.sin(Y)
            
            fig = go.Figure()
            
            # Add vector field
            fig.add_trace(go.Scatter(
                x=X.flatten(),
                y=Y.flatten(),
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightblue',
                    line=dict(width=1, color='darkblue')
                ),
                name='Flow Field',
                showlegend=False
            ))
            
            # Add arrows (simplified representation)
            for i in range(0, len(x), 2):
                for j in range(0, len(y), 2):
                    fig.add_annotation(
                        x=X[i,j], y=Y[i,j],
                        ax=X[i,j] + U[i,j]*0.3, ay=Y[i,j] + V[i,j]*0.3,
                        arrowhead=2, arrowsize=1, arrowwidth=2,
                        arrowcolor='red'
                    )
            
            fig.update_layout(
                title="Semantic Flow Vector Field",
                xaxis_title="Semantic Dimension 1",
                yaxis_title="Semantic Dimension 2",
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create flow vector field: {e}")
            return None
    
    def _create_word_evolution_heatmap(self) -> Optional[go.Figure]:
        """Create word evolution heatmap."""
        try:
            # Sample data for demonstration
            words = ['technology', 'innovation', 'digital', 'artificial', 'machine', 'learning']
            timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
            
            # Create sample evolution data (semantic change intensity)
            evolution_matrix = np.random.rand(len(words), len(timestamps))
            
            fig = go.Figure(data=go.Heatmap(
                z=evolution_matrix,
                x=timestamps,
                y=words,
                colorscale='RdYlBu_r',
                hoverongaps=True,
                colorbar=dict(title="Semantic Change Intensity")
            ))
            
            fig.update_layout(
                title="Word Evolution Heatmap",
                xaxis_title="Time Period",
                yaxis_title="Words",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create evolution heatmap: {e}")
            return None
    
    def _create_burst_timeline(self) -> Optional[go.Figure]:
        """Create burst events timeline."""
        try:
            # Sample burst events
            fig = go.Figure()
            
            # Sample data
            timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
            burst_intensities = [0.2, 0.8, 0.3, 1.2, 0.4]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=burst_intensities,
                mode='lines+markers',
                name='Burst Intensity',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            # Add threshold line
            fig.add_hline(y=1.0, line_dash="dash", line_color="orange", 
                         annotation_text="Burst Threshold")
            
            fig.update_layout(
                title="Semantic Burst Events Timeline",
                xaxis_title="Time Period",
                yaxis_title="Burst Intensity",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create burst timeline: {e}")
            return None
    
    def _create_community_network(self) -> Optional[go.Figure]:
        """Create community network visualization."""
        try:
            # Sample network data
            import networkx as nx
            
            G = nx.karate_club_graph()
            pos = nx.spring_layout(G)
            
            # Extract node and edge coordinates
            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode='markers+text',
                text=[f"Node {node}" for node in G.nodes()],
                textposition="middle center",
                marker=dict(
                    size=20,
                    color=list(range(len(G.nodes()))),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Community")
                ),
                hovertemplate='<b>%{text}</b><extra></extra>'
            )
            
            edge_trace = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    showlegend=False,
                    hoverinfo='none'
                ))
            
            fig = go.Figure(data=[node_trace] + edge_trace)
            fig.update_layout(
                title="Community Network Structure",
                showlegend=False,
                hovermode='closest',
                height=600,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create community network: {e}")
            return None
    
    def _create_community_evolution(self) -> Optional[go.Figure]:
        """Create community evolution timeline."""
        try:
            # Sample community evolution data
            timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
            communities = ['Tech', 'Business', 'Research', 'Social']
            
            fig = go.Figure()
            
            for i, community in enumerate(communities):
                sizes = np.random.rand(len(timestamps)) * 100 + 20
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=sizes,
                    mode='lines+markers',
                    name=community,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Community Size Evolution",
                xaxis_title="Time Period",
                yaxis_title="Community Size",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create community evolution: {e}")
            return None
    
    def _create_community_overlap(self) -> Optional[go.Figure]:
        """Create community overlap matrix."""
        try:
            communities = ['Tech', 'Business', 'Research', 'Social', 'Innovation']
            
            # Sample overlap matrix
            overlap_matrix = np.random.rand(len(communities), len(communities))
            np.fill_diagonal(overlap_matrix, 1.0)  # Perfect self-overlap
            
            fig = go.Figure(data=go.Heatmap(
                z=overlap_matrix,
                x=communities,
                y=communities,
                colorscale='Blues',
                colorbar=dict(title="Overlap Coefficient")
            ))
            
            fig.update_layout(
                title="Community Overlap Matrix",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create community overlap: {e}")
            return None
    
    def _create_community_health(self) -> Optional[go.Figure]:
        """Create community health metrics."""
        try:
            metrics = ['Cohesion', 'Stability', 'Growth', 'Innovation', 'Diversity']
            values = np.random.rand(len(metrics))
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='Community Health'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Community Health Metrics",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create community health: {e}")
            return None
    
    def _create_phase_space_plot(self) -> Optional[go.Figure]:
        """Create phase space plot."""
        try:
            # Sample phase space data
            t = np.linspace(0, 4*np.pi, 100)
            x = np.sin(t) * np.exp(-t/10)
            y = np.cos(t) * np.exp(-t/10)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines+markers',
                name='System Trajectory',
                line=dict(width=3, color='blue'),
                marker=dict(
                    size=4,
                    color=t,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time")
                )
            ))
            
            fig.update_layout(
                title="System Phase Space",
                xaxis_title="Order Parameter 1",
                yaxis_title="Order Parameter 2",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create phase space plot: {e}")
            return None
    
    def _create_cascade_risk_plot(self) -> Optional[go.Figure]:
        """Create cascade risk evolution plot."""
        try:
            timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
            risk_values = np.random.rand(len(timestamps)) * 2
            r0_values = np.random.rand(len(timestamps)) * 2 + 0.5
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=risk_values,
                mode='lines+markers',
                name='Cascade Risk',
                line=dict(color='red', width=3)
            ), secondary_y=False)
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=r0_values,
                mode='lines+markers',
                name='R₀',
                line=dict(color='blue', width=3)
            ), secondary_y=True)
            
            # Add critical threshold
            fig.add_hline(y=1.0, line_dash="dash", line_color="orange",
                         annotation_text="Critical Threshold")
            
            fig.update_yaxes(title_text="Cascade Risk", secondary_y=False)
            fig.update_yaxes(title_text="R₀", secondary_y=True)
            fig.update_xaxes(title_text="Time Period")
            
            fig.update_layout(
                title="Cascade Risk Evolution",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create cascade risk plot: {e}")
            return None
    
    def _create_complexity_metrics(self) -> Optional[go.Figure]:
        """Create system complexity metrics."""
        try:
            metrics = ['Entropy', 'Connectivity', 'Modularity', 'Criticality', 'Coherence']
            values = np.random.rand(len(metrics))
            
            fig = go.Figure(data=[
                go.Bar(
                    x=metrics,
                    y=values,
                    marker_color='lightblue',
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="System Complexity Metrics",
                yaxis_title="Metric Value",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create complexity metrics: {e}")
            return None
    
    def _create_global_flow_patterns(self) -> Optional[go.Figure]:
        """Create global flow patterns visualization."""
        try:
            # Sample global flow data
            timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
            
            inflow = np.random.rand(len(timestamps)) * 100
            outflow = np.random.rand(len(timestamps)) * 100
            net_flow = inflow - outflow
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=inflow,
                mode='lines+markers',
                name='Inflow',
                line=dict(color='green', width=3),
                fill='tonexty'
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=outflow,
                mode='lines+markers', 
                name='Outflow',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Global Flow Patterns",
                xaxis_title="Time Period",
                yaxis_title="Flow Magnitude",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create global flow patterns: {e}")
            return None
    
    def _create_long_term_trends(self) -> Optional[go.Figure]:
        """Create long-term trend analysis."""
        try:
            # Sample trend data
            timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
            
            stability = 0.7 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(timestamps)))
            innovation = 0.5 + 0.2 * np.cos(np.linspace(0, 2*np.pi, len(timestamps)))
            complexity = 0.6 + 0.15 * np.sin(np.linspace(0, 6*np.pi, len(timestamps)))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=stability,
                mode='lines+markers',
                name='Stability Trend',
                line=dict(width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=innovation,
                mode='lines+markers',
                name='Innovation Trend',
                line=dict(width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=complexity,
                mode='lines+markers',
                name='Complexity Trend',
                line=dict(width=3)
            ))
            
            fig.update_layout(
                title="Long-term System Trends",
                xaxis_title="Time Period",
                yaxis_title="Normalized Metric",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create long-term trends: {e}")
            return None
    
    def _create_regime_detection(self) -> Optional[go.Figure]:
        """Create regime detection plot."""
        try:
            timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
            regime_indicators = ['Stable', 'Transitional', 'Stable', 'Critical', 'Stable']
            regime_values = [0.2, 0.7, 0.3, 1.5, 0.4]
            
            colors = ['green' if v < 0.5 else 'orange' if v < 1.0 else 'red' for v in regime_values]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=timestamps,
                    y=regime_values,
                    marker_color=colors,
                    text=regime_indicators,
                    textposition='auto'
                )
            ])
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                         annotation_text="Transition Threshold")
            fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                         annotation_text="Critical Threshold")
            
            fig.update_layout(
                title="Regime Detection",
                xaxis_title="Time Period",
                yaxis_title="Regime Indicator",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create regime detection: {e}")
            return None
    
    def _create_cross_temporal_correlation(self) -> Optional[go.Figure]:
        """Create cross-temporal correlation matrix."""
        try:
            timestamps = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
            
            # Sample correlation matrix
            correlation_matrix = np.random.rand(len(timestamps), len(timestamps))
            np.fill_diagonal(correlation_matrix, 1.0)
            
            # Make symmetric
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=timestamps,
                y=timestamps,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Cross-temporal Correlation",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create cross-temporal correlation: {e}")
            return None
    
    def _create_predictive_indicators(self) -> Optional[go.Figure]:
        """Create predictive indicators plot."""
        try:
            indicators = ['Stability', 'Volatility', 'Cascade Risk', 'Innovation', 'Coherence']
            current_values = np.random.rand(len(indicators))
            predicted_values = current_values + np.random.randn(len(indicators)) * 0.1
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=indicators,
                y=current_values,
                mode='markers',
                name='Current',
                marker=dict(size=12, color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=indicators,
                y=predicted_values,
                mode='markers',
                name='Predicted',
                marker=dict(size=12, color='red', symbol='diamond')
            ))
            
            # Add error bars
            error_values = np.abs(predicted_values - current_values)
            fig.add_trace(go.Scatter(
                x=indicators,
                y=predicted_values,
                error_y=dict(
                    type='data',
                    array=error_values,
                    visible=True
                ),
                mode='markers',
                marker=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                name='Uncertainty'
            ))
            
            fig.update_layout(
                title="Predictive Indicators",
                yaxis_title="Indicator Value",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Failed to create predictive indicators: {e}")
            return None
    
    def _create_integrated_dashboard(self) -> Dict[str, Any]:
        """Create integrated multi-layer dashboard."""
        try:
            # Create subplot dashboard with key figures from each layer
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Word Trajectories', 'Community Evolution', 
                               'System Metrics', 'Long-term Trends'),
                specs=[[{"type": "scatter3d"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Add simplified versions of key plots
            
            # Word trajectories (simplified 2D)
            t = np.linspace(0, 2*np.pi, 20)
            for i, word in enumerate(['tech', 'innovation'][:2]):
                x = np.cos(t) + 0.2 * i
                y = np.sin(t) + 0.2 * i
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines+markers',
                    name=word,
                    showlegend=False
                ), row=1, col=1)
            
            # Community evolution
            timestamps = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
            for i, community in enumerate(['Tech', 'Business'][:2]):
                sizes = np.random.rand(len(timestamps)) * 100 + 20
                fig.add_trace(go.Scatter(
                    x=timestamps, y=sizes,
                    mode='lines+markers',
                    name=community,
                    showlegend=False
                ), row=1, col=2)
            
            # System metrics
            metrics = ['Entropy', 'Connectivity', 'Modularity']
            values = np.random.rand(len(metrics))
            fig.add_trace(go.Bar(
                x=metrics, y=values,
                showlegend=False
            ), row=2, col=1)
            
            # Long-term trends
            trend_values = 0.7 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(timestamps)))
            fig.add_trace(go.Scatter(
                x=timestamps, y=trend_values,
                mode='lines+markers',
                name='Stability',
                showlegend=False
            ), row=2, col=2)
            
            fig.update_layout(
                title_text="Semantic Flow Analysis Dashboard",
                height=800,
                showlegend=False
            )
            
            return {
                'dashboard_figure': fig,
                'dashboard_available': True,
                'layers_integrated': 4
            }
            
        except Exception as e:
            logger.warning(f"Failed to create integrated dashboard: {e}")
            return {'dashboard_available': False, 'error': str(e)}
    
    def get_layer(self, layer_name: str) -> Optional[VisualizationLayer]:
        """Get a specific visualization layer."""
        return self.layers.get(layer_name)
    
    def toggle_layer(self, layer_name: str) -> bool:
        """Toggle layer visibility."""
        if layer_name in self.layers:
            self.layers[layer_name].is_active = not self.layers[layer_name].is_active
            return self.layers[layer_name].is_active
        return False
    
    def export_layer(self, layer_name: str, format: str = 'html') -> Dict[str, str]:
        """Export layer figures."""
        if layer_name not in self.layers:
            return {'error': f'Layer {layer_name} not found'}
        
        layer = self.layers[layer_name]
        exported_files = {}
        
        for figure_name, figure in layer.figures.items():
            filename = f"{layer_name}_{figure_name}.{format}"
            
            try:
                if format == 'html':
                    figure.write_html(filename)
                elif format in ['png', 'jpg', 'pdf', 'svg']:
                    figure.write_image(filename)
                else:
                    continue
                
                exported_files[figure_name] = filename
                
            except Exception as e:
                logger.warning(f"Failed to export {figure_name}: {e}")
        
        return exported_files