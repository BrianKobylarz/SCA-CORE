"""3D animated flow network visualization for semantic flows."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

from ..core.types import Word, Timestamp, SemanticFlow
from ..core.base import BaseAnalyzer

class FlowNetworkVisualizer(BaseAnalyzer):
    """Creates 3D animated visualizations of semantic flow networks"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("FlowNetworkVisualizer")
        self.embeddings = embeddings_store
        self.config = config
        
        # Visualization parameters
        self.viz_config = {
            'max_nodes': 500,              # Maximum nodes to display
            'min_flow_strength': 0.1,      # Minimum flow strength to show
            'node_size_range': (5, 50),    # Node size range
            'edge_width_range': (1, 10),   # Edge width range
            'animation_duration': 1000,    # Animation duration in ms
            'color_scheme': 'viridis',     # Color scheme for nodes
            'layout_algorithm': 'spring',  # Network layout algorithm
            'particle_trails': True,       # Show particle trails for flows
            'flow_animation_speed': 0.5,   # Flow animation speed
            'opacity': 0.8,                # Node/edge opacity
            'show_labels': True,           # Show word labels
            'label_font_size': 12,         # Label font size
            'background_color': 'black',   # Background color
            'show_legend': True            # Show color legend
        }
        
        # Network data
        self.networks = {}  # timestamp -> network
        self.layouts = {}   # timestamp -> node positions
        self.flow_animations = []
        
    def analyze(self) -> Dict[str, Any]:
        """Generate comprehensive network visualization analysis"""
        # Build networks for all timestamps
        self._build_networks()
        
        # Compute layouts
        self._compute_layouts()
        
        # Create 3D animated visualization
        animated_plot = self.create_animated_flow_network()
        
        # Create static network snapshots
        network_snapshots = self.create_network_snapshots()
        
        # Create flow particle animation
        particle_animation = self.create_flow_particle_animation()
        
        # Generate network statistics
        network_stats = self._compute_network_statistics()
        
        return {
            'animated_plot': animated_plot,
            'network_snapshots': network_snapshots,
            'particle_animation': particle_animation,
            'network_statistics': network_stats,
            'visualization_config': self.viz_config
        }
    
    def _build_networks(self) -> None:
        """Build network representations for each timestamp"""
        timestamps = self.embeddings.get_timestamps()
        
        for timestamp in timestamps:
            G = nx.Graph()
            vocabulary = list(self.embeddings.get_vocabulary(timestamp))
            
            # Limit vocabulary size for visualization
            if len(vocabulary) > self.viz_config['max_nodes']:
                # Sample most important nodes (could be based on degree, betweenness, etc.)
                vocabulary = vocabulary[:self.viz_config['max_nodes']]
            
            # Add nodes with attributes
            for word in vocabulary:
                embedding = self.embeddings.get_embedding(word, timestamp)
                if embedding is not None:
                    # Use embedding norm as node size indicator
                    node_size = np.linalg.norm(embedding)
                    G.add_node(word, 
                              size=node_size,
                              embedding=embedding,
                              timestamp=timestamp)
            
            # Add edges based on semantic similarity
            self._add_semantic_edges(G, vocabulary, timestamp)
            
            self.networks[timestamp] = G
    
    def _add_semantic_edges(self, G: nx.Graph, vocabulary: List[Word], timestamp: Timestamp) -> None:
        """Add edges based on semantic similarity"""
        from scipy.spatial.distance import cosine
        
        similarity_threshold = 0.3
        
        for i, word1 in enumerate(vocabulary):
            emb1 = self.embeddings.get_embedding(word1, timestamp)
            if emb1 is None:
                continue
                
            for word2 in vocabulary[i+1:]:
                emb2 = self.embeddings.get_embedding(word2, timestamp)
                if emb2 is not None:
                    similarity = 1 - cosine(emb1, emb2)
                    
                    if similarity > similarity_threshold:
                        # Edge weight represents flow strength
                        flow_strength = similarity
                        
                        if flow_strength >= self.viz_config['min_flow_strength']:
                            G.add_edge(word1, word2, 
                                     weight=flow_strength,
                                     similarity=similarity)
    
    def _compute_layouts(self) -> None:
        """Compute 3D layouts for networks"""
        layout_algorithm = self.viz_config['layout_algorithm']
        
        for timestamp, network in self.networks.items():
            if network.number_of_nodes() == 0:
                self.layouts[timestamp] = {}
                continue
            
            if layout_algorithm == 'spring':
                # Use spring layout in 2D then add z-dimension
                pos_2d = nx.spring_layout(network, weight='weight', k=1.0, iterations=50)
                
                # Add z-dimension based on node centrality
                centrality = nx.degree_centrality(network)
                pos_3d = {}
                for node, (x, y) in pos_2d.items():
                    z = centrality.get(node, 0) * 2 - 1  # Scale to [-1, 1]
                    pos_3d[node] = (x, y, z)
                
                self.layouts[timestamp] = pos_3d
                
            elif layout_algorithm == 'embedding_based':
                # Use embedding dimensions for positioning
                pos_3d = self._embedding_based_layout(network, timestamp)
                self.layouts[timestamp] = pos_3d
            
            else:
                # Default to random layout
                pos_3d = {}
                for node in network.nodes():
                    pos_3d[node] = (
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1), 
                        np.random.uniform(-1, 1)
                    )
                self.layouts[timestamp] = pos_3d
    
    def _embedding_based_layout(self, network: nx.Graph, timestamp: Timestamp) -> Dict[Word, Tuple[float, float, float]]:
        """Create layout based on embedding space"""
        pos_3d = {}
        
        # Use PCA to reduce embeddings to 3D
        embeddings = []
        words = []
        
        for word in network.nodes():
            emb = self.embeddings.get_embedding(word, timestamp)
            if emb is not None:
                embeddings.append(emb)
                words.append(word)
        
        if len(embeddings) < 3:
            # Fallback to random
            for word in words:
                pos_3d[word] = (
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1)
                )
            return pos_3d
        
        # PCA to 3D
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(embeddings)
            
            # Normalize to [-1, 1] range
            for i, coord in enumerate(['x', 'y', 'z']):
                col = embeddings_3d[:, i]
                col_min, col_max = col.min(), col.max()
                if col_max > col_min:
                    embeddings_3d[:, i] = 2 * (col - col_min) / (col_max - col_min) - 1
            
            for i, word in enumerate(words):
                pos_3d[word] = tuple(embeddings_3d[i])
                
        except ImportError:
            # Fallback if sklearn not available
            for word in words:
                emb = self.embeddings.get_embedding(word, timestamp)
                # Use first 3 dimensions, normalized
                if len(emb) >= 3:
                    x, y, z = emb[:3]
                    # Simple normalization
                    norm = np.linalg.norm([x, y, z])
                    if norm > 0:
                        x, y, z = x/norm, y/norm, z/norm
                    pos_3d[word] = (x, y, z)
                else:
                    pos_3d[word] = (
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1)
                    )
        
        return pos_3d
    
    def create_animated_flow_network(self) -> go.Figure:
        """Create animated 3D flow network visualization"""
        timestamps = sorted(self.networks.keys())
        
        if not timestamps:
            return go.Figure()
        
        # Create frames for animation
        frames = []
        
        for timestamp in timestamps:
            frame_data = self._create_network_frame(timestamp)
            frames.append(go.Frame(
                data=frame_data,
                name=str(timestamp),
                layout=go.Layout(title=f"Semantic Flow Network - {timestamp}")
            ))
        
        # Create initial frame
        initial_data = self._create_network_frame(timestamps[0])
        
        # Create figure
        fig = go.Figure(data=initial_data, frames=frames)
        
        # Update layout
        fig.update_layout(
            title="Animated Semantic Flow Network",
            scene=dict(
                xaxis_title="X Dimension",
                yaxis_title="Y Dimension", 
                zaxis_title="Z Dimension",
                bgcolor=self.viz_config['background_color'],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': self.viz_config['animation_duration']},
                            'fromcurrent': True
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[timestamp], {
                            'frame': {'duration': 0},
                            'mode': 'immediate'
                        }],
                        'label': str(timestamp),
                        'method': 'animate'
                    }
                    for timestamp in timestamps
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Timestamp: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }],
            showlegend=self.viz_config['show_legend'],
            height=800
        )
        
        return fig
    
    def _create_network_frame(self, timestamp: Timestamp) -> List[go.Scatter3d]:
        """Create network frame for given timestamp"""
        network = self.networks.get(timestamp)
        layout = self.layouts.get(timestamp, {})
        
        if not network or not layout:
            return []
        
        traces = []
        
        # Create edges trace
        edge_trace = self._create_edges_trace(network, layout)
        if edge_trace:
            traces.append(edge_trace)
        
        # Create nodes trace
        nodes_trace = self._create_nodes_trace(network, layout, timestamp)
        if nodes_trace:
            traces.append(nodes_trace)
        
        return traces
    
    def _create_edges_trace(self, network: nx.Graph, layout: Dict[Word, Tuple[float, float, float]]) -> Optional[go.Scatter3d]:
        """Create edges trace for network"""
        edge_x, edge_y, edge_z = [], [], []
        edge_info = []
        
        for edge in network.edges(data=True):
            word1, word2, data = edge
            
            if word1 in layout and word2 in layout:
                x1, y1, z1 = layout[word1]
                x2, y2, z2 = layout[word2]
                
                # Add edge coordinates
                edge_x.extend([x1, x2, None])
                edge_y.extend([y1, y2, None])
                edge_z.extend([z1, z2, None])
                
                # Store edge info
                weight = data.get('weight', 0)
                edge_info.append(f"{word1} ↔ {word2}<br>Strength: {weight:.3f}")
        
        if not edge_x:
            return None
        
        return go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                color='rgba(125, 125, 125, 0.5)',
                width=2
            ),
            hoverinfo='none',
            name='Connections'
        )
    
    def _create_nodes_trace(self, network: nx.Graph, layout: Dict[Word, Tuple[float, float, float]], 
                           timestamp: Timestamp) -> Optional[go.Scatter3d]:
        """Create nodes trace for network"""
        node_x, node_y, node_z = [], [], []
        node_text, node_info = [], []
        node_sizes, node_colors = [], []
        
        for node in network.nodes(data=True):
            word, data = node
            
            if word in layout:
                x, y, z = layout[word]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                
                # Node text (labels)
                if self.viz_config['show_labels']:
                    node_text.append(word)
                else:
                    node_text.append('')
                
                # Node info for hover
                degree = network.degree(word)
                size = data.get('size', 1.0)
                node_info.append(f"Word: {word}<br>Degree: {degree}<br>Size: {size:.3f}")
                
                # Node size based on degree
                min_size, max_size = self.viz_config['node_size_range']
                max_degree = max([d for n, d in network.degree()] + [1])
                normalized_degree = degree / max_degree
                node_size = min_size + normalized_degree * (max_size - min_size)
                node_sizes.append(node_size)
                
                # Node color based on size or centrality
                node_colors.append(size)
        
        if not node_x:
            return None
        
        return go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale=self.viz_config['color_scheme'],
                opacity=self.viz_config['opacity'],
                colorbar=dict(title="Node Importance") if self.viz_config['show_legend'] else None,
                line=dict(width=1, color='white')
            ),
            text=node_text,
            textposition='middle center',
            textfont=dict(size=self.viz_config['label_font_size'], color='white'),
            hovertext=node_info,
            hoverinfo='text',
            name='Words'
        )
    
    def create_network_snapshots(self) -> Dict[str, go.Figure]:
        """Create static network snapshots for each timestamp"""
        snapshots = {}
        
        for timestamp in self.networks.keys():
            fig = self._create_static_network(timestamp)
            snapshots[timestamp] = fig
        
        return snapshots
    
    def _create_static_network(self, timestamp: Timestamp) -> go.Figure:
        """Create static network visualization for timestamp"""
        frame_data = self._create_network_frame(timestamp)
        
        fig = go.Figure(data=frame_data)
        
        fig.update_layout(
            title=f"Semantic Network - {timestamp}",
            scene=dict(
                xaxis_title="X Dimension",
                yaxis_title="Y Dimension",
                zaxis_title="Z Dimension",
                bgcolor=self.viz_config['background_color']
            ),
            showlegend=self.viz_config['show_legend'],
            height=600
        )
        
        return fig
    
    def create_flow_particle_animation(self) -> go.Figure:
        """Create particle animation showing flow dynamics"""
        if not self.viz_config['particle_trails']:
            return go.Figure()
        
        # This would create animated particles flowing along edges
        # For now, return a placeholder
        timestamps = sorted(self.networks.keys())
        
        if not timestamps:
            return go.Figure()
        
        # Create simple particle trail visualization
        fig = go.Figure()
        
        # Add sample particle traces
        for i in range(5):  # 5 particle trails
            x_coords = np.random.uniform(-1, 1, 50)
            y_coords = np.random.uniform(-1, 1, 50)
            z_coords = np.random.uniform(-1, 1, 50)
            
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers+lines',
                marker=dict(
                    size=3,
                    color=f'rgba({100 + i*30}, {150}, {200}, 0.6)'
                ),
                line=dict(width=2),
                name=f'Flow Particle {i+1}'
            ))
        
        fig.update_layout(
            title="Semantic Flow Particles",
            scene=dict(
                xaxis_title="X Dimension",
                yaxis_title="Y Dimension",
                zaxis_title="Z Dimension"
            ),
            height=600
        )
        
        return fig
    
    def _compute_network_statistics(self) -> Dict[str, Any]:
        """Compute network statistics across timestamps"""
        stats = {
            'num_timestamps': len(self.networks),
            'temporal_stats': {},
            'evolution_metrics': {}
        }
        
        # Per-timestamp statistics
        for timestamp, network in self.networks.items():
            timestamp_stats = {
                'num_nodes': network.number_of_nodes(),
                'num_edges': network.number_of_edges(),
                'density': nx.density(network) if network.number_of_nodes() > 1 else 0,
                'average_clustering': nx.average_clustering(network) if network.number_of_nodes() > 2 else 0
            }
            
            # Add centrality measures
            if network.number_of_nodes() > 0:
                degree_centrality = nx.degree_centrality(network)
                timestamp_stats['max_degree_centrality'] = max(degree_centrality.values())
                timestamp_stats['average_degree_centrality'] = np.mean(list(degree_centrality.values()))
            
            stats['temporal_stats'][timestamp] = timestamp_stats
        
        # Evolution metrics
        if len(self.networks) > 1:
            timestamps = sorted(self.networks.keys())
            
            # Network size evolution
            node_counts = [self.networks[t].number_of_nodes() for t in timestamps]
            edge_counts = [self.networks[t].number_of_edges() for t in timestamps]
            
            stats['evolution_metrics'] = {
                'node_count_trend': np.polyfit(range(len(node_counts)), node_counts, 1)[0] if len(node_counts) > 1 else 0,
                'edge_count_trend': np.polyfit(range(len(edge_counts)), edge_counts, 1)[0] if len(edge_counts) > 1 else 0,
                'average_nodes': np.mean(node_counts),
                'average_edges': np.mean(edge_counts)
            }
        
        return stats
    
    def export_network_data(self, timestamp: Timestamp, format_type: str = 'networkx') -> Any:
        """Export network data in various formats"""
        network = self.networks.get(timestamp)
        layout = self.layouts.get(timestamp, {})
        
        if not network:
            return None
        
        if format_type == 'networkx':
            return network
        
        elif format_type == 'gephi':
            # Export for Gephi visualization
            node_data = []
            for node in network.nodes(data=True):
                word, data = node
                if word in layout:
                    x, y, z = layout[word]
                    node_data.append({
                        'Id': word,
                        'Label': word,
                        'X': x,
                        'Y': y,
                        'Z': z,
                        'Size': data.get('size', 1.0)
                    })
            
            edge_data = []
            for edge in network.edges(data=True):
                word1, word2, data = edge
                edge_data.append({
                    'Source': word1,
                    'Target': word2,
                    'Weight': data.get('weight', 1.0)
                })
            
            return {'nodes': node_data, 'edges': edge_data}
        
        elif format_type == 'json':
            # Export as JSON
            return {
                'nodes': [
                    {
                        'id': word,
                        'position': layout.get(word, (0, 0, 0)),
                        'attributes': data
                    }
                    for word, data in network.nodes(data=True)
                ],
                'edges': [
                    {
                        'source': word1,
                        'target': word2,
                        'attributes': data
                    }
                    for word1, word2, data in network.edges(data=True)
                ]
            }
        
        return network