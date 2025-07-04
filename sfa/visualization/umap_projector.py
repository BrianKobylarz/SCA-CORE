"""UMAP projection and dimensionality reduction for semantic flow visualization."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from ..core.types import Word, Timestamp, Embedding
from ..core.base import BaseAnalyzer

class UMAPProjector(BaseAnalyzer):
    """Projects high-dimensional embeddings to 2D/3D using UMAP with temporal alignment"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("UMAPProjector")
        self.embeddings = embeddings_store
        self.config = config
        
        # UMAP parameters
        self.umap_config = {
            'n_components': 2,           # Output dimensions (2 or 3)
            'n_neighbors': 15,           # Number of neighbors for UMAP
            'min_dist': 0.1,             # Minimum distance in embedding
            'metric': 'cosine',          # Distance metric
            'random_state': 42,          # Random seed for reproducibility
            'temporal_alignment': True,   # Align embeddings across time
            'animation_frames': True,     # Create animation frames
            'color_by': 'cluster',       # Color points by: cluster, time, centrality
            'point_size': 5,             # Point size in visualization
            'trail_length': 5,           # Length of temporal trails
            'opacity': 0.7,              # Point opacity
            'show_labels': False,        # Show word labels
            'label_threshold': 0.1,      # Threshold for showing labels
            'background_color': 'white', # Background color
            'color_palette': 'viridis'   # Color palette
        }
        
        # UMAP results
        self.umap_embeddings = {}     # timestamp -> 2D/3D coordinates
        self.umap_model = None        # Fitted UMAP model
        self.temporal_trajectories = {}  # word -> trajectory coordinates
        self.cluster_assignments = {}    # timestamp -> cluster labels
        
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive UMAP analysis"""
        # Prepare embedding data
        embedding_data = self._prepare_embedding_data()
        
        # Fit UMAP model
        umap_results = self._fit_umap_model(embedding_data)
        
        # Generate temporal trajectories
        trajectories = self._compute_temporal_trajectories()
        
        # Create visualizations
        static_plot = self.create_static_umap_plot()
        animated_plot = self.create_animated_umap_plot()
        trajectory_plot = self.create_trajectory_plot()
        
        # Analyze embedding space properties
        space_analysis = self._analyze_embedding_space()
        
        return {
            'umap_results': umap_results,
            'temporal_trajectories': trajectories,
            'static_plot': static_plot,
            'animated_plot': animated_plot,
            'trajectory_plot': trajectory_plot,
            'space_analysis': space_analysis,
            'config': self.umap_config
        }
    
    def _prepare_embedding_data(self) -> Dict[str, Any]:
        """Prepare embedding data for UMAP projection"""
        timestamps = self.embeddings.get_timestamps()
        
        # Collect all embeddings
        all_embeddings = []
        all_words = []
        all_timestamps = []
        word_to_indices = {}  # word -> list of indices
        
        for timestamp in timestamps:
            vocabulary = self.embeddings.get_vocabulary(timestamp)
            
            for word in vocabulary:
                embedding = self.embeddings.get_embedding(word, timestamp)
                if embedding is not None:
                    all_embeddings.append(embedding)
                    all_words.append(word)
                    all_timestamps.append(timestamp)
                    
                    if word not in word_to_indices:
                        word_to_indices[word] = []
                    word_to_indices[word].append(len(all_embeddings) - 1)
        
        if not all_embeddings:
            return {}
        
        embedding_matrix = np.array(all_embeddings)
        
        return {
            'embeddings': embedding_matrix,
            'words': all_words,
            'timestamps': all_timestamps,
            'word_indices': word_to_indices,
            'unique_words': list(word_to_indices.keys()),
            'unique_timestamps': timestamps
        }
    
    def _fit_umap_model(self, embedding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fit UMAP model and compute projections"""
        if not embedding_data:
            return {}
        
        try:
            import umap
        except ImportError:
            # Fallback to PCA if UMAP not available
            return self._fit_pca_fallback(embedding_data)
        
        embeddings = embedding_data['embeddings']
        words = embedding_data['words']
        timestamps = embedding_data['timestamps']
        
        # Configure UMAP
        umap_model = umap.UMAP(
            n_components=self.umap_config['n_components'],
            n_neighbors=self.umap_config['n_neighbors'],
            min_dist=self.umap_config['min_dist'],
            metric=self.umap_config['metric'],
            random_state=self.umap_config['random_state']
        )
        
        # Fit and transform
        umap_coords = umap_model.fit_transform(embeddings)
        
        # Organize results by timestamp
        for i, (word, timestamp) in enumerate(zip(words, timestamps)):
            if timestamp not in self.umap_embeddings:
                self.umap_embeddings[timestamp] = {}
            
            coords = umap_coords[i]
            self.umap_embeddings[timestamp][word] = coords
        
        self.umap_model = umap_model
        
        # Compute clusters
        self._compute_clusters(umap_coords, words, timestamps)
        
        return {
            'umap_coordinates': umap_coords,
            'model_params': self.umap_config,
            'explained_variance': self._estimate_explained_variance(embeddings, umap_coords),
            'neighborhood_preservation': self._compute_neighborhood_preservation(embeddings, umap_coords)
        }
    
    def _fit_pca_fallback(self, embedding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to PCA if UMAP is not available"""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            return {'error': 'Neither UMAP nor scikit-learn available'}
        
        embeddings = embedding_data['embeddings']
        words = embedding_data['words']
        timestamps = embedding_data['timestamps']
        
        # Use PCA for dimensionality reduction
        n_components = self.umap_config['n_components']
        pca = PCA(n_components=n_components, random_state=self.umap_config['random_state'])
        pca_coords = pca.fit_transform(embeddings)
        
        # Organize results by timestamp
        for i, (word, timestamp) in enumerate(zip(words, timestamps)):
            if timestamp not in self.umap_embeddings:
                self.umap_embeddings[timestamp] = {}
            
            coords = pca_coords[i]
            self.umap_embeddings[timestamp][word] = coords
        
        self.umap_model = pca
        
        # Compute clusters
        self._compute_clusters(pca_coords, words, timestamps)
        
        return {
            'pca_coordinates': pca_coords,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'total_explained_variance': np.sum(pca.explained_variance_ratio_),
            'method': 'PCA_fallback'
        }
    
    def _compute_clusters(self, coords: np.ndarray, words: List[Word], timestamps: List[Timestamp]) -> None:
        """Compute cluster assignments for points"""
        try:
            from sklearn.cluster import KMeans
            
            # Determine optimal number of clusters
            n_clusters = min(10, max(2, len(np.unique(words)) // 20))
            
            if len(coords) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(coords)
                
                # Organize by timestamp
                for i, timestamp in enumerate(timestamps):
                    if timestamp not in self.cluster_assignments:
                        self.cluster_assignments[timestamp] = {}
                    
                    word = words[i]
                    self.cluster_assignments[timestamp][word] = cluster_labels[i]
        
        except ImportError:
            # Simple distance-based clustering fallback
            self._simple_clustering(coords, words, timestamps)
    
    def _simple_clustering(self, coords: np.ndarray, words: List[Word], timestamps: List[Timestamp]) -> None:
        """Simple clustering fallback"""
        # Use coordinate-based clustering
        for i, timestamp in enumerate(timestamps):
            if timestamp not in self.cluster_assignments:
                self.cluster_assignments[timestamp] = {}
            
            word = words[i]
            coord = coords[i]
            
            # Simple quadrant-based clustering for 2D
            if len(coord) >= 2:
                x, y = coord[0], coord[1]
                if x >= 0 and y >= 0:
                    cluster = 0
                elif x < 0 and y >= 0:
                    cluster = 1
                elif x < 0 and y < 0:
                    cluster = 2
                else:
                    cluster = 3
            else:
                cluster = 0
            
            self.cluster_assignments[timestamp][word] = cluster
    
    def _compute_temporal_trajectories(self) -> Dict[Word, List[Tuple[float, ...]]]:
        """Compute temporal trajectories for words"""
        trajectories = {}
        timestamps = sorted(self.umap_embeddings.keys())
        
        # Get words that appear in multiple timestamps
        word_counts = {}
        for timestamp_coords in self.umap_embeddings.values():
            for word in timestamp_coords.keys():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Only track words that appear in multiple timestamps
        tracked_words = [word for word, count in word_counts.items() if count > 1]
        
        for word in tracked_words:
            trajectory = []
            
            for timestamp in timestamps:
                if word in self.umap_embeddings[timestamp]:
                    coords = self.umap_embeddings[timestamp][word]
                    trajectory.append(tuple(coords))
            
            if len(trajectory) > 1:
                trajectories[word] = trajectory
        
        self.temporal_trajectories = trajectories
        return trajectories
    
    def create_static_umap_plot(self) -> go.Figure:
        """Create static UMAP plot for latest timestamp"""
        timestamps = sorted(self.umap_embeddings.keys())
        if not timestamps:
            return go.Figure()
        
        latest_timestamp = timestamps[-1]
        return self._create_umap_plot_for_timestamp(latest_timestamp)
    
    def _create_umap_plot_for_timestamp(self, timestamp: Timestamp) -> go.Figure:
        """Create UMAP plot for specific timestamp"""
        coords_dict = self.umap_embeddings.get(timestamp, {})
        cluster_dict = self.cluster_assignments.get(timestamp, {})
        
        if not coords_dict:
            return go.Figure()
        
        # Prepare data
        words = list(coords_dict.keys())
        coords = list(coords_dict.values())
        coords_array = np.array(coords)
        
        # Get clusters
        clusters = [cluster_dict.get(word, 0) for word in words]
        
        if self.umap_config['n_components'] == 2:
            # 2D plot
            fig = self._create_2d_plot(words, coords_array, clusters, timestamp)
        else:
            # 3D plot
            fig = self._create_3d_plot(words, coords_array, clusters, timestamp)
        
        return fig
    
    def _create_2d_plot(self, words: List[Word], coords: np.ndarray, 
                       clusters: List[int], timestamp: Timestamp) -> go.Figure:
        """Create 2D UMAP plot"""
        # Create DataFrame for easier handling
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'word': words,
            'cluster': clusters
        })
        
        # Create scatter plot
        fig = px.scatter(
            df, x='x', y='y', 
            color='cluster',
            hover_data=['word'],
            title=f'UMAP Projection - {timestamp}',
            color_continuous_scale=self.umap_config['color_palette']
        )
        
        # Customize traces
        fig.update_traces(
            marker=dict(
                size=self.umap_config['point_size'],
                opacity=self.umap_config['opacity'],
                line=dict(width=1, color='white')
            )
        )
        
        # Add labels if requested
        if self.umap_config['show_labels']:
            self._add_labels_2d(fig, df)
        
        # Update layout
        fig.update_layout(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            plot_bgcolor=self.umap_config['background_color'],
            height=600
        )
        
        return fig
    
    def _create_3d_plot(self, words: List[Word], coords: np.ndarray, 
                       clusters: List[int], timestamp: Timestamp) -> go.Figure:
        """Create 3D UMAP plot"""
        # Ensure we have 3 dimensions
        if coords.shape[1] < 3:
            # Pad with zeros if needed
            padding = np.zeros((coords.shape[0], 3 - coords.shape[1]))
            coords = np.hstack([coords, padding])
        
        fig = go.Figure()
        
        # Create trace
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1], 
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=self.umap_config['point_size'],
                color=clusters,
                colorscale=self.umap_config['color_palette'],
                opacity=self.umap_config['opacity'],
                colorbar=dict(title="Cluster"),
                line=dict(width=1, color='white')
            ),
            text=words,
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
            name='Words'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'3D UMAP Projection - {timestamp}',
            scene=dict(
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                zaxis_title="UMAP Dimension 3",
                bgcolor=self.umap_config['background_color']
            ),
            height=700
        )
        
        return fig
    
    def create_animated_umap_plot(self) -> go.Figure:
        """Create animated UMAP plot showing temporal evolution"""
        timestamps = sorted(self.umap_embeddings.keys())
        
        if len(timestamps) < 2:
            return self.create_static_umap_plot()
        
        # Create frames
        frames = []
        
        for timestamp in timestamps:
            coords_dict = self.umap_embeddings[timestamp]
            cluster_dict = self.cluster_assignments.get(timestamp, {})
            
            if coords_dict:
                words = list(coords_dict.keys())
                coords = np.array(list(coords_dict.values()))
                clusters = [cluster_dict.get(word, 0) for word in words]
                
                if self.umap_config['n_components'] == 2:
                    frame_data = [go.Scatter(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        mode='markers',
                        marker=dict(
                            size=self.umap_config['point_size'],
                            color=clusters,
                            colorscale=self.umap_config['color_palette'],
                            opacity=self.umap_config['opacity']
                        ),
                        text=words,
                        name='Words'
                    )]
                else:
                    # Pad to 3D if needed
                    if coords.shape[1] < 3:
                        padding = np.zeros((coords.shape[0], 3 - coords.shape[1]))
                        coords = np.hstack([coords, padding])
                    
                    frame_data = [go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode='markers',
                        marker=dict(
                            size=self.umap_config['point_size'],
                            color=clusters,
                            colorscale=self.umap_config['color_palette'],
                            opacity=self.umap_config['opacity']
                        ),
                        text=words,
                        name='Words'
                    )]
                
                frames.append(go.Frame(
                    data=frame_data,
                    name=str(timestamp),
                    layout=go.Layout(title=f"UMAP Evolution - {timestamp}")
                ))
        
        # Create initial frame
        if frames:
            initial_data = frames[0].data
        else:
            initial_data = []
        
        # Create figure
        fig = go.Figure(data=initial_data, frames=frames)
        
        # Add animation controls
        fig.update_layout(
            title="Animated UMAP Projection",
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 1000}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[timestamp], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                        'label': str(timestamp),
                        'method': 'animate'
                    }
                    for timestamp in timestamps
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Timestamp: '}
            }],
            height=600
        )
        
        return fig
    
    def create_trajectory_plot(self) -> go.Figure:
        """Create plot showing temporal trajectories of words"""
        if not self.temporal_trajectories:
            return go.Figure()
        
        fig = go.Figure()
        
        # Sample trajectories to avoid overcrowding
        max_trajectories = 20
        sampled_words = list(self.temporal_trajectories.keys())[:max_trajectories]
        
        for word in sampled_words:
            trajectory = self.temporal_trajectories[word]
            
            if len(trajectory) >= 2:
                coords = np.array(trajectory)
                
                if self.umap_config['n_components'] == 2:
                    # 2D trajectory
                    fig.add_trace(go.Scatter(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        mode='lines+markers',
                        name=word,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                else:
                    # 3D trajectory
                    if coords.shape[1] < 3:
                        padding = np.zeros((coords.shape[0], 3 - coords.shape[1]))
                        coords = np.hstack([coords, padding])
                    
                    fig.add_trace(go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode='lines+markers',
                        name=word,
                        line=dict(width=4),
                        marker=dict(size=4)
                    ))
        
        # Update layout
        if self.umap_config['n_components'] == 2:
            fig.update_layout(
                title="Word Trajectories in UMAP Space",
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                height=600
            )
        else:
            fig.update_layout(
                title="Word Trajectories in 3D UMAP Space",
                scene=dict(
                    xaxis_title="UMAP Dimension 1",
                    yaxis_title="UMAP Dimension 2",
                    zaxis_title="UMAP Dimension 3"
                ),
                height=700
            )
        
        return fig
    
    def _add_labels_2d(self, fig: go.Figure, df: pd.DataFrame) -> None:
        """Add labels to 2D plot"""
        # Only show labels for points that are well-separated
        threshold = self.umap_config['label_threshold']
        
        # Simple spacing-based label filtering
        shown_labels = []
        for i, row in df.iterrows():
            show_label = True
            
            for prev_x, prev_y in shown_labels:
                distance = np.sqrt((row['x'] - prev_x)**2 + (row['y'] - prev_y)**2)
                if distance < threshold:
                    show_label = False
                    break
            
            if show_label:
                fig.add_annotation(
                    x=row['x'], y=row['y'],
                    text=row['word'],
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
                shown_labels.append((row['x'], row['y']))
    
    def _analyze_embedding_space(self) -> Dict[str, Any]:
        """Analyze properties of the UMAP embedding space"""
        analysis = {
            'dimensionality_reduction_quality': {},
            'cluster_analysis': {},
            'temporal_analysis': {},
            'space_coverage': {}
        }
        
        if not self.umap_embeddings:
            return analysis
        
        # Analyze cluster quality
        all_coords = []
        all_clusters = []
        
        for timestamp in self.umap_embeddings.keys():
            coords_dict = self.umap_embeddings[timestamp]
            cluster_dict = self.cluster_assignments.get(timestamp, {})
            
            for word, coords in coords_dict.items():
                all_coords.append(coords)
                all_clusters.append(cluster_dict.get(word, 0))
        
        if all_coords:
            all_coords = np.array(all_coords)
            
            # Cluster analysis
            try:
                from sklearn.metrics import silhouette_score
                if len(set(all_clusters)) > 1:
                    silhouette = silhouette_score(all_coords, all_clusters)
                    analysis['cluster_analysis']['silhouette_score'] = silhouette
            except ImportError:
                pass
            
            # Space coverage
            analysis['space_coverage'] = {
                'x_range': float(all_coords[:, 0].max() - all_coords[:, 0].min()),
                'y_range': float(all_coords[:, 1].max() - all_coords[:, 1].min()),
                'density': len(all_coords) / (
                    (all_coords[:, 0].max() - all_coords[:, 0].min()) *
                    (all_coords[:, 1].max() - all_coords[:, 1].min())
                ) if all_coords.shape[1] >= 2 else 0
            }
            
            if all_coords.shape[1] >= 3:
                analysis['space_coverage']['z_range'] = float(all_coords[:, 2].max() - all_coords[:, 2].min())
        
        # Temporal analysis
        if len(self.temporal_trajectories) > 0:
            trajectory_lengths = [len(traj) for traj in self.temporal_trajectories.values()]
            trajectory_distances = []
            
            for trajectory in self.temporal_trajectories.values():
                if len(trajectory) >= 2:
                    coords = np.array(trajectory)
                    total_distance = 0
                    for i in range(1, len(coords)):
                        distance = np.linalg.norm(coords[i] - coords[i-1])
                        total_distance += distance
                    trajectory_distances.append(total_distance)
            
            analysis['temporal_analysis'] = {
                'num_tracked_words': len(self.temporal_trajectories),
                'average_trajectory_length': np.mean(trajectory_lengths),
                'average_trajectory_distance': np.mean(trajectory_distances) if trajectory_distances else 0,
                'max_trajectory_distance': np.max(trajectory_distances) if trajectory_distances else 0
            }
        
        return analysis
    
    def _estimate_explained_variance(self, original_embeddings: np.ndarray, 
                                   projected_embeddings: np.ndarray) -> float:
        """Estimate explained variance for UMAP projection"""
        # This is an approximation since UMAP is non-linear
        try:
            from sklearn.metrics import r2_score
            
            # Use distance preservation as proxy for explained variance
            n_samples = min(1000, len(original_embeddings))
            indices = np.random.choice(len(original_embeddings), n_samples, replace=False)
            
            orig_subset = original_embeddings[indices]
            proj_subset = projected_embeddings[indices]
            
            # Compute pairwise distances
            from scipy.spatial.distance import pdist
            orig_distances = pdist(orig_subset)
            proj_distances = pdist(proj_subset)
            
            # Compute correlation between distance matrices
            correlation = np.corrcoef(orig_distances, proj_distances)[0, 1]
            
            return max(0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _compute_neighborhood_preservation(self, original_embeddings: np.ndarray,
                                         projected_embeddings: np.ndarray) -> float:
        """Compute neighborhood preservation score"""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            k = min(10, len(original_embeddings) - 1)
            if k <= 0:
                return 0.0
            
            # Find k-nearest neighbors in original space
            nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(original_embeddings)
            _, indices_orig = nbrs_orig.kneighbors(original_embeddings)
            
            # Find k-nearest neighbors in projected space  
            nbrs_proj = NearestNeighbors(n_neighbors=k+1).fit(projected_embeddings)
            _, indices_proj = nbrs_proj.kneighbors(projected_embeddings)
            
            # Compute neighborhood preservation
            preservation_scores = []
            for i in range(len(original_embeddings)):
                neighbors_orig = set(indices_orig[i][1:])  # Exclude self
                neighbors_proj = set(indices_proj[i][1:])  # Exclude self
                
                intersection = len(neighbors_orig & neighbors_proj)
                preservation = intersection / k
                preservation_scores.append(preservation)
            
            return np.mean(preservation_scores)
            
        except Exception:
            return 0.0
    
    def export_coordinates(self, timestamp: Optional[Timestamp] = None) -> Dict[str, Any]:
        """Export UMAP coordinates for external use"""
        if timestamp is None:
            return {
                'all_timestamps': self.umap_embeddings,
                'trajectories': self.temporal_trajectories,
                'clusters': self.cluster_assignments
            }
        else:
            return {
                'coordinates': self.umap_embeddings.get(timestamp, {}),
                'clusters': self.cluster_assignments.get(timestamp, {}),
                'timestamp': timestamp
            }