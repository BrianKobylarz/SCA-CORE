"""Gravitational field engine for semantic flow analysis."""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d
import umap

from .types import Word, Timestamp, Embedding
from .base import BaseAnalyzer
from .storage import HDF5EmbeddingStore

logger = logging.getLogger(__name__)

@dataclass
class SemanticGravityField:
    """Represents a gravitational field in semantic space."""
    focal_word: str
    timestamp: str
    neighbor_words: List[str]
    gravitational_strengths: np.ndarray  # Force magnitudes
    gravitational_vectors: np.ndarray    # Force directions in high-D
    orbit_radii: np.ndarray             # Distance from focal word
    escape_velocities: np.ndarray       # Velocity needed to escape orbit
    stability_indices: np.ndarray       # Orbital stability measure

@dataclass
class SemanticBurst:
    """Represents a burst event in gravitational field terms."""
    timestamp: str
    focal_word: str
    burst_magnitude: float
    affected_words: List[str]
    gravitational_disruption: np.ndarray
    new_orbits_formed: List[str]
    orbits_destroyed: List[str]

class HighDimensionalFlowMetrics(BaseAnalyzer):
    """
    Computes semantic flow metrics in full high-dimensional embedding space.
    
    This engine operates on the complete embeddings from HDF5 storage,
    computing gravitational fields, orbital mechanics, and burst dynamics
    before any dimensionality reduction for visualization.
    """
    
    def __init__(self, embedding_store: HDF5EmbeddingStore):
        super().__init__("HighDimensionalFlowMetrics")
        self.store = embedding_store
        self.cache = {}
        
        # Physical constants for semantic gravity
        self.semantic_G = 1.0  # Gravitational constant
        self.burst_threshold = 2.0  # Burst detection threshold
        self.stability_threshold = 0.5  # Orbital stability threshold
    
    def compute_semantic_flow(self, word: str, t1: str, t2: str, 
                            k_neighbors: int = 50) -> Dict[str, Any]:
        """
        Compute semantic flow between two timestamps using gravitational field theory.
        
        Returns displacement, velocity, acceleration, and orbital changes.
        """
        cache_key = f"flow:{word}:{t1}:{t2}:{k_neighbors}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get embeddings at both timestamps
        emb1 = self.store.get_embedding(word, t1)
        emb2 = self.store.get_embedding(word, t2)
        
        if emb1 is None or emb2 is None:
            return {'error': f'Missing embeddings for {word}'}
        
        # Compute displacement vector in high-D space
        displacement = emb2 - emb1
        displacement_magnitude = np.linalg.norm(displacement)
        
        # Get gravitational fields at both timestamps
        field1 = self.compute_gravitational_field(word, t1, k_neighbors)
        field2 = self.compute_gravitational_field(word, t2, k_neighbors)
        
        # Compute velocity (assuming unit time step)
        velocity = displacement
        velocity_magnitude = displacement_magnitude
        
        # Compute acceleration from gravitational field change
        if field1 and field2:
            # Compare gravitational influences
            force_change = self._compute_force_change(field1, field2)
            acceleration = force_change
        else:
            acceleration = np.zeros_like(displacement)
        
        # Orbital analysis
        orbital_changes = self._analyze_orbital_changes(field1, field2)
        
        flow_data = {
            'word': word,
            'timestamps': [t1, t2],
            'displacement': displacement,
            'displacement_magnitude': displacement_magnitude,
            'velocity': velocity,
            'velocity_magnitude': velocity_magnitude,
            'acceleration': acceleration,
            'acceleration_magnitude': np.linalg.norm(acceleration),
            'orbital_changes': orbital_changes,
            'gravitational_field_t1': field1,
            'gravitational_field_t2': field2
        }
        
        self.cache[cache_key] = flow_data
        return flow_data
    
    def compute_gravitational_field(self, focal_word: str, timestamp: str, 
                                  k: int = 50) -> Optional[SemanticGravityField]:
        """
        Compute gravitational field around a focal word in high-D space.
        
        Each word acts as a gravitational body with mass proportional to
        semantic coherence and centrality in the embedding space.
        """
        cache_key = f"gravity:{focal_word}:{timestamp}:{k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get focal word embedding
        focal_embedding = self.store.get_embedding(focal_word, timestamp)
        if focal_embedding is None:
            return None
        
        # Get vocabulary and neighbor embeddings
        vocab = self.store.get_vocabulary(timestamp)
        if len(vocab) <= k:
            neighbor_words = [w for w in vocab if w != focal_word]
        else:
            # Find k-nearest neighbors efficiently
            neighbor_words = self._find_k_neighbors(focal_word, timestamp, k)
        
        if not neighbor_words:
            return None
        
        # Get neighbor embeddings
        neighbor_embeddings = self.store.get_embeddings_batch(neighbor_words, timestamp)
        
        # Compute gravitational field
        gravitational_strengths = []
        gravitational_vectors = []
        orbit_radii = []
        escape_velocities = []
        stability_indices = []
        
        valid_neighbors = []
        
        for word in neighbor_words:
            if word not in neighbor_embeddings:
                continue
                
            neighbor_emb = neighbor_embeddings[word]
            
            # Compute distance (orbital radius)
            distance_vector = neighbor_emb - focal_embedding
            distance = np.linalg.norm(distance_vector)
            
            if distance == 0:
                continue
            
            # Compute gravitational force (inversely proportional to distance squared)
            # Mass is proportional to embedding norm (semantic weight)
            focal_mass = np.linalg.norm(focal_embedding)
            neighbor_mass = np.linalg.norm(neighbor_emb)
            
            gravitational_force = (self.semantic_G * focal_mass * neighbor_mass) / (distance ** 2)
            
            # Force direction (unit vector from neighbor to focal)
            force_direction = -distance_vector / distance
            gravitational_vector = gravitational_force * force_direction
            
            # Compute escape velocity
            escape_velocity = np.sqrt(2 * self.semantic_G * focal_mass / distance)
            
            # Compute orbital stability (based on local curvature)
            stability = self._compute_orbital_stability(
                focal_embedding, neighbor_emb, neighbor_embeddings, timestamp
            )
            
            gravitational_strengths.append(gravitational_force)
            gravitational_vectors.append(gravitational_vector)
            orbit_radii.append(distance)
            escape_velocities.append(escape_velocity)
            stability_indices.append(stability)
            valid_neighbors.append(word)
        
        if not valid_neighbors:
            return None
        
        gravity_field = SemanticGravityField(
            focal_word=focal_word,
            timestamp=timestamp,
            neighbor_words=valid_neighbors,
            gravitational_strengths=np.array(gravitational_strengths),
            gravitational_vectors=np.array(gravitational_vectors),
            orbit_radii=np.array(orbit_radii),
            escape_velocities=np.array(escape_velocities),
            stability_indices=np.array(stability_indices)
        )
        
        self.cache[cache_key] = gravity_field
        return gravity_field
    
    def detect_semantic_bursts(self, focal_word: str, timestamps: List[str],
                             k_neighbors: int = 50) -> List[SemanticBurst]:
        """
        Detect burst events as sudden gravitational field disruptions.
        
        Bursts occur when the gravitational field around a word changes
        dramatically, indicating rapid semantic evolution.
        """
        bursts = []
        
        for i in range(1, len(timestamps)):
            t1, t2 = timestamps[i-1], timestamps[i]
            
            # Get gravitational fields
            field1 = self.compute_gravitational_field(focal_word, t1, k_neighbors)
            field2 = self.compute_gravitational_field(focal_word, t2, k_neighbors)
            
            if field1 is None or field2 is None:
                continue
            
            # Compute field disruption magnitude
            disruption = self._compute_field_disruption(field1, field2)
            
            if disruption['magnitude'] > self.burst_threshold:
                burst = SemanticBurst(
                    timestamp=t2,
                    focal_word=focal_word,
                    burst_magnitude=disruption['magnitude'],
                    affected_words=disruption['affected_words'],
                    gravitational_disruption=disruption['field_change'],
                    new_orbits_formed=disruption['new_orbits'],
                    orbits_destroyed=disruption['destroyed_orbits']
                )
                bursts.append(burst)
        
        return bursts
    
    def compute_temporal_interpolation(self, focal_word: str, timestamps: List[str],
                                     target_fps: int = 30) -> Dict[str, Any]:
        """
        Compute smooth temporal interpolation for gravitational field animation.
        
        Creates intermediate frames between actual timestamps for fluid visualization.
        """
        if len(timestamps) < 2:
            return {'error': 'Need at least 2 timestamps for interpolation'}
        
        # Get gravitational fields at all timestamps
        fields = []
        valid_timestamps = []
        
        for timestamp in timestamps:
            field = self.compute_gravitational_field(focal_word, timestamp)
            if field is not None:
                fields.append(field)
                valid_timestamps.append(timestamp)
        
        if len(fields) < 2:
            return {'error': 'Need at least 2 valid gravitational fields'}
        
        # Create interpolation data
        interpolation_data = {
            'original_timestamps': valid_timestamps,
            'original_fields': fields,
            'interpolated_frames': []
        }
        
        # Generate intermediate timestamps
        total_frames = (len(valid_timestamps) - 1) * target_fps
        time_indices = np.linspace(0, len(valid_timestamps) - 1, total_frames)
        
        # Interpolate gravitational field properties
        for t_idx in time_indices:
            frame_data = self._interpolate_gravitational_field(fields, t_idx)
            interpolation_data['interpolated_frames'].append(frame_data)
        
        return interpolation_data
    
    def _find_k_neighbors(self, focal_word: str, timestamp: str, k: int) -> List[str]:
        """Find k-nearest neighbors efficiently using similarity."""
        focal_embedding = self.store.get_embedding(focal_word, timestamp)
        vocab = self.store.get_vocabulary(timestamp)
        
        # Sample vocabulary if too large
        if len(vocab) > 1000:
            vocab = np.random.choice(vocab, 1000, replace=False).tolist()
        
        similarities = []
        words = []
        
        for word in vocab:
            if word == focal_word:
                continue
            
            embedding = self.store.get_embedding(word, timestamp)
            if embedding is not None:
                # Cosine similarity
                sim = np.dot(focal_embedding, embedding) / (
                    np.linalg.norm(focal_embedding) * np.linalg.norm(embedding)
                )
                similarities.append(sim)
                words.append(word)
        
        # Get top-k
        if len(similarities) <= k:
            return words
        
        top_indices = np.argsort(similarities)[-k:]
        return [words[i] for i in top_indices]
    
    def _compute_orbital_stability(self, focal_emb: np.ndarray, neighbor_emb: np.ndarray,
                                 all_embeddings: Dict[str, np.ndarray], 
                                 timestamp: str) -> float:
        """Compute orbital stability based on local embedding curvature."""
        # Simple stability measure based on local density and uniformity
        distance = np.linalg.norm(neighbor_emb - focal_emb)
        
        # Count nearby neighbors within 2x distance
        nearby_count = 0
        for emb in all_embeddings.values():
            if np.linalg.norm(emb - neighbor_emb) < 2 * distance:
                nearby_count += 1
        
        # Stability inversely related to local crowding
        stability = 1.0 / (1.0 + nearby_count * 0.1)
        return stability
    
    def _compute_force_change(self, field1: SemanticGravityField, 
                            field2: SemanticGravityField) -> np.ndarray:
        """Compute change in gravitational forces between two fields."""
        # Find common words
        common_words = set(field1.neighbor_words) & set(field2.neighbor_words)
        
        if not common_words:
            return np.zeros(field1.gravitational_vectors.shape[1])
        
        # Compute force difference
        force_diff = np.zeros(field1.gravitational_vectors.shape[1])
        
        for word in common_words:
            idx1 = field1.neighbor_words.index(word)
            idx2 = field2.neighbor_words.index(word)
            
            force_diff += field2.gravitational_vectors[idx2] - field1.gravitational_vectors[idx1]
        
        return force_diff / len(common_words)
    
    def _analyze_orbital_changes(self, field1: Optional[SemanticGravityField],
                               field2: Optional[SemanticGravityField]) -> Dict[str, Any]:
        """Analyze changes in orbital structure between two gravitational fields."""
        if field1 is None or field2 is None:
            return {'error': 'Missing gravitational field data'}
        
        words1 = set(field1.neighbor_words)
        words2 = set(field2.neighbor_words)
        
        # Orbital changes
        new_orbits = words2 - words1
        destroyed_orbits = words1 - words2
        stable_orbits = words1 & words2
        
        # Stability changes for stable orbits
        stability_changes = {}
        for word in stable_orbits:
            idx1 = field1.neighbor_words.index(word)
            idx2 = field2.neighbor_words.index(word)
            
            stability_change = field2.stability_indices[idx2] - field1.stability_indices[idx1]
            stability_changes[word] = stability_change
        
        return {
            'new_orbits': list(new_orbits),
            'destroyed_orbits': list(destroyed_orbits),
            'stable_orbits': list(stable_orbits),
            'stability_changes': stability_changes,
            'orbital_stability_delta': np.mean(list(stability_changes.values())) if stability_changes else 0.0
        }
    
    def _compute_field_disruption(self, field1: SemanticGravityField,
                                field2: SemanticGravityField) -> Dict[str, Any]:
        """Compute the magnitude of gravitational field disruption."""
        # Find overlapping words
        words1 = set(field1.neighbor_words)
        words2 = set(field2.neighbor_words)
        common_words = words1 & words2
        
        disruption_magnitude = 0.0
        field_changes = []
        
        # Compute disruption for common words
        for word in common_words:
            idx1 = field1.neighbor_words.index(word)
            idx2 = field2.neighbor_words.index(word)
            
            # Force magnitude change
            force_change = abs(field2.gravitational_strengths[idx2] - 
                             field1.gravitational_strengths[idx1])
            
            # Orbital radius change
            radius_change = abs(field2.orbit_radii[idx2] - field1.orbit_radii[idx1])
            
            # Combined disruption
            word_disruption = force_change + radius_change
            disruption_magnitude += word_disruption
            field_changes.append(word_disruption)
        
        # Add disruption from new/destroyed orbits
        orbit_changes = len(words1.symmetric_difference(words2))
        disruption_magnitude += orbit_changes * 0.5
        
        return {
            'magnitude': disruption_magnitude,
            'affected_words': list(common_words),
            'field_change': np.array(field_changes),
            'new_orbits': list(words2 - words1),
            'destroyed_orbits': list(words1 - words2),
            'orbit_change_count': orbit_changes
        }
    
    def _interpolate_gravitational_field(self, fields: List[SemanticGravityField], 
                                       t_index: float) -> Dict[str, Any]:
        """Interpolate gravitational field properties at fractional time index."""
        # Find surrounding fields
        t_floor = int(np.floor(t_index))
        t_ceil = int(np.ceil(t_index))
        
        if t_floor == t_ceil:
            # Exact timestamp
            return {'field': fields[t_floor], 'interpolation_weight': 1.0}
        
        # Interpolation weight
        weight = t_index - t_floor
        
        field1 = fields[t_floor]
        field2 = fields[t_ceil]
        
        # Find common words for interpolation
        common_words = set(field1.neighbor_words) & set(field2.neighbor_words)
        
        interpolated_data = {
            'timestamp_fraction': t_index,
            'interpolation_weight': weight,
            'common_words': list(common_words),
            'interpolated_properties': {}
        }
        
        # Interpolate properties for common words
        for word in common_words:
            idx1 = field1.neighbor_words.index(word)
            idx2 = field2.neighbor_words.index(word)
            
            # Linear interpolation of key properties
            interpolated_data['interpolated_properties'][word] = {
                'gravitational_strength': (1 - weight) * field1.gravitational_strengths[idx1] + 
                                        weight * field2.gravitational_strengths[idx2],
                'orbit_radius': (1 - weight) * field1.orbit_radii[idx1] + 
                              weight * field2.orbit_radii[idx2],
                'stability_index': (1 - weight) * field1.stability_indices[idx1] + 
                                 weight * field2.stability_indices[idx2]
            }
        
        return interpolated_data