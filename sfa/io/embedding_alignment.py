"""Embedding alignment and temporal synchronization for semantic flow analysis."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from scipy.spatial.distance import cosine
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import logging

from ..core.types import Word, Timestamp, Embedding
from ..core.base import BaseAnalyzer

class EmbeddingAlignment(BaseAnalyzer):
    """Handles alignment of embeddings across different time periods and sources"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("EmbeddingAlignment")
        self.embeddings = embeddings_store
        self.config = config
        
        # Alignment parameters
        self.alignment_config = {
            'alignment_method': 'procrustes',  # procrustes, orthogonal, linear, nonlinear
            'anchor_words': None,  # Words to use as anchors for alignment
            'min_anchor_words': 10,
            'max_anchor_words': 100,
            'stability_threshold': 0.8,  # Threshold for considering words stable
            'alignment_regularization': 0.01,
            'iterative_refinement': True,
            'max_iterations': 10,
            'convergence_threshold': 1e-6,
            'preserve_distances': True,
            'orthogonal_constraint': True
        }
        
        # Alignment state
        self.alignment_matrices = {}  # timestamp -> transformation matrix
        self.anchor_words = set()
        self.alignment_quality = {}  # timestamp -> quality metrics
        self.reference_embedding_space = None
        self.aligned_embeddings = {}  # timestamp -> aligned embeddings
        
    def analyze(self) -> Dict[str, Any]:
        """Standard analyze method for BaseAnalyzer compatibility"""
        # Perform comprehensive alignment analysis
        alignment_analysis = self.align_temporal_embeddings()
        
        return {
            'alignment_analysis': alignment_analysis,
            'alignment_quality': self.alignment_quality,
            'anchor_words': list(self.anchor_words),
            'alignment_config': self.alignment_config
        }
    
    def align_temporal_embeddings(self) -> Dict[str, Any]:
        """Align embeddings across all timestamps"""
        timestamps = self.embeddings.get_timestamps()
        
        if len(timestamps) < 2:
            return {'error': 'Need at least 2 timestamps for alignment'}
        
        # Step 1: Identify anchor words
        self._identify_anchor_words(timestamps)
        
        # Step 2: Establish reference space
        self._establish_reference_space(timestamps)
        
        # Step 3: Compute alignment transformations
        alignment_results = self._compute_alignment_transformations(timestamps)
        
        # Step 4: Apply alignments and validate
        validation_results = self._validate_alignments(timestamps)
        
        # Step 5: Analyze alignment quality
        quality_analysis = self._analyze_alignment_quality(timestamps)
        
        return {
            'anchor_selection': {
                'num_anchor_words': len(self.anchor_words),
                'anchor_words': list(self.anchor_words),
                'anchor_quality': self._assess_anchor_quality()
            },
            'alignment_transformations': alignment_results,
            'validation_results': validation_results,
            'quality_analysis': quality_analysis,
            'aligned_timestamps': timestamps
        }
    
    def _identify_anchor_words(self, timestamps: List[Timestamp]) -> None:
        """Identify stable words to use as anchors for alignment"""
        # Get words present across all timestamps
        all_vocabularies = [set(self.embeddings.get_vocabulary(t)) for t in timestamps]
        common_words = set.intersection(*all_vocabularies)
        
        if len(common_words) < self.alignment_config['min_anchor_words']:
            logging.warning(f"Only {len(common_words)} common words found, "
                          f"minimum required: {self.alignment_config['min_anchor_words']}")
            self.anchor_words = common_words
            return
        
        # Assess stability of common words
        word_stability = self._compute_word_stability(common_words, timestamps)
        
        # Select most stable words as anchors
        stable_words = [
            word for word, stability in word_stability.items()
            if stability >= self.alignment_config['stability_threshold']
        ]
        
        # Limit to maximum number of anchors
        max_anchors = self.alignment_config['max_anchor_words']
        if len(stable_words) > max_anchors:
            # Sort by stability and take top anchors
            sorted_words = sorted(stable_words, 
                                key=lambda w: word_stability[w], 
                                reverse=True)
            stable_words = sorted_words[:max_anchors]
        
        self.anchor_words = set(stable_words)
        
        logging.info(f"Selected {len(self.anchor_words)} anchor words for alignment")
    
    def _compute_word_stability(self, words: Set[Word], 
                               timestamps: List[Timestamp]) -> Dict[Word, float]:
        """Compute stability score for each word across timestamps"""
        word_stability = {}
        
        for word in words:
            embeddings = []
            for timestamp in timestamps:
                emb = self.embeddings.get_embedding(word, timestamp)
                if emb is not None:
                    embeddings.append(emb)
            
            if len(embeddings) < 2:
                word_stability[word] = 0.0
                continue
            
            # Compute pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = 1 - cosine(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            # Stability = mean pairwise similarity
            stability = np.mean(similarities) if similarities else 0.0
            word_stability[word] = stability
        
        return word_stability
    
    def _establish_reference_space(self, timestamps: List[Timestamp]) -> None:
        """Establish reference embedding space for alignment"""
        # Use first timestamp as reference by default
        reference_timestamp = timestamps[0]
        
        # Get embeddings for anchor words in reference space
        reference_embeddings = {}
        for word in self.anchor_words:
            emb = self.embeddings.get_embedding(word, reference_timestamp)
            if emb is not None:
                reference_embeddings[word] = emb
        
        if not reference_embeddings:
            raise ValueError("No anchor word embeddings found in reference timestamp")
        
        # Store reference space
        self.reference_embedding_space = {
            'timestamp': reference_timestamp,
            'embeddings': reference_embeddings,
            'dimension': len(next(iter(reference_embeddings.values())))
        }
        
        logging.info(f"Established reference space with {len(reference_embeddings)} anchor embeddings")
    
    def _compute_alignment_transformations(self, timestamps: List[Timestamp]) -> Dict[str, Any]:
        """Compute transformation matrices for each timestamp"""
        alignment_results = {}
        reference_timestamp = self.reference_embedding_space['timestamp']
        
        for timestamp in timestamps:
            if timestamp == reference_timestamp:
                # Identity transformation for reference
                ref_dim = self.reference_embedding_space['dimension']
                self.alignment_matrices[timestamp] = np.eye(ref_dim)
                alignment_results[timestamp] = {
                    'transformation_type': 'identity',
                    'quality': 1.0
                }
                continue
            
            # Get anchor embeddings for current timestamp
            current_embeddings = {}
            for word in self.anchor_words:
                emb = self.embeddings.get_embedding(word, timestamp)
                if emb is not None:
                    current_embeddings[word] = emb
            
            if len(current_embeddings) < self.alignment_config['min_anchor_words']:
                logging.warning(f"Insufficient anchor words for timestamp {timestamp}")
                continue
            
            # Compute transformation
            transformation_result = self._compute_transformation(
                self.reference_embedding_space['embeddings'],
                current_embeddings
            )
            
            self.alignment_matrices[timestamp] = transformation_result['matrix']
            alignment_results[timestamp] = transformation_result
        
        return alignment_results
    
    def _compute_transformation(self, reference_embeddings: Dict[Word, Embedding],
                               target_embeddings: Dict[Word, Embedding]) -> Dict[str, Any]:
        """Compute transformation matrix between embedding spaces"""
        # Find common words
        common_words = set(reference_embeddings.keys()) & set(target_embeddings.keys())
        
        if len(common_words) < 3:
            raise ValueError("Need at least 3 common words for transformation")
        
        # Create matrices
        ref_matrix = np.array([reference_embeddings[word] for word in common_words])
        target_matrix = np.array([target_embeddings[word] for word in common_words])
        
        method = self.alignment_config['alignment_method']
        
        if method == 'procrustes':
            return self._procrustes_alignment(ref_matrix, target_matrix)
        elif method == 'orthogonal':
            return self._orthogonal_alignment(ref_matrix, target_matrix)
        elif method == 'linear':
            return self._linear_alignment(ref_matrix, target_matrix)
        elif method == 'nonlinear':
            return self._nonlinear_alignment(ref_matrix, target_matrix)
        else:
            raise ValueError(f"Unknown alignment method: {method}")
    
    def _procrustes_alignment(self, reference: np.ndarray, 
                             target: np.ndarray) -> Dict[str, Any]:
        """Compute Procrustes alignment transformation"""
        # Center the data
        ref_centered = reference - np.mean(reference, axis=0)
        target_centered = target - np.mean(target, axis=0)
        
        # Compute SVD
        H = target_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = U @ Vt
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        
        # Compute scaling factor if not preserving distances
        if not self.alignment_config['preserve_distances']:
            scale = np.trace(S) / np.trace(target_centered.T @ target_centered)
        else:
            scale = 1.0
        
        # Full transformation matrix (includes scaling and rotation)
        transformation_matrix = scale * R
        
        # Compute alignment quality
        aligned_target = target_centered @ transformation_matrix.T
        quality = self._compute_alignment_quality_score(ref_centered, aligned_target)
        
        return {
            'matrix': transformation_matrix,
            'method': 'procrustes',
            'quality': quality,
            'scale': scale,
            'rotation': R,
            'explained_variance': np.sum(S) / np.sum(np.diag(target_centered.T @ target_centered))
        }
    
    def _orthogonal_alignment(self, reference: np.ndarray, 
                             target: np.ndarray) -> Dict[str, Any]:
        """Compute orthogonal alignment transformation"""
        # This is similar to Procrustes but with additional orthogonality constraints
        return self._procrustes_alignment(reference, target)
    
    def _linear_alignment(self, reference: np.ndarray, 
                         target: np.ndarray) -> Dict[str, Any]:
        """Compute general linear alignment transformation"""
        # Solve for transformation matrix W such that target @ W ≈ reference
        # Using least squares: W = (target.T @ target)^(-1) @ target.T @ reference
        
        try:
            W = np.linalg.lstsq(target, reference, rcond=None)[0]
            
            # Compute quality
            aligned_target = target @ W
            quality = self._compute_alignment_quality_score(reference, aligned_target)
            
            return {
                'matrix': W,
                'method': 'linear',
                'quality': quality,
                'condition_number': np.linalg.cond(target)
            }
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            W = np.linalg.pinv(target) @ reference
            aligned_target = target @ W
            quality = self._compute_alignment_quality_score(reference, aligned_target)
            
            return {
                'matrix': W,
                'method': 'linear_pinv',
                'quality': quality,
                'condition_number': np.inf
            }
    
    def _nonlinear_alignment(self, reference: np.ndarray, 
                            target: np.ndarray) -> Dict[str, Any]:
        """Compute nonlinear alignment transformation using optimization"""
        dim = reference.shape[1]
        
        # Initialize with linear solution
        linear_result = self._linear_alignment(reference, target)
        initial_matrix = linear_result['matrix']
        
        # Define objective function
        def objective(matrix_flat):
            matrix = matrix_flat.reshape((dim, dim))
            aligned = target @ matrix
            return np.sum((reference - aligned) ** 2)
        
        # Add regularization
        def objective_with_reg(matrix_flat):
            matrix = matrix_flat.reshape((dim, dim))
            aligned = target @ matrix
            mse = np.sum((reference - aligned) ** 2)
            reg = self.alignment_config['alignment_regularization'] * np.sum(matrix ** 2)
            return mse + reg
        
        # Optimize
        try:
            result = minimize(objective_with_reg, initial_matrix.flatten(), 
                            method='L-BFGS-B')
            
            if result.success:
                optimal_matrix = result.x.reshape((dim, dim))
                aligned_target = target @ optimal_matrix
                quality = self._compute_alignment_quality_score(reference, aligned_target)
                
                return {
                    'matrix': optimal_matrix,
                    'method': 'nonlinear',
                    'quality': quality,
                    'optimization_success': True,
                    'iterations': result.nit
                }
            else:
                # Fallback to linear
                return linear_result
        except Exception:
            # Fallback to linear
            return linear_result
    
    def _compute_alignment_quality_score(self, reference: np.ndarray, 
                                       aligned: np.ndarray) -> float:
        """Compute quality score for alignment"""
        # Compute mean squared error
        mse = np.mean((reference - aligned) ** 2)
        
        # Compute explained variance
        total_variance = np.var(reference)
        residual_variance = np.var(reference - aligned)
        
        if total_variance > 0:
            explained_variance = 1 - (residual_variance / total_variance)
        else:
            explained_variance = 0.0
        
        # Combine metrics (higher is better)
        quality = max(0.0, explained_variance)
        
        return quality
    
    def _validate_alignments(self, timestamps: List[Timestamp]) -> Dict[str, Any]:
        """Validate alignment quality across timestamps"""
        validation_results = {}
        
        for timestamp in timestamps:
            if timestamp not in self.alignment_matrices:
                continue
            
            # Apply alignment and measure quality
            validation_metrics = self._validate_single_alignment(timestamp)
            validation_results[timestamp] = validation_metrics
        
        # Compute overall validation metrics
        if validation_results:
            all_qualities = [v['alignment_quality'] for v in validation_results.values()]
            validation_results['overall'] = {
                'mean_quality': np.mean(all_qualities),
                'min_quality': np.min(all_qualities),
                'max_quality': np.max(all_qualities),
                'std_quality': np.std(all_qualities)
            }
        
        return validation_results
    
    def _validate_single_alignment(self, timestamp: Timestamp) -> Dict[str, Any]:
        """Validate alignment for a single timestamp"""
        if timestamp not in self.alignment_matrices:
            return {'error': 'No alignment matrix found'}
        
        transformation = self.alignment_matrices[timestamp]
        
        # Test on anchor words
        anchor_errors = []
        for word in self.anchor_words:
            target_emb = self.embeddings.get_embedding(word, timestamp)
            ref_emb = self.reference_embedding_space['embeddings'].get(word)
            
            if target_emb is not None and ref_emb is not None:
                aligned_emb = target_emb @ transformation.T
                error = np.linalg.norm(ref_emb - aligned_emb)
                anchor_errors.append(error)
        
        # Test on non-anchor words (if available)
        non_anchor_words = (set(self.embeddings.get_vocabulary(timestamp)) - 
                           self.anchor_words)
        non_anchor_errors = []
        
        for word in list(non_anchor_words)[:20]:  # Sample 20 words
            target_emb = self.embeddings.get_embedding(word, timestamp)
            if target_emb is not None:
                aligned_emb = target_emb @ transformation.T
                # We don't have reference for non-anchor words, so we measure consistency
                # by checking if aligned embedding maintains reasonable properties
                norm_diff = abs(np.linalg.norm(aligned_emb) - np.linalg.norm(target_emb))
                non_anchor_errors.append(norm_diff)
        
        # Compute validation metrics
        validation_metrics = {
            'alignment_quality': 1.0 / (1.0 + np.mean(anchor_errors)) if anchor_errors else 0.0,
            'anchor_word_error': {
                'mean': np.mean(anchor_errors) if anchor_errors else float('inf'),
                'std': np.std(anchor_errors) if anchor_errors else 0.0,
                'max': np.max(anchor_errors) if anchor_errors else float('inf')
            },
            'non_anchor_consistency': {
                'mean_norm_change': np.mean(non_anchor_errors) if non_anchor_errors else 0.0,
                'std_norm_change': np.std(non_anchor_errors) if non_anchor_errors else 0.0
            },
            'transformation_properties': {
                'determinant': np.linalg.det(transformation),
                'condition_number': np.linalg.cond(transformation),
                'frobenius_norm': np.linalg.norm(transformation, 'fro')
            }
        }
        
        return validation_metrics
    
    def _analyze_alignment_quality(self, timestamps: List[Timestamp]) -> Dict[str, Any]:
        """Analyze overall alignment quality"""
        quality_analysis = {
            'temporal_consistency': self._measure_temporal_consistency(timestamps),
            'anchor_stability': self._measure_anchor_stability(timestamps),
            'semantic_preservation': self._measure_semantic_preservation(timestamps),
            'alignment_drift': self._measure_alignment_drift(timestamps)
        }
        
        return quality_analysis
    
    def _measure_temporal_consistency(self, timestamps: List[Timestamp]) -> Dict[str, float]:
        """Measure consistency of alignments across time"""
        if len(timestamps) < 3:
            return {'error': 'Need at least 3 timestamps'}
        
        # Check if transformations change smoothly over time
        transformation_changes = []
        
        for i in range(1, len(timestamps)):
            prev_timestamp = timestamps[i-1]
            curr_timestamp = timestamps[i]
            
            if (prev_timestamp in self.alignment_matrices and 
                curr_timestamp in self.alignment_matrices):
                
                prev_matrix = self.alignment_matrices[prev_timestamp]
                curr_matrix = self.alignment_matrices[curr_timestamp]
                
                # Measure change in transformation
                matrix_diff = np.linalg.norm(curr_matrix - prev_matrix, 'fro')
                transformation_changes.append(matrix_diff)
        
        if transformation_changes:
            return {
                'mean_change': np.mean(transformation_changes),
                'std_change': np.std(transformation_changes),
                'max_change': np.max(transformation_changes),
                'consistency_score': 1.0 / (1.0 + np.mean(transformation_changes))
            }
        else:
            return {'error': 'No valid transformation pairs'}
    
    def _measure_anchor_stability(self, timestamps: List[Timestamp]) -> Dict[str, float]:
        """Measure stability of anchor words after alignment"""
        anchor_stabilities = []
        
        for word in self.anchor_words:
            aligned_embeddings = []
            
            for timestamp in timestamps:
                if timestamp in self.alignment_matrices:
                    emb = self.embeddings.get_embedding(word, timestamp)
                    if emb is not None:
                        transformation = self.alignment_matrices[timestamp]
                        aligned_emb = emb @ transformation.T
                        aligned_embeddings.append(aligned_emb)
            
            if len(aligned_embeddings) >= 2:
                # Compute pairwise similarities
                similarities = []
                for i in range(len(aligned_embeddings)):
                    for j in range(i + 1, len(aligned_embeddings)):
                        sim = 1 - cosine(aligned_embeddings[i], aligned_embeddings[j])
                        similarities.append(sim)
                
                if similarities:
                    anchor_stabilities.append(np.mean(similarities))
        
        if anchor_stabilities:
            return {
                'mean_stability': np.mean(anchor_stabilities),
                'std_stability': np.std(anchor_stabilities),
                'min_stability': np.min(anchor_stabilities)
            }
        else:
            return {'error': 'No stable anchor words found'}
    
    def _measure_semantic_preservation(self, timestamps: List[Timestamp]) -> Dict[str, float]:
        """Measure preservation of semantic relationships after alignment"""
        # Sample word pairs to test semantic preservation
        test_pairs = []
        reference_timestamp = self.reference_embedding_space['timestamp']
        ref_vocab = set(self.embeddings.get_vocabulary(reference_timestamp))
        
        # Create test pairs from reference vocabulary
        ref_words = list(ref_vocab)[:50]  # Sample 50 words
        for i in range(min(20, len(ref_words))):
            for j in range(i + 1, min(i + 5, len(ref_words))):
                test_pairs.append((ref_words[i], ref_words[j]))
        
        preservation_scores = []
        
        for timestamp in timestamps:
            if timestamp == reference_timestamp or timestamp not in self.alignment_matrices:
                continue
            
            pair_correlations = []
            transformation = self.alignment_matrices[timestamp]
            
            for word1, word2 in test_pairs:
                # Get reference similarity
                ref_emb1 = self.embeddings.get_embedding(word1, reference_timestamp)
                ref_emb2 = self.embeddings.get_embedding(word2, reference_timestamp)
                
                # Get target similarity (after alignment)
                target_emb1 = self.embeddings.get_embedding(word1, timestamp)
                target_emb2 = self.embeddings.get_embedding(word2, timestamp)
                
                if all(emb is not None for emb in [ref_emb1, ref_emb2, target_emb1, target_emb2]):
                    ref_sim = 1 - cosine(ref_emb1, ref_emb2)
                    
                    aligned_emb1 = target_emb1 @ transformation.T
                    aligned_emb2 = target_emb2 @ transformation.T
                    aligned_sim = 1 - cosine(aligned_emb1, aligned_emb2)
                    
                    pair_correlations.append((ref_sim, aligned_sim))
            
            if pair_correlations:
                ref_sims, aligned_sims = zip(*pair_correlations)
                correlation = np.corrcoef(ref_sims, aligned_sims)[0, 1]
                if not np.isnan(correlation):
                    preservation_scores.append(correlation)
        
        if preservation_scores:
            return {
                'mean_preservation': np.mean(preservation_scores),
                'std_preservation': np.std(preservation_scores),
                'min_preservation': np.min(preservation_scores)
            }
        else:
            return {'error': 'Could not measure semantic preservation'}
    
    def _measure_alignment_drift(self, timestamps: List[Timestamp]) -> Dict[str, float]:
        """Measure cumulative drift in alignments over time"""
        if len(timestamps) < 3:
            return {'error': 'Need at least 3 timestamps'}
        
        reference_timestamp = self.reference_embedding_space['timestamp']
        drift_scores = []
        
        for i, timestamp in enumerate(timestamps):
            if timestamp == reference_timestamp or timestamp not in self.alignment_matrices:
                continue
            
            # Measure drift by comparing aligned embeddings to reference
            word_drifts = []
            transformation = self.alignment_matrices[timestamp]
            
            for word in list(self.anchor_words)[:20]:  # Sample anchor words
                ref_emb = self.reference_embedding_space['embeddings'].get(word)
                target_emb = self.embeddings.get_embedding(word, timestamp)
                
                if ref_emb is not None and target_emb is not None:
                    aligned_emb = target_emb @ transformation.T
                    drift = np.linalg.norm(ref_emb - aligned_emb)
                    word_drifts.append(drift)
            
            if word_drifts:
                timestamp_drift = np.mean(word_drifts)
                drift_scores.append((i, timestamp_drift))
        
        if len(drift_scores) >= 2:
            # Fit linear trend to drift scores
            indices, drifts = zip(*drift_scores)
            drift_trend = np.polyfit(indices, drifts, 1)[0]  # Slope
            
            return {
                'drift_trend': drift_trend,
                'mean_drift': np.mean(drifts),
                'max_drift': np.max(drifts),
                'drift_acceleration': drift_trend > 0
            }
        else:
            return {'error': 'Insufficient data for drift analysis'}
    
    def _assess_anchor_quality(self) -> Dict[str, float]:
        """Assess quality of selected anchor words"""
        if not self.anchor_words:
            return {'error': 'No anchor words selected'}
        
        timestamps = self.embeddings.get_timestamps()
        
        # Compute stability metrics for anchor words
        stability_scores = self._compute_word_stability(self.anchor_words, timestamps)
        
        quality_metrics = {
            'num_anchors': len(self.anchor_words),
            'mean_stability': np.mean(list(stability_scores.values())),
            'min_stability': min(stability_scores.values()),
            'std_stability': np.std(list(stability_scores.values()))
        }
        
        return quality_metrics
    
    def apply_alignment(self, word: Word, timestamp: Timestamp) -> Optional[Embedding]:
        """Apply alignment transformation to a word embedding"""
        if timestamp not in self.alignment_matrices:
            return None
        
        original_emb = self.embeddings.get_embedding(word, timestamp)
        if original_emb is None:
            return None
        
        transformation = self.alignment_matrices[timestamp]
        aligned_emb = original_emb @ transformation.T
        
        return aligned_emb
    
    def get_aligned_vocabulary(self, timestamp: Timestamp) -> List[Word]:
        """Get vocabulary for aligned timestamp"""
        return self.embeddings.get_vocabulary(timestamp)
    
    def get_alignment_transformation(self, timestamp: Timestamp) -> Optional[np.ndarray]:
        """Get alignment transformation matrix for timestamp"""
        return self.alignment_matrices.get(timestamp)
    
    def export_alignments(self) -> Dict[str, Any]:
        """Export alignment data for external use"""
        return {
            'alignment_matrices': {ts: matrix.tolist() for ts, matrix in self.alignment_matrices.items()},
            'anchor_words': list(self.anchor_words),
            'reference_space': {
                'timestamp': self.reference_embedding_space['timestamp'],
                'dimension': self.reference_embedding_space['dimension']
            },
            'alignment_config': self.alignment_config,
            'alignment_quality': self.alignment_quality
        }
    
    def import_alignments(self, alignment_data: Dict[str, Any]) -> None:
        """Import previously computed alignments"""
        if 'alignment_matrices' in alignment_data:
            self.alignment_matrices = {
                ts: np.array(matrix) 
                for ts, matrix in alignment_data['alignment_matrices'].items()
            }
        
        if 'anchor_words' in alignment_data:
            self.anchor_words = set(alignment_data['anchor_words'])
        
        if 'reference_space' in alignment_data:
            self.reference_embedding_space = alignment_data['reference_space']
        
        if 'alignment_config' in alignment_data:
            self.alignment_config.update(alignment_data['alignment_config'])
        
        if 'alignment_quality' in alignment_data:
            self.alignment_quality = alignment_data['alignment_quality']