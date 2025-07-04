"""Confidence band estimation for semantic flow analysis."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import logging
from collections import defaultdict

from ..core.types import Word, Timestamp, Embedding
from ..core.base import BaseProcessor
from ..core.embeddings import TemporalEmbeddingStore

logger = logging.getLogger(__name__)

class ConfidenceBandEstimator(BaseProcessor):
    """
    Estimates confidence bands for semantic trajectories and flow predictions.
    
    Supports multiple uncertainty quantification methods:
    - Bootstrap resampling
    - Jackknife estimation
    - Gaussian process uncertainty
    - Ensemble-based uncertainty
    - Local neighborhood variance
    """
    
    def __init__(self, method: str = 'bootstrap', confidence_level: float = 0.95):
        super().__init__("ConfidenceBandEstimator")
        self.method = method
        self.confidence_level = confidence_level
        
        # Bootstrap parameters
        self.bootstrap_samples = 1000
        self.bootstrap_fraction = 0.8
        
        # Gaussian process parameters
        self.gp_length_scale = 1.0
        self.gp_variance = 1.0
        self.gp_noise_level = 0.1
        
        # Neighborhood parameters
        self.neighborhood_size = 10
        self.distance_metric = 'cosine'
        
        # Results storage
        self.confidence_bands = {}
        self.uncertainty_estimates = {}
        
    def configure(self, **config) -> None:
        """Configure confidence band estimation."""
        super().configure(**config)
        
        if 'bootstrap_samples' in config:
            self.bootstrap_samples = config['bootstrap_samples']
        if 'bootstrap_fraction' in config:
            self.bootstrap_fraction = config['bootstrap_fraction']
        if 'gp_length_scale' in config:
            self.gp_length_scale = config['gp_length_scale']
        if 'neighborhood_size' in config:
            self.neighborhood_size = config['neighborhood_size']
    
    def process(self, embedding_store: TemporalEmbeddingStore, 
                target_words: Optional[List[Word]] = None) -> Dict[str, Any]:
        """
        Estimate confidence bands for word trajectories.
        
        Args:
            embedding_store: Temporal embedding store
            target_words: Words to analyze (None for all words)
            
        Returns:
            Dictionary with confidence band estimates
        """
        logger.info(f"Estimating confidence bands using {self.method} method")
        
        if target_words is None:
            # Get all persistent words
            target_words = list(embedding_store.get_persistent_words(min_appearances=3))
        
        results = {
            'confidence_bands': {},
            'uncertainty_estimates': {},
            'method': self.method,
            'confidence_level': self.confidence_level,
            'analyzed_words': len(target_words)
        }
        
        # Estimate confidence bands for each word
        for word in target_words:
            try:
                timeline = embedding_store.get_word_timeline(word)
                if len(timeline) < 3:
                    continue  # Need at least 3 points for meaningful confidence bands
                
                if self.method == 'bootstrap':
                    bands = self._bootstrap_confidence_bands(embedding_store, word)
                elif self.method == 'jackknife':
                    bands = self._jackknife_confidence_bands(embedding_store, word)
                elif self.method == 'gaussian_process':
                    bands = self._gp_confidence_bands(embedding_store, word)
                elif self.method == 'neighborhood':
                    bands = self._neighborhood_confidence_bands(embedding_store, word)
                elif self.method == 'ensemble':
                    bands = self._ensemble_confidence_bands(embedding_store, word)
                else:
                    logger.warning(f"Unknown method: {self.method}")
                    continue
                
                if bands is not None:
                    results['confidence_bands'][word] = bands
                    
            except Exception as e:
                logger.warning(f"Failed to estimate confidence bands for {word}: {e}")
        
        self.confidence_bands = results['confidence_bands']
        logger.info(f"Estimated confidence bands for {len(results['confidence_bands'])} words")
        
        return results
    
    def _bootstrap_confidence_bands(self, embedding_store: TemporalEmbeddingStore, 
                                   word: Word) -> Dict[str, Any]:
        """Estimate confidence bands using bootstrap resampling."""
        timeline = embedding_store.get_word_timeline(word)
        timestamps = [ts for ts, _ in timeline]
        embeddings = [emb for _, emb in timeline]
        
        if len(embeddings) < 3:
            return None
        
        # Bootstrap samples
        bootstrap_trajectories = []
        
        for _ in range(self.bootstrap_samples):
            # Sample with replacement
            sample_size = int(len(embeddings) * self.bootstrap_fraction)
            indices = np.random.choice(len(embeddings), size=sample_size, replace=True)
            
            sampled_timestamps = [timestamps[i] for i in indices]
            sampled_embeddings = [embeddings[i] for i in indices]
            
            # Sort by timestamp
            sorted_pairs = sorted(zip(sampled_timestamps, sampled_embeddings))
            sampled_timestamps = [ts for ts, _ in sorted_pairs]
            sampled_embeddings = [emb for _, emb in sorted_pairs]
            
            # Interpolate to original timeline if needed
            interpolated_trajectory = self._interpolate_trajectory(
                sampled_timestamps, sampled_embeddings, timestamps
            )
            
            if interpolated_trajectory is not None:
                bootstrap_trajectories.append(interpolated_trajectory)
        
        if not bootstrap_trajectories:
            return None
        
        # Compute confidence bands
        bootstrap_array = np.array(bootstrap_trajectories)
        
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bands = np.percentile(bootstrap_array, lower_percentile, axis=0)
        upper_bands = np.percentile(bootstrap_array, upper_percentile, axis=0)
        mean_trajectory = np.mean(bootstrap_array, axis=0)
        
        return {
            'timestamps': timestamps,
            'mean_trajectory': mean_trajectory.tolist(),
            'lower_bands': lower_bands.tolist(),
            'upper_bands': upper_bands.tolist(),
            'bootstrap_samples': len(bootstrap_trajectories),
            'confidence_level': self.confidence_level
        }
    
    def _jackknife_confidence_bands(self, embedding_store: TemporalEmbeddingStore,
                                   word: Word) -> Dict[str, Any]:
        """Estimate confidence bands using jackknife resampling."""
        timeline = embedding_store.get_word_timeline(word)
        timestamps = [ts for ts, _ in timeline]
        embeddings = [emb for _, emb in timeline]
        
        if len(embeddings) < 4:
            return None
        
        # Jackknife samples (leave-one-out)
        jackknife_trajectories = []
        
        for i in range(len(embeddings)):
            # Remove one sample
            jackknife_timestamps = timestamps[:i] + timestamps[i+1:]
            jackknife_embeddings = embeddings[:i] + embeddings[i+1:]
            
            # Interpolate to original timeline
            interpolated_trajectory = self._interpolate_trajectory(
                jackknife_timestamps, jackknife_embeddings, timestamps
            )
            
            if interpolated_trajectory is not None:
                jackknife_trajectories.append(interpolated_trajectory)
        
        if not jackknife_trajectories:
            return None
        
        # Compute confidence bands using jackknife variance estimation
        jackknife_array = np.array(jackknife_trajectories)
        n = len(jackknife_trajectories)
        
        # Jackknife mean
        jackknife_mean = np.mean(jackknife_array, axis=0)
        
        # Jackknife variance (bias-corrected)
        jackknife_var = ((n - 1) / n) * np.sum(
            (jackknife_array - jackknife_mean)**2, axis=0
        )
        jackknife_std = np.sqrt(jackknife_var)
        
        # Confidence bands using t-distribution
        t_value = stats.t.ppf(1 - (1 - self.confidence_level) / 2, df=n-1)
        margin = t_value * jackknife_std
        
        lower_bands = jackknife_mean - margin
        upper_bands = jackknife_mean + margin
        
        return {
            'timestamps': timestamps,
            'mean_trajectory': jackknife_mean.tolist(),
            'lower_bands': lower_bands.tolist(),
            'upper_bands': upper_bands.tolist(),
            'jackknife_samples': n,
            'confidence_level': self.confidence_level
        }
    
    def _gp_confidence_bands(self, embedding_store: TemporalEmbeddingStore,
                            word: Word) -> Dict[str, Any]:
        """Estimate confidence bands using Gaussian process regression."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        except ImportError:
            logger.warning("sklearn not available for Gaussian process")
            return self._bootstrap_confidence_bands(embedding_store, word)
        
        timeline = embedding_store.get_word_timeline(word)
        timestamps = [self._timestamp_to_numeric(ts) for ts, _ in timeline]
        embeddings = [emb for _, emb in timeline]
        
        if len(embeddings) < 3:
            return None
        
        # Prepare data
        X = np.array(timestamps).reshape(-1, 1)
        
        # Fit GP for each embedding dimension
        gp_results = []
        embedding_dim = len(embeddings[0])
        
        for dim in range(embedding_dim):
            y = np.array([emb[dim] for emb in embeddings])
            
            # Define kernel
            kernel = ConstantKernel(self.gp_variance) * RBF(self.gp_length_scale)
            
            # Fit Gaussian process
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.gp_noise_level**2,
                n_restarts_optimizer=3
            )
            gp.fit(X, y)
            
            # Predict with uncertainty
            y_pred, y_std = gp.predict(X, return_std=True)
            
            gp_results.append({
                'mean': y_pred,
                'std': y_std
            })
        
        # Combine results
        mean_trajectory = np.array([
            [gp_results[dim]['mean'][i] for dim in range(embedding_dim)]
            for i in range(len(timestamps))
        ])
        
        # Compute confidence bands
        z_value = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        lower_bands = np.array([
            [gp_results[dim]['mean'][i] - z_value * gp_results[dim]['std'][i] 
             for dim in range(embedding_dim)]
            for i in range(len(timestamps))
        ])
        
        upper_bands = np.array([
            [gp_results[dim]['mean'][i] + z_value * gp_results[dim]['std'][i] 
             for dim in range(embedding_dim)]
            for i in range(len(timestamps))
        ])
        
        return {
            'timestamps': [ts for ts, _ in timeline],
            'mean_trajectory': mean_trajectory.tolist(),
            'lower_bands': lower_bands.tolist(),
            'upper_bands': upper_bands.tolist(),
            'gp_parameters': {
                'length_scale': self.gp_length_scale,
                'variance': self.gp_variance,
                'noise_level': self.gp_noise_level
            },
            'confidence_level': self.confidence_level
        }
    
    def _neighborhood_confidence_bands(self, embedding_store: TemporalEmbeddingStore,
                                      word: Word) -> Dict[str, Any]:
        """Estimate confidence bands using local neighborhood variance."""
        timeline = embedding_store.get_word_timeline(word)
        timestamps = [ts for ts, _ in timeline]
        embeddings = [emb for _, emb in timeline]
        
        if len(embeddings) < 3:
            return None
        
        confidence_estimates = []
        
        for i, (timestamp, embedding) in enumerate(timeline):
            # Find semantic neighbors at this timestamp
            vocab = embedding_store.get_vocabulary(timestamp)
            if len(vocab) < self.neighborhood_size:
                neighbors = list(vocab)
            else:
                # Find most similar words
                similar_words = embedding_store.find_similar_words(
                    word, timestamp, k=self.neighborhood_size
                )
                neighbors = [w for w, _ in similar_words]
            
            # Get neighbor embeddings
            neighbor_embeddings = []
            for neighbor in neighbors:
                neighbor_emb = embedding_store.get_embedding(neighbor, timestamp)
                if neighbor_emb is not None:
                    neighbor_embeddings.append(neighbor_emb)
            
            if len(neighbor_embeddings) < 3:
                # Use global variance estimate
                all_embeddings = list(embedding_store.get_embeddings(timestamp).values())
                if len(all_embeddings) >= 3:
                    neighbor_embeddings = all_embeddings
                else:
                    confidence_estimates.append(np.zeros_like(embedding))
                    continue
            
            # Compute local variance
            neighbor_array = np.array(neighbor_embeddings)
            local_variance = np.var(neighbor_array, axis=0)
            local_std = np.sqrt(local_variance)
            
            confidence_estimates.append(local_std)
        
        # Smooth confidence estimates
        confidence_array = np.array(confidence_estimates)
        smoothed_confidence = self._smooth_confidence_estimates(confidence_array)
        
        # Compute confidence bands
        mean_trajectory = np.array(embeddings)
        z_value = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        lower_bands = mean_trajectory - z_value * smoothed_confidence
        upper_bands = mean_trajectory + z_value * smoothed_confidence
        
        return {
            'timestamps': timestamps,
            'mean_trajectory': mean_trajectory.tolist(),
            'lower_bands': lower_bands.tolist(),
            'upper_bands': upper_bands.tolist(),
            'neighborhood_size': self.neighborhood_size,
            'confidence_level': self.confidence_level
        }
    
    def _ensemble_confidence_bands(self, embedding_store: TemporalEmbeddingStore,
                                  word: Word) -> Dict[str, Any]:
        """Estimate confidence bands using ensemble of methods."""
        # Run multiple methods
        methods = ['bootstrap', 'jackknife', 'neighborhood']
        ensemble_results = []
        
        original_method = self.method
        
        for method in methods:
            try:
                self.method = method
                
                if method == 'bootstrap':
                    result = self._bootstrap_confidence_bands(embedding_store, word)
                elif method == 'jackknife':
                    result = self._jackknife_confidence_bands(embedding_store, word)
                elif method == 'neighborhood':
                    result = self._neighborhood_confidence_bands(embedding_store, word)
                
                if result is not None:
                    ensemble_results.append(result)
                    
            except Exception as e:
                logger.debug(f"Ensemble method {method} failed: {e}")
                continue
        
        self.method = original_method
        
        if not ensemble_results:
            return None
        
        # Combine ensemble results
        timestamps = ensemble_results[0]['timestamps']
        
        # Average the trajectories and confidence bands
        mean_trajectories = [np.array(result['mean_trajectory']) for result in ensemble_results]
        lower_bands_list = [np.array(result['lower_bands']) for result in ensemble_results]
        upper_bands_list = [np.array(result['upper_bands']) for result in ensemble_results]
        
        ensemble_mean = np.mean(mean_trajectories, axis=0)
        ensemble_lower = np.mean(lower_bands_list, axis=0)
        ensemble_upper = np.mean(upper_bands_list, axis=0)
        
        # Compute ensemble uncertainty (variance across methods)
        method_variance = np.var(mean_trajectories, axis=0)
        ensemble_uncertainty = np.sqrt(method_variance)
        
        return {
            'timestamps': timestamps,
            'mean_trajectory': ensemble_mean.tolist(),
            'lower_bands': ensemble_lower.tolist(),
            'upper_bands': ensemble_upper.tolist(),
            'ensemble_uncertainty': ensemble_uncertainty.tolist(),
            'methods_used': len(ensemble_results),
            'confidence_level': self.confidence_level
        }
    
    def _interpolate_trajectory(self, source_timestamps: List[str], 
                               source_embeddings: List[Embedding],
                               target_timestamps: List[str]) -> Optional[np.ndarray]:
        """Interpolate trajectory to target timestamps."""
        if len(source_timestamps) < 2:
            return None
        
        try:
            # Convert timestamps to numeric
            source_numeric = [self._timestamp_to_numeric(ts) for ts in source_timestamps]
            target_numeric = [self._timestamp_to_numeric(ts) for ts in target_timestamps]
            
            # Interpolate each dimension
            embedding_dim = len(source_embeddings[0])
            interpolated = np.zeros((len(target_timestamps), embedding_dim))
            
            for dim in range(embedding_dim):
                values = [emb[dim] for emb in source_embeddings]
                interpolated[:, dim] = np.interp(target_numeric, source_numeric, values)
            
            return interpolated
            
        except Exception as e:
            logger.debug(f"Interpolation failed: {e}")
            return None
    
    def _timestamp_to_numeric(self, timestamp: str) -> float:
        """Convert timestamp to numeric value."""
        try:
            from datetime import datetime
            
            if '-' in timestamp:
                if len(timestamp.split('-')) == 2:  # YYYY-MM
                    dt = datetime.strptime(timestamp, '%Y-%m')
                else:  # YYYY-MM-DD
                    dt = datetime.strptime(timestamp, '%Y-%m-%d')
                
                # Convert to days since epoch
                epoch = datetime(1970, 1, 1)
                return (dt - epoch).total_seconds() / (24 * 3600)
            else:
                return float(timestamp)
        except Exception:
            # Fall back to hash-based representation
            return hash(timestamp) % 100000
    
    def _smooth_confidence_estimates(self, confidence_array: np.ndarray,
                                    window_size: int = 3) -> np.ndarray:
        """Smooth confidence estimates using moving average."""
        if len(confidence_array) < window_size:
            return confidence_array
        
        smoothed = np.zeros_like(confidence_array)
        
        for i in range(len(confidence_array)):
            start = max(0, i - window_size // 2)
            end = min(len(confidence_array), i + window_size // 2 + 1)
            smoothed[i] = np.mean(confidence_array[start:end], axis=0)
        
        return smoothed
    
    def get_uncertainty_summary(self, word: Word) -> Optional[Dict[str, float]]:
        """Get uncertainty summary for a specific word."""
        if word not in self.confidence_bands:
            return None
        
        bands = self.confidence_bands[word]
        
        # Compute average uncertainty
        lower_bands = np.array(bands['lower_bands'])
        upper_bands = np.array(bands['upper_bands'])
        band_widths = upper_bands - lower_bands
        
        # Average uncertainty across time and dimensions
        mean_uncertainty = np.mean(band_widths)
        max_uncertainty = np.max(band_widths)
        min_uncertainty = np.min(band_widths)
        
        return {
            'mean_uncertainty': float(mean_uncertainty),
            'max_uncertainty': float(max_uncertainty),
            'min_uncertainty': float(min_uncertainty),
            'uncertainty_std': float(np.std(band_widths)),
            'confidence_level': bands['confidence_level']
        }
    
    def visualize_confidence_bands(self, word: Word, 
                                  embedding_store: TemporalEmbeddingStore) -> Optional[Dict[str, Any]]:
        """Create visualization data for confidence bands."""
        if word not in self.confidence_bands:
            return None
        
        bands = self.confidence_bands[word]
        
        # Get original trajectory
        timeline = embedding_store.get_word_timeline(word)
        original_embeddings = [emb for _, emb in timeline]
        
        # Project to 2D for visualization (using PCA)
        try:
            from sklearn.decomposition import PCA
            
            # Combine all trajectories for PCA
            all_trajectories = (
                original_embeddings + 
                bands['lower_bands'] + 
                bands['upper_bands']
            )
            
            pca = PCA(n_components=2)
            projected = pca.fit_transform(all_trajectories)
            
            n_points = len(original_embeddings)
            original_2d = projected[:n_points]
            lower_2d = projected[n_points:2*n_points]
            upper_2d = projected[2*n_points:3*n_points]
            
            return {
                'timestamps': bands['timestamps'],
                'original_trajectory': original_2d.tolist(),
                'mean_trajectory': pca.transform(bands['mean_trajectory']).tolist(),
                'lower_bounds': lower_2d.tolist(),
                'upper_bounds': upper_2d.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
            }
            
        except ImportError:
            # Fall back to first two dimensions
            return {
                'timestamps': bands['timestamps'],
                'original_trajectory': [[emb[0], emb[1]] for emb in original_embeddings],
                'mean_trajectory': [[emb[0], emb[1]] for emb in bands['mean_trajectory']],
                'lower_bounds': [[emb[0], emb[1]] for emb in bands['lower_bands']],
                'upper_bounds': [[emb[0], emb[1]] for emb in bands['upper_bands']]
            }

# Utility functions
def estimate_trajectory_uncertainty(embedding_store: TemporalEmbeddingStore,
                                   word: Word,
                                   method: str = 'bootstrap',
                                   confidence_level: float = 0.95) -> Optional[Dict[str, Any]]:
    """Convenience function to estimate uncertainty for a single word trajectory."""
    estimator = ConfidenceBandEstimator(method, confidence_level)
    results = estimator.process(embedding_store, [word])
    
    if word in results['confidence_bands']:
        return results['confidence_bands'][word]
    return None

def compare_uncertainty_methods(embedding_store: TemporalEmbeddingStore,
                               word: Word) -> Dict[str, Any]:
    """Compare uncertainty estimates from different methods."""
    methods = ['bootstrap', 'jackknife', 'neighborhood', 'gaussian_process']
    comparison_results = {}
    
    for method in methods:
        try:
            result = estimate_trajectory_uncertainty(
                embedding_store, word, method=method
            )
            if result is not None:
                comparison_results[method] = result
        except Exception as e:
            logger.debug(f"Method {method} failed for {word}: {e}")
    
    return comparison_results

def detect_high_uncertainty_periods(embedding_store: TemporalEmbeddingStore,
                                   words: List[Word],
                                   uncertainty_threshold: float = 0.5) -> Dict[str, List[str]]:
    """Detect time periods with high uncertainty across multiple words."""
    estimator = ConfidenceBandEstimator('ensemble', 0.95)
    results = estimator.process(embedding_store, words)
    
    high_uncertainty_periods = defaultdict(list)
    
    for word, bands in results['confidence_bands'].items():
        timestamps = bands['timestamps']
        lower_bands = np.array(bands['lower_bands'])
        upper_bands = np.array(bands['upper_bands'])
        uncertainties = np.mean(upper_bands - lower_bands, axis=1)
        
        for i, (timestamp, uncertainty) in enumerate(zip(timestamps, uncertainties)):
            if uncertainty > uncertainty_threshold:
                high_uncertainty_periods[timestamp].append(word)
    
    return dict(high_uncertainty_periods)