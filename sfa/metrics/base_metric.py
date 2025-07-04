"""Base metric class for semantic flow analysis."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class BaseMetric(ABC):
    """Abstract base class for all metrics in semantic flow analysis."""
    
    def __init__(self, name: str = None, cache_enabled: bool = True):
        self.name = name or self.__class__.__name__
        self.cache_enabled = cache_enabled
        self.cache = {} if cache_enabled else None
        self.computation_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'last_computation_time': 0.0
        }
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> Union[float, np.ndarray, Dict]:
        """Compute the metric value.
        
        This method must be implemented by all concrete metric classes.
        
        Returns:
            The computed metric value. Type depends on the specific metric.
        """
        pass
    
    def compute_with_cache(self, *args, **kwargs) -> Union[float, np.ndarray, Dict]:
        """Compute metric with caching support."""
        self.computation_stats['total_calls'] += 1
        
        if self.cache_enabled:
            cache_key = self._generate_cache_key(*args, **kwargs)
            
            if cache_key in self.cache:
                self.computation_stats['cache_hits'] += 1
                return self.cache[cache_key]
        
        # Compute the metric
        start_time = time.time()
        try:
            result = self.compute(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error computing {self.name}: {e}")
            raise
        
        computation_time = time.time() - start_time
        self.computation_stats['total_time'] += computation_time
        self.computation_stats['last_computation_time'] = computation_time
        
        # Cache the result
        if self.cache_enabled:
            self.cache[cache_key] = result
        
        return result
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key for the given arguments."""
        # Convert arguments to a hashable representation
        arg_strs = []
        
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg_strs.append(f"array_{hash(arg.tobytes())}")
            elif hasattr(arg, '__iter__') and not isinstance(arg, str):
                arg_strs.append(f"iter_{hash(tuple(arg))}")
            else:
                arg_strs.append(str(arg))
        
        for key, value in sorted(kwargs.items()):
            if isinstance(value, np.ndarray):
                arg_strs.append(f"{key}=array_{hash(value.tobytes())}")
            elif hasattr(value, '__iter__') and not isinstance(value, str):
                arg_strs.append(f"{key}=iter_{hash(tuple(value))}")
            else:
                arg_strs.append(f"{key}={value}")
        
        return f"{self.name}:{'|'.join(arg_strs)}"
    
    def clear_cache(self) -> None:
        """Clear the metric cache."""
        if self.cache:
            self.cache.clear()
        self.computation_stats['cache_hits'] = 0
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self.cache) if self.cache else 0
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        stats = self.computation_stats.copy()
        stats['average_time'] = (
            stats['total_time'] / stats['total_calls'] 
            if stats['total_calls'] > 0 else 0.0
        )
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / stats['total_calls']
            if stats['total_calls'] > 0 else 0.0
        )
        return stats
    
    def validate_inputs(self, *args, **kwargs) -> None:
        """Validate input arguments.
        
        Override this method to add input validation for specific metrics.
        
        Raises:
            ValueError: If inputs are invalid.
        """
        pass
    
    def preprocess_inputs(self, *args, **kwargs) -> tuple:
        """Preprocess input arguments.
        
        Override this method to add input preprocessing for specific metrics.
        
        Returns:
            Tuple of (processed_args, processed_kwargs)
        """
        return args, kwargs
    
    def postprocess_result(self, result: Any) -> Any:
        """Postprocess the computed result.
        
        Override this method to add result postprocessing for specific metrics.
        
        Args:
            result: Raw computation result
            
        Returns:
            Processed result
        """
        return result
    
    def compute_safe(self, *args, **kwargs) -> Union[float, np.ndarray, Dict]:
        """Safely compute metric with validation and preprocessing."""
        try:
            # Validate inputs
            self.validate_inputs(*args, **kwargs)
            
            # Preprocess inputs
            processed_args, processed_kwargs = self.preprocess_inputs(*args, **kwargs)
            
            # Compute with cache
            result = self.compute_with_cache(*processed_args, **processed_kwargs)
            
            # Postprocess result
            final_result = self.postprocess_result(result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in safe computation of {self.name}: {e}")
            raise
    
    def batch_compute(self, input_list: List[tuple], **common_kwargs) -> List[Any]:
        """Compute metric for a batch of inputs.
        
        Args:
            input_list: List of argument tuples
            **common_kwargs: Common keyword arguments for all computations
            
        Returns:
            List of computed results
        """
        results = []
        
        for args in input_list:
            try:
                result = self.compute_safe(*args, **common_kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to compute {self.name} for args {args}: {e}")
                results.append(None)
        
        return results
    
    def reset_stats(self) -> None:
        """Reset computation statistics."""
        self.computation_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'last_computation_time': 0.0
        }
    
    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name}(cache_enabled={self.cache_enabled})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the metric."""
        stats = self.get_computation_stats()
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"cache_enabled={self.cache_enabled}, "
                f"total_calls={stats['total_calls']}, "
                f"cache_hit_rate={stats['cache_hit_rate']:.2f})")
    
    @classmethod
    def create_ensemble(cls, metrics: List['BaseMetric'], 
                       weights: Optional[List[float]] = None) -> 'EnsembleMetric':
        """Create an ensemble of metrics.
        
        Args:
            metrics: List of metrics to ensemble
            weights: Optional weights for each metric
            
        Returns:
            EnsembleMetric instance
        """
        return EnsembleMetric(metrics, weights)

class EnsembleMetric(BaseMetric):
    """Ensemble of multiple metrics with weighted averaging."""
    
    def __init__(self, metrics: List[BaseMetric], 
                 weights: Optional[List[float]] = None,
                 aggregation_method: str = 'weighted_mean'):
        super().__init__(f"Ensemble({len(metrics)}metrics)")
        self.metrics = metrics
        self.weights = weights or [1.0] * len(metrics)
        self.aggregation_method = aggregation_method
        
        if len(self.weights) != len(self.metrics):
            raise ValueError("Number of weights must match number of metrics")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def compute(self, *args, **kwargs) -> float:
        """Compute ensemble metric."""
        results = []
        
        for metric in self.metrics:
            try:
                result = metric.compute_safe(*args, **kwargs)
                # Convert result to scalar if needed
                if isinstance(result, np.ndarray):
                    result = float(np.mean(result))
                elif isinstance(result, dict):
                    # Use first numeric value
                    result = next((v for v in result.values() 
                                 if isinstance(v, (int, float))), 0.0)
                results.append(float(result))
            except Exception as e:
                logger.warning(f"Failed to compute {metric.name} in ensemble: {e}")
                results.append(0.0)
        
        if not results:
            return 0.0
        
        # Aggregate results
        if self.aggregation_method == 'weighted_mean':
            return sum(r * w for r, w in zip(results, self.weights))
        elif self.aggregation_method == 'mean':
            return np.mean(results)
        elif self.aggregation_method == 'median':
            return np.median(results)
        elif self.aggregation_method == 'max':
            return max(results)
        elif self.aggregation_method == 'min':
            return min(results)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def get_component_results(self, *args, **kwargs) -> Dict[str, float]:
        """Get results from individual component metrics."""
        results = {}
        
        for metric in self.metrics:
            try:
                result = metric.compute_safe(*args, **kwargs)
                if isinstance(result, np.ndarray):
                    result = float(np.mean(result))
                elif isinstance(result, dict):
                    result = next((v for v in result.values() 
                                 if isinstance(v, (int, float))), 0.0)
                results[metric.name] = float(result)
            except Exception as e:
                logger.warning(f"Failed to compute {metric.name}: {e}")
                results[metric.name] = 0.0
        
        return results