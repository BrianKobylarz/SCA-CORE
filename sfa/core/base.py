"""Abstract base classes for Semantic Flow Analyzer."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    import warnings
    warnings.warn("NumPy is not installed. Core functionality will be limited.", ImportWarning)
    np = None

from .types import Word, Timestamp, Embedding

# BaseMetric is now imported from metrics module to avoid duplication
# from ..metrics.base_metric import BaseMetric

class BaseAnalogy(ABC):
    """Abstract base class for scientific analogies."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.parameters = {}
        self.state = {}
    
    @abstractmethod
    def initialize(self, embeddings: Dict[Timestamp, Dict[Word, Embedding]]) -> None:
        """Initialize the analogy with embedding data."""
        pass
    
    @abstractmethod
    def step(self, current_time: Timestamp, next_time: Timestamp) -> Dict[str, Any]:
        """Execute one step of the analogy."""
        pass
    
    @abstractmethod
    def predict(self, word: Word, timestamp: Timestamp, 
                horizon: int = 1) -> List[Tuple[Timestamp, float]]:
        """Predict future behavior based on the analogy."""
        pass
    
    def set_parameters(self, **params) -> None:
        """Set analogy parameters."""
        self.parameters.update(params)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current analogy state."""
        return self.state.copy()
    
    def reset(self) -> None:
        """Reset analogy state."""
        self.state = {}

class BaseVisualizer(ABC):
    """Abstract base class for visualizers."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.config = {}
    
    @abstractmethod
    def create_figure(self, *args, **kwargs) -> Any:
        """Create the main visualization figure."""
        pass
    
    @abstractmethod
    def update_figure(self, figure: Any, *args, **kwargs) -> Any:
        """Update existing figure with new data."""
        pass
    
    def configure(self, **config) -> None:
        """Configure visualizer settings."""
        self.config.update(config)
    
    def export(self, figure: Any, path: str, format: str = 'html') -> None:
        """Export figure to file."""
        if hasattr(figure, 'write_html') and format == 'html':
            figure.write_html(path)
        elif hasattr(figure, 'write_image'):
            figure.write_image(path)
        else:
            raise NotImplementedError(f"Export format {format} not supported")

class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.config = {}
    
    @abstractmethod
    def load(self, *args, **kwargs) -> Dict[Timestamp, Dict[Word, Embedding]]:
        """Load embedding data."""
        pass
    
    @abstractmethod
    def validate(self, data: Dict[Timestamp, Dict[Word, Embedding]]) -> List[str]:
        """Validate loaded data."""
        pass
    
    def configure(self, **config) -> None:
        """Configure data loader."""
        self.config.update(config)

class BaseExporter(ABC):
    """Abstract base class for data exporters."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.config = {}
    
    @abstractmethod
    def export(self, data: Any, path: str, **kwargs) -> None:
        """Export data to file."""
        pass
    
    def configure(self, **config) -> None:
        """Configure exporter."""
        self.config.update(config)

class BaseCache(ABC):
    """Abstract base class for caching systems."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

class BaseAnalyzer(ABC):
    """Abstract base class for analyzers."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.config = {}
        self.metrics = {}
        self.results = {}
    
    @abstractmethod
    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Perform analysis and return results."""
        pass
    
    def add_metric(self, name: str, metric: Any) -> None:
        """Add a metric to the analyzer."""
        self.metrics[name] = metric
    
    def remove_metric(self, name: str) -> None:
        """Remove a metric from the analyzer."""
        if name in self.metrics:
            del self.metrics[name]
    
    def configure(self, **config) -> None:
        """Configure analyzer."""
        self.config.update(config)
    
    def get_results(self) -> Dict[str, Any]:
        """Get analysis results."""
        return self.results.copy()
    
    def reset(self) -> None:
        """Reset analyzer state."""
        self.results = {}
        for metric in self.metrics.values():
            if hasattr(metric, 'reset_cache'):
                metric.reset_cache()

class BaseProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.config = {}
    
    @abstractmethod
    def process(self, data: Any, *args, **kwargs) -> Any:
        """Process input data."""
        pass
    
    def configure(self, **config) -> None:
        """Configure processor."""
        self.config.update(config)

class BaseTransformer(ABC):
    """Abstract base class for data transformers."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.fitted = False
    
    @abstractmethod
    def fit(self, data: Any, *args, **kwargs) -> 'BaseTransformer':
        """Fit transformer to data."""
        pass
    
    @abstractmethod
    def transform(self, data: Any, *args, **kwargs) -> Any:
        """Transform data."""
        pass
    
    def fit_transform(self, data: Any, *args, **kwargs) -> Any:
        """Fit transformer and transform data."""
        return self.fit(data, *args, **kwargs).transform(data, *args, **kwargs)
    
    def is_fitted(self) -> bool:
        """Check if transformer is fitted."""
        return self.fitted