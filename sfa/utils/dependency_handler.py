"""Handle missing dependencies gracefully."""

import warnings
import importlib
from typing import Any, Optional, Callable


class DependencyError(ImportError):
    """Custom error for missing dependencies."""
    pass


def check_dependencies(dependencies: list) -> dict:
    """Check if required dependencies are installed."""
    missing = []
    available = {}
    
    for dep in dependencies:
        try:
            module = importlib.import_module(dep)
            available[dep] = module
        except ImportError:
            missing.append(dep)
    
    if missing:
        warnings.warn(
            f"Missing optional dependencies: {', '.join(missing)}. "
            f"Some features may not be available.",
            UserWarning
        )
    
    return available


def handle_import_error(module_name: str, feature_name: str) -> Callable:
    """Decorator to handle import errors gracefully."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                if module_name in str(e):
                    raise DependencyError(
                        f"Feature '{feature_name}' requires '{module_name}'. "
                        f"Please install it with: pip install {module_name}"
                    )
                raise
        return wrapper
    return decorator


# Create placeholder classes for missing dependencies
def create_placeholder_class(name: str, module: str) -> type:
    """Create a placeholder class for missing dependencies."""
    class PlaceholderClass:
        def __init__(self, *args, **kwargs):
            raise DependencyError(
                f"{name} requires '{module}' to be installed. "
                f"Please install it with: pip install {module}"
            )
    
    PlaceholderClass.__name__ = name
    PlaceholderClass.__qualname__ = name
    return PlaceholderClass


# Common dependency checks
CORE_DEPENDENCIES = [
    'numpy',
    'scipy',
    'scikit-learn',
    'pandas',
    'networkx'
]

VISUALIZATION_DEPENDENCIES = [
    'plotly',
    'matplotlib',
    'seaborn',
    'dash'
]

EMBEDDING_DEPENDENCIES = [
    'sentence_transformers',
    'transformers',
    'torch',
    'faiss'
]

OPTIONAL_DEPENDENCIES = [
    'streamlit',
    'umap-learn',
    'hdbscan',
    'leidenalg',
    'numba'
]


def check_core_dependencies():
    """Check if core dependencies are available."""
    available = check_dependencies(CORE_DEPENDENCIES)
    missing = [dep for dep in CORE_DEPENDENCIES if dep not in available]
    
    if missing:
        raise DependencyError(
            f"Core dependencies missing: {', '.join(missing)}. "
            f"Please install them with: pip install -r requirements.txt"
        )
    
    return available


def safe_import(module_name: str, feature: Optional[str] = None) -> Any:
    """Safely import a module with informative error messages."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if feature:
            raise DependencyError(
                f"Feature '{feature}' requires '{module_name}'. "
                f"Please install the required dependencies."
            )
        else:
            raise DependencyError(
                f"Module '{module_name}' is not installed. "
                f"Please install the required dependencies."
            )