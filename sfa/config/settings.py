"""Core configuration settings for Semantic Flow Analyzer."""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/semantic_flow_db')
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    dimension: int = int(os.getenv('EMBEDDING_DIMENSION', '384'))
    cache_size: int = int(os.getenv('EMBEDDING_CACHE_SIZE', '10000'))
    device: str = 'cuda' if os.getenv('GPU_ENABLED', 'false').lower() == 'true' else 'cpu'
    batch_size: int = int(os.getenv('BATCH_SIZE', '1000'))

@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    backend: str = os.getenv('PLOT_BACKEND', 'plotly')
    dashboard_host: str = os.getenv('DASHBOARD_HOST', '127.0.0.1')
    dashboard_port: int = int(os.getenv('DASHBOARD_PORT', '8050'))
    dashboard_debug: bool = os.getenv('DASHBOARD_DEBUG', 'false').lower() == 'true'
    max_points: int = 10000
    animation_fps: int = 30
    color_scheme: str = 'viridis'

@dataclass
class PerformanceConfig:
    """Performance and resource configuration."""
    n_jobs: int = int(os.getenv('N_JOBS', '4'))
    batch_size: int = int(os.getenv('BATCH_SIZE', '1000'))
    memory_limit: str = os.getenv('MEMORY_LIMIT', '8GB')
    gpu_enabled: bool = os.getenv('GPU_ENABLED', 'false').lower() == 'true'
    parallel_backend: str = 'threading'
    chunk_size: int = 1000

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    file: str = os.getenv('LOG_FILE', 'logs/sfa.log')
    format: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    max_size: str = '10MB'
    backup_count: int = 5
    console_output: bool = True

@dataclass
class ExportConfig:
    """Export configuration."""
    format: str = os.getenv('EXPORT_FORMAT', 'parquet')
    compression: str = os.getenv('EXPORT_COMPRESSION', 'snappy')
    directory: str = os.getenv('EXPORT_DIR', 'exports/')
    include_metadata: bool = True
    timestamp_format: str = '%Y%m%d_%H%M%S'

@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    ttl: int = int(os.getenv('CACHE_TTL', '3600'))
    max_size: int = int(os.getenv('CACHE_MAX_SIZE', '1000000'))
    backend: str = 'redis'
    key_prefix: str = 'sfa:'

@dataclass
class FeatureFlags:
    """Feature flags for experimental features."""
    experimental_features: bool = os.getenv('ENABLE_EXPERIMENTAL_FEATURES', 'false').lower() == 'true'
    metrics_collection: bool = os.getenv('ENABLE_METRICS_COLLECTION', 'true').lower() == 'true'
    performance_monitoring: bool = os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true'
    debug_mode: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

@dataclass
class Settings:
    """Main settings class combining all configuration."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
    # Global settings
    project_name: str = "Semantic Flow Analyzer"
    version: str = "0.1.0"
    data_dir: str = "data/"
    config_dir: str = "config/"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Settings':
        """Load settings from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create settings instance
        settings = cls()
        
        # Update with YAML data
        for section, values in config_data.items():
            if hasattr(settings, section):
                section_obj = getattr(settings, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        return settings
    
    def to_yaml(self, config_path: str) -> None:
        """Save settings to YAML file."""
        config_data = {}
        
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name)
            if hasattr(field_value, '__dataclass_fields__'):
                # It's a dataclass, extract its fields
                config_data[field_name] = {
                    sub_field: getattr(field_value, sub_field)
                    for sub_field in field_value.__dataclass_fields__
                }
            else:
                config_data[field_name] = field_value
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def validate(self) -> List[str]:
        """Validate configuration settings."""
        errors = []
        
        # Validate directories exist
        for directory in [self.data_dir, self.config_dir]:
            if not os.path.exists(directory):
                errors.append(f"Directory does not exist: {directory}")
        
        # Validate database URL
        if not self.database.url.startswith(('postgresql://', 'sqlite:///')):
            errors.append("Invalid database URL format")
        
        # Validate embedding dimension
        if self.embedding.dimension <= 0:
            errors.append("Embedding dimension must be positive")
        
        # Validate performance settings
        if self.performance.n_jobs <= 0:
            errors.append("Number of jobs must be positive")
        
        return errors

# Global settings instance
settings = Settings()