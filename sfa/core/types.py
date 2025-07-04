"""Type definitions for Semantic Flow Analyzer."""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Type aliases for clarity
Embedding = np.ndarray
Timestamp = str
Word = str
Similarity = float
FlowVector = Tuple[float, float, float]  # (x, y, magnitude)

class FlowEventType(Enum):
    """Types of flow events"""
    BURST = "burst"
    CASCADE = "cascade"
    CONVERGENCE = "convergence"
    DIVERGENCE = "divergence"
    PHASE_TRANSITION = "phase_transition"
    COMMUNITY_BIRTH = "community_birth"
    COMMUNITY_DEATH = "community_death"
    BRIDGE_FORMATION = "bridge_formation"
    BRIDGE_COLLAPSE = "bridge_collapse"

@dataclass
class FlowEvent:
    """Represents a semantic flow event"""
    timestamp: Timestamp
    event_type: FlowEventType
    primary_words: List[Word]
    magnitude: float
    affected_radius: int
    metadata: Dict[str, Any]
    
@dataclass
class SemanticFlow:
    """Tracks flow between timestamps"""
    source_word: Word
    target_words: List[Word]
    flow_vectors: List[FlowVector]
    total_magnitude: float
    coherence: float  # How directed vs dispersed
    timestamp: Optional[Timestamp] = None
    
    @property
    def magnitude(self) -> float:
        """Alias for total_magnitude for backward compatibility"""
        return self.total_magnitude
    
    @property
    def target_word(self) -> Optional[Word]:
        """Return first target word for backward compatibility"""
        return self.target_words[0] if self.target_words else None
    
@dataclass
class WordTrajectory:
    """Complete trajectory of a word through time"""
    word: Word
    positions: List[Tuple[Timestamp, Embedding]]
    velocities: List[float]
    accelerations: List[float]
    cumulative_distance: float
    stochasticity: float
    path_dependency: float
    
@dataclass
class CommunityLineage:
    """Tracks community evolution"""
    lineage_id: str
    birth_time: Timestamp
    death_time: Optional[Timestamp]
    peak_size: int
    total_persistence: int
    splits: List[Timestamp]
    merges: List[Timestamp]
    core_vocabulary: Set[Word]

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    timestamp: Timestamp
    word: Optional[Word]
    metric_name: str
    value: Any
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class FlowField:
    """Represents a flow field in semantic space"""
    timestamp: Timestamp
    flow_vectors: Dict[Word, FlowVector]
    divergence: Dict[Word, float]
    curl: Dict[Word, float]
    potential: Dict[Word, float]

@dataclass
class SemanticNetwork:
    """Represents semantic network at a time point"""
    timestamp: Timestamp
    nodes: Set[Word]
    edges: Dict[Tuple[Word, Word], float]
    communities: List[Set[Word]]
    centrality: Dict[Word, float]
    clustering: float
    modularity: float

@dataclass
class CascadeEvent:
    """Represents a cascade event"""
    timestamp: Timestamp
    seed_words: List[Word]
    affected_words: List[Word]
    cascade_risk: float
    propagation_tree: Dict[Word, List[Word]]
    duration: int
    peak_magnitude: float

@dataclass
class PhaseTransition:
    """Represents a phase transition"""
    timestamp: Timestamp
    transition_type: str
    order_parameter_before: float
    order_parameter_after: float
    criticality_indicators: Dict[str, float]
    affected_communities: List[Set[Word]]

@dataclass
class BurstProfile:
    """Profile of a burst event"""
    word: Word
    burst_time: Timestamp
    magnitude: float
    duration: int
    z_score: float
    baseline: float
    affected_neighbors: List[Word]

@dataclass
class FlowStatistics:
    """Statistical summary of flows"""
    timestamp: Timestamp
    total_flow: float
    mean_flow: float
    std_flow: float
    flow_entropy: float
    dominant_direction: FlowVector
    coherence_global: float

@dataclass
class AnalogiesResult:
    """Result from scientific analogies"""
    timestamp: Timestamp
    analogy_name: str
    prediction: float
    confidence: float
    parameters: Dict[str, float]
    state_variables: Dict[str, float]

@dataclass
class InterventionScenario:
    """Intervention scenario for semantic systems"""
    name: str
    description: str
    target_words: List[Word]
    intervention_type: str
    expected_impact: float
    confidence: float
    side_effects: List[str]

@dataclass
class ComparisonResult:
    """Result of comparing two semantic states"""
    timestamp1: Timestamp
    timestamp2: Timestamp
    distance: float
    similarity: float
    changed_words: List[Word]
    new_words: List[Word]
    disappeared_words: List[Word]
    flow_correlation: float

@dataclass
class EvolutionSummary:
    """Summary of semantic evolution"""
    word: Word
    time_span: Tuple[Timestamp, Timestamp]
    total_distance: float
    net_displacement: float
    volatility: float
    trend_direction: str
    major_events: List[FlowEvent]
    final_state: str

# Utility type aliases
EmbeddingMatrix = Dict[Timestamp, Dict[Word, Embedding]]
SimilarityMatrix = Dict[Tuple[Word, Word], float]
FlowMatrix = Dict[Timestamp, Dict[Word, SemanticFlow]]
EventTimeline = Dict[Timestamp, List[FlowEvent]]
CommunityDynamics = Dict[Timestamp, List[Set[Word]]]
MetricTimeseries = Dict[Timestamp, Dict[str, float]]

# Configuration types
@dataclass
class AnalysisConfig:
    """Configuration for analysis"""
    target_words: List[Word]
    time_range: Tuple[Timestamp, Timestamp]
    metrics: List[str]
    analogies: List[str]
    visualization: bool
    export_results: bool

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    plot_type: str
    dimensions: int
    color_scheme: str
    animation: bool
    interactive: bool
    export_format: str

@dataclass
class ExportConfig:
    """Configuration for data export"""
    format: str
    compression: str
    include_metadata: bool
    include_embeddings: bool
    include_visualizations: bool