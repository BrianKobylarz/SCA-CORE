"""Core flow orchestration for Semantic Flow Analyzer."""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
import logging
from datetime import datetime

from .types import (
    Word, Timestamp, SemanticFlow, FlowEvent, FlowEventType,
    WordTrajectory, FlowMatrix, EventTimeline
)
from .embeddings import EmbeddingStore
from .base import BaseAnalyzer

# Avoid circular import by importing FlowConfig when needed
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config.flow_config import FlowConfig

logger = logging.getLogger(__name__)

class FlowManager(BaseAnalyzer):
    """Core orchestrator for semantic flow analysis."""
    
    def __init__(self, embeddings_store: EmbeddingStore, config: Any):
        super().__init__("FlowManager")
        self.embeddings = embeddings_store
        self.config = config
        
        # Core components (will be initialized by other modules)
        self.flow_tracker = None
        self.burst_detector = None
        self.cascade_analyzer = None
        
        # Results storage
        self.flows: FlowMatrix = {}
        self.events: EventTimeline = {}
        self.trajectories: Dict[Word, WordTrajectory] = {}
        
        # Analysis state
        self.analysis_timestamp = None
        self.last_update = None
        
        logger.info("FlowManager initialized")
    
    def register_flow_tracker(self, flow_tracker) -> None:
        """Register flow tracker component."""
        self.flow_tracker = flow_tracker
        logger.info("Flow tracker registered")
    
    def register_burst_detector(self, burst_detector) -> None:
        """Register burst detector component."""
        self.burst_detector = burst_detector
        logger.info("Burst detector registered")
    
    def register_cascade_analyzer(self, cascade_analyzer) -> None:
        """Register cascade analyzer component."""
        self.cascade_analyzer = cascade_analyzer
        logger.info("Cascade analyzer registered")
    
    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Implement BaseAnalyzer interface with flexible parameters."""
        # Extract parameters from args and kwargs
        target_words = kwargs.get('target_words', None)
        time_range = kwargs.get('time_range', None)
        
        # Handle positional arguments if provided
        if args:
            if len(args) >= 1:
                target_words = args[0]
            if len(args) >= 2:
                time_range = args[1]
        
        return self._analyze_internal(target_words, time_range)
    
    def _analyze_internal(self, target_words: Optional[List[Word]] = None,
                         time_range: Optional[Tuple[Timestamp, Timestamp]] = None) -> Dict[str, Any]:
        """Perform comprehensive flow analysis."""
        logger.info("Starting comprehensive flow analysis")
        
        # Determine analysis scope
        if target_words is None:
            # Use top frequent words across all timestamps
            target_words = self._get_top_words(n=100)
        
        if time_range is None:
            timestamps = self.embeddings.get_timestamps()
            time_range = (timestamps[0], timestamps[-1])
        else:
            timestamps = [t for t in self.embeddings.get_timestamps() 
                         if time_range[0] <= t <= time_range[1]]
        
        # Track flows for target words
        self._track_flows(target_words)
        
        # Detect events
        self._detect_events()
        
        # Compute trajectories
        self._compute_trajectories(target_words)
        
        # Generate summary
        results = self._generate_analysis_summary(target_words, time_range)
        
        self.analysis_timestamp = datetime.now().isoformat()
        self.last_update = datetime.now()
        
        logger.info("Flow analysis completed")
        return results
    
    def get_word_flows(self, word: Word) -> Dict[Timestamp, SemanticFlow]:
        """Get flows for a specific word."""
        return {t: flow for t, flows in self.flows.items() 
                if word in flows for flow in [flows[word]] if flow is not None}
    
    def get_events(self, event_type: Optional[FlowEventType] = None,
                   time_range: Optional[Tuple[Timestamp, Timestamp]] = None) -> List[FlowEvent]:
        """Get events filtered by type and time range."""
        events = []
        
        for timestamp, event_list in self.events.items():
            # Check time range
            if time_range and not (time_range[0] <= timestamp <= time_range[1]):
                continue
            
            for event in event_list:
                # Check event type
                if event_type and event.event_type != event_type:
                    continue
                
                events.append(event)
        
        return events
    
    def get_trajectory(self, word: Word) -> Optional[WordTrajectory]:
        """Get trajectory for a specific word."""
        return self.trajectories.get(word)
    
    def compute_flow_statistics(self, timestamp: Timestamp) -> Dict[str, float]:
        """Compute flow statistics for a timestamp."""
        if timestamp not in self.flows:
            return {}
        
        flows = self.flows[timestamp]
        magnitudes = [flow.total_magnitude for flow in flows.values() if flow is not None]
        coherences = [flow.coherence for flow in flows.values() if flow is not None]
        
        if not magnitudes:
            return {}
        
        return {
            'total_flow': sum(magnitudes),
            'mean_flow': np.mean(magnitudes),
            'std_flow': np.std(magnitudes),
            'median_flow': np.median(magnitudes),
            'max_flow': max(magnitudes),
            'mean_coherence': np.mean(coherences) if coherences else 0.0,
            'flow_entropy': self._compute_flow_entropy(flows),
            'active_words': len([f for f in flows.values() if f is not None])
        }
    
    def detect_flow_anomalies(self, window_size: int = 5) -> List[Dict[str, Any]]:
        """Detect anomalous flow patterns."""
        anomalies = []
        timestamps = sorted(self.flows.keys())
        
        if len(timestamps) < window_size:
            return anomalies
        
        # Compute statistics over sliding window
        for i in range(window_size, len(timestamps)):
            current_timestamp = timestamps[i]
            window_timestamps = timestamps[i-window_size:i]
            
            # Get baseline statistics
            baseline_stats = []
            for t in window_timestamps:
                stats = self.compute_flow_statistics(t)
                if stats:
                    baseline_stats.append(stats)
            
            if not baseline_stats:
                continue
            
            current_stats = self.compute_flow_statistics(current_timestamp)
            if not current_stats:
                continue
            
            # Check for anomalies
            for metric in ['total_flow', 'mean_flow', 'mean_coherence']:
                baseline_values = [s[metric] for s in baseline_stats if metric in s]
                if not baseline_values:
                    continue
                
                baseline_mean = np.mean(baseline_values)
                baseline_std = np.std(baseline_values)
                
                if baseline_std > 0:
                    z_score = (current_stats[metric] - baseline_mean) / baseline_std
                    
                    if abs(z_score) > 3.0:  # 3-sigma threshold
                        anomalies.append({
                            'timestamp': current_timestamp,
                            'metric': metric,
                            'value': current_stats[metric],
                            'baseline_mean': baseline_mean,
                            'baseline_std': baseline_std,
                            'z_score': z_score,
                            'anomaly_type': 'high' if z_score > 0 else 'low'
                        })
        
        return anomalies
    
    def _track_flows(self, target_words: List[Word]) -> None:
        """Track flows for target words."""
        if not self.flow_tracker:
            logger.warning("Flow tracker not registered, skipping flow tracking")
            return
        
        logger.info(f"Tracking flows for {len(target_words)} words")
        
        for word in target_words:
            try:
                word_flows = self.flow_tracker.track_word_flow(word)
                
                for timestamp, flow in word_flows.items():
                    if timestamp not in self.flows:
                        self.flows[timestamp] = {}
                    self.flows[timestamp][word] = flow
                    
            except Exception as e:
                logger.error(f"Error tracking flows for word {word}: {e}")
    
    def _detect_events(self) -> None:
        """Detect flow events."""
        events = []
        
        # Burst detection
        if self.burst_detector:
            try:
                burst_events = self.burst_detector.detect_bursts()
                for timestamp, burst_list in burst_events.items():
                    events.extend(burst_list)
                logger.info(f"Detected {len(events)} burst events")
            except Exception as e:
                logger.error(f"Error in burst detection: {e}")
        
        # Cascade detection
        if self.cascade_analyzer:
            try:
                timestamps = self.embeddings.get_timestamps()
                for timestamp in timestamps:
                    cascade_analysis = self.cascade_analyzer.analyze_cascade_potential(timestamp)
                    
                    # Convert to events if risk is high
                    if cascade_analysis['cascade_risk'] > self.config.cascade_risk_threshold:
                        cascade_event = FlowEvent(
                            timestamp=timestamp,
                            event_type=FlowEventType.CASCADE,
                            primary_words=[w for w, _ in cascade_analysis['superspreaders'][:5]],
                            magnitude=cascade_analysis['cascade_risk'],
                            affected_radius=len(cascade_analysis['superspreaders']),
                            metadata=cascade_analysis
                        )
                        events.append(cascade_event)
                
                logger.info(f"Detected {len([e for e in events if e.event_type == FlowEventType.CASCADE])} cascade events")
            except Exception as e:
                logger.error(f"Error in cascade detection: {e}")
        
        # Group events by timestamp
        self.events = defaultdict(list)
        for event in events:
            self.events[event.timestamp].append(event)
    
    def _compute_trajectories(self, target_words: List[Word]) -> None:
        """Compute trajectories for target words."""
        if not self.flow_tracker:
            logger.warning("Flow tracker not registered, skipping trajectory computation")
            return
        
        logger.info(f"Computing trajectories for {len(target_words)} words")
        
        for word in target_words:
            try:
                trajectory = self.flow_tracker.compute_word_trajectory(word)
                if trajectory:
                    self.trajectories[word] = trajectory
            except Exception as e:
                logger.error(f"Error computing trajectory for word {word}: {e}")
    
    def _generate_analysis_summary(self, target_words: List[Word],
                                  time_range: Tuple[Timestamp, Timestamp]) -> Dict[str, Any]:
        """Generate analysis summary."""
        timestamps = [t for t in self.embeddings.get_timestamps() 
                     if time_range[0] <= t <= time_range[1]]
        
        # Flow statistics
        flow_stats = {}
        for timestamp in timestamps:
            flow_stats[timestamp] = self.compute_flow_statistics(timestamp)
        
        # Event summary
        event_summary = defaultdict(int)
        for event_list in self.events.values():
            for event in event_list:
                event_summary[event.event_type.value] += 1
        
        # Trajectory summary
        trajectory_summary = {
            'high_volatility': [],
            'high_drift': [],
            'stable': []
        }
        
        for word, trajectory in self.trajectories.items():
            if trajectory.stochasticity > 0.7:
                trajectory_summary['high_volatility'].append(word)
            elif trajectory.path_dependency > 0.7:
                trajectory_summary['high_drift'].append(word)
            else:
                trajectory_summary['stable'].append(word)
        
        # Top movers
        top_movers = self._get_top_movers(target_words, n=10)
        
        return {
            'analysis_info': {
                'target_words': len(target_words),
                'time_range': time_range,
                'timestamps_analyzed': len(timestamps),
                'analysis_timestamp': self.analysis_timestamp
            },
            'flow_statistics': flow_stats,
            'events': {
                'total_events': sum(event_summary.values()),
                'by_type': dict(event_summary),
                'event_density': sum(event_summary.values()) / len(timestamps) if timestamps else 0
            },
            'trajectories': {
                'total_trajectories': len(self.trajectories),
                'by_category': {k: len(v) for k, v in trajectory_summary.items()}
            },
            'top_movers': top_movers,
            'anomalies': self.detect_flow_anomalies()
        }
    
    def _get_top_words(self, n: int = 100) -> List[Word]:
        """Get top frequent words across all timestamps."""
        word_counts = defaultdict(int)
        
        for timestamp in self.embeddings.get_timestamps():
            vocab = self.embeddings.get_vocabulary(timestamp)
            for word in vocab:
                word_counts[word] += 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:n]]
    
    def _get_top_movers(self, words: List[Word], n: int = 10) -> List[Dict[str, Any]]:
        """Get words with highest movement."""
        movers = []
        
        for word in words:
            trajectory = self.trajectories.get(word)
            if trajectory:
                movers.append({
                    'word': word,
                    'cumulative_distance': trajectory.cumulative_distance,
                    'stochasticity': trajectory.stochasticity,
                    'path_dependency': trajectory.path_dependency,
                    'total_velocity': sum(trajectory.velocities)
                })
        
        # Sort by cumulative distance
        movers.sort(key=lambda x: x['cumulative_distance'], reverse=True)
        return movers[:n]
    
    def _compute_flow_entropy(self, flows: Dict[Word, SemanticFlow]) -> float:
        """Compute entropy of flow magnitudes."""
        magnitudes = [flow.total_magnitude for flow in flows.values() if flow is not None]
        
        if not magnitudes:
            return 0.0
        
        # Normalize to probabilities
        total = sum(magnitudes)
        if total == 0:
            return 0.0
        
        probabilities = [m / total for m in magnitudes]
        
        # Compute Shannon entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def export_results(self, path: str, format: str = 'json') -> None:
        """Export analysis results."""
        import json
        from pathlib import Path
        
        results = {
            'flows': {t: {w: self._serialize_flow(f) for w, f in flows.items()} 
                     for t, flows in self.flows.items()},
            'events': {t: [self._serialize_event(e) for e in events] 
                      for t, events in self.events.items()},
            'trajectories': {w: self._serialize_trajectory(t) 
                           for w, t in self.trajectories.items()},
            'analysis_timestamp': self.analysis_timestamp
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        logger.info(f"Results exported to {path}")
    
    def _serialize_flow(self, flow: SemanticFlow) -> Dict[str, Any]:
        """Serialize flow for export."""
        return {
            'source_word': flow.source_word,
            'target_words': flow.target_words,
            'flow_vectors': flow.flow_vectors,
            'total_magnitude': flow.total_magnitude,
            'coherence': flow.coherence
        }
    
    def _serialize_event(self, event: FlowEvent) -> Dict[str, Any]:
        """Serialize event for export."""
        return {
            'timestamp': event.timestamp,
            'event_type': event.event_type.value,
            'primary_words': event.primary_words,
            'magnitude': event.magnitude,
            'affected_radius': event.affected_radius,
            'metadata': event.metadata
        }
    
    def _serialize_trajectory(self, trajectory: WordTrajectory) -> Dict[str, Any]:
        """Serialize trajectory for export."""
        return {
            'word': trajectory.word,
            'positions': [(t, emb.tolist()) for t, emb in trajectory.positions],
            'velocities': trajectory.velocities,
            'accelerations': trajectory.accelerations,
            'cumulative_distance': trajectory.cumulative_distance,
            'stochasticity': trajectory.stochasticity,
            'path_dependency': trajectory.path_dependency
        }