"""Event correlation analysis for semantic flow systems."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from ..core.types import Word, Timestamp, SemanticFlow, FlowEvent
from ..core.base import BaseAnalyzer

@dataclass
class CorrelatedEvent:
    """Represents a correlation between events"""
    event1: FlowEvent
    event2: FlowEvent
    correlation_type: str  # 'causal', 'temporal', 'semantic', 'cascade'
    correlation_strength: float  # 0-1
    time_lag: float  # Time delay between events
    confidence: float  # Confidence in correlation
    mechanism: str  # Proposed mechanism for correlation

class EventCorrelator(BaseAnalyzer):
    """Analyzes correlations between semantic flow events"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("EventCorrelator")
        self.embeddings = embeddings_store
        self.config = config
        
        # Correlation analysis parameters
        self.correlation_config = {
            'max_time_lag': 5,              # Maximum time lag to consider
            'min_correlation_strength': 0.3, # Minimum correlation strength
            'significance_threshold': 0.05,   # Statistical significance threshold
            'semantic_similarity_threshold': 0.5,  # Threshold for semantic similarity
            'cascade_window': 3,             # Time window for cascade detection
            'causal_inference_method': 'granger',  # granger, transfer_entropy, ccm
            'temporal_resolution': 'day',    # Temporal resolution for analysis
            'bootstrap_samples': 1000,      # Bootstrap samples for confidence intervals
            'multiple_testing_correction': 'bonferroni'  # Multiple testing correction
        }
        
        # Event correlation results
        self.correlated_events = []
        self.correlation_network = {}
        self.causal_chains = []
        self.cascade_events = []
        
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive event correlation analysis"""
        # Extract events from embeddings and flows
        events = self._extract_events()
        
        # Detect temporal correlations
        temporal_correlations = self._detect_temporal_correlations(events)
        
        # Detect semantic correlations
        semantic_correlations = self._detect_semantic_correlations(events)
        
        # Detect cascade events
        cascade_correlations = self._detect_cascade_correlations(events)
        
        # Perform causal inference
        causal_correlations = self._detect_causal_correlations(events)
        
        # Build correlation network
        correlation_network = self._build_correlation_network(
            temporal_correlations + semantic_correlations + 
            cascade_correlations + causal_correlations
        )
        
        # Identify correlation patterns
        correlation_patterns = self._identify_correlation_patterns(correlation_network)
        
        # Generate correlation summary
        correlation_summary = self._generate_correlation_summary(
            temporal_correlations, semantic_correlations, 
            cascade_correlations, causal_correlations
        )
        
        return {
            'events': [self._serialize_event(e) for e in events],
            'temporal_correlations': [self._serialize_correlation(c) for c in temporal_correlations],
            'semantic_correlations': [self._serialize_correlation(c) for c in semantic_correlations],
            'cascade_correlations': [self._serialize_correlation(c) for c in cascade_correlations],
            'causal_correlations': [self._serialize_correlation(c) for c in causal_correlations],
            'correlation_network': correlation_network,
            'correlation_patterns': correlation_patterns,
            'summary': correlation_summary
        }
    
    def _extract_events(self) -> List[FlowEvent]:
        """Extract events from the semantic flow system"""
        events = []
        timestamps = self.embeddings.get_timestamps()
        
        for i, timestamp in enumerate(timestamps):
            vocabulary = self.embeddings.get_vocabulary(timestamp)
            
            # Detect various types of events
            # 1. New word appearances
            if i > 0:
                prev_vocab = set(self.embeddings.get_vocabulary(timestamps[i-1]))
                curr_vocab = set(vocabulary)
                new_words = curr_vocab - prev_vocab
                
                for word in new_words:
                    events.append(FlowEvent(
                        timestamp=timestamp,
                        event_type='word_emergence',
                        primary_word=word,
                        secondary_words=[],
                        magnitude=1.0,
                        confidence=0.9,
                        description=f"New word '{word}' appeared"
                    ))
            
            # 2. Word disappearances
            if i > 0:
                prev_vocab = set(self.embeddings.get_vocabulary(timestamps[i-1]))
                curr_vocab = set(vocabulary)
                lost_words = prev_vocab - curr_vocab
                
                for word in lost_words:
                    events.append(FlowEvent(
                        timestamp=timestamp,
                        event_type='word_disappearance',
                        primary_word=word,
                        secondary_words=[],
                        magnitude=1.0,
                        confidence=0.8,
                        description=f"Word '{word}' disappeared"
                    ))
            
            # 3. Semantic shifts (large changes in word embeddings)
            if i > 0:
                prev_timestamp = timestamps[i-1]
                semantic_shifts = self._detect_semantic_shifts(prev_timestamp, timestamp)
                events.extend(semantic_shifts)
            
            # 4. Burst events (sudden increases in word activity)
            burst_events = self._detect_burst_events(timestamp)
            events.extend(burst_events)
            
            # 5. Community formation/dissolution events
            community_events = self._detect_community_events(timestamp, i)
            events.extend(community_events)
        
        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    def _detect_semantic_shifts(self, prev_timestamp: Timestamp, 
                               curr_timestamp: Timestamp) -> List[FlowEvent]:
        """Detect semantic shift events"""
        events = []
        prev_vocab = set(self.embeddings.get_vocabulary(prev_timestamp))
        curr_vocab = set(self.embeddings.get_vocabulary(curr_timestamp))
        common_words = prev_vocab & curr_vocab
        
        for word in common_words:
            prev_emb = self.embeddings.get_embedding(word, prev_timestamp)
            curr_emb = self.embeddings.get_embedding(word, curr_timestamp)
            
            if prev_emb is not None and curr_emb is not None:
                # Compute semantic shift magnitude
                from scipy.spatial.distance import cosine
                shift_magnitude = cosine(prev_emb, curr_emb)
                
                # Threshold for significant shift
                if shift_magnitude > 0.3:
                    events.append(FlowEvent(
                        timestamp=curr_timestamp,
                        event_type='semantic_shift',
                        primary_word=word,
                        secondary_words=[],
                        magnitude=shift_magnitude,
                        confidence=0.85,
                        description=f"Semantic shift in '{word}' (magnitude: {shift_magnitude:.3f})"
                    ))
        
        return events
    
    def _detect_burst_events(self, timestamp: Timestamp) -> List[FlowEvent]:
        """Detect burst events at given timestamp"""
        events = []
        
        # This would typically analyze flow magnitudes, frequencies, etc.
        # For now, create placeholder burst detection
        vocabulary = self.embeddings.get_vocabulary(timestamp)
        
        # Sample some words as having burst activity
        sample_words = list(vocabulary)[:5]
        for word in sample_words:
            # Simulate burst detection
            burst_magnitude = np.random.random()
            if burst_magnitude > 0.7:
                events.append(FlowEvent(
                    timestamp=timestamp,
                    event_type='burst',
                    primary_word=word,
                    secondary_words=[],
                    magnitude=burst_magnitude,
                    confidence=0.7,
                    description=f"Burst activity in '{word}'"
                ))
        
        return events
    
    def _detect_community_events(self, timestamp: Timestamp, time_index: int) -> List[FlowEvent]:
        """Detect community formation/dissolution events"""
        events = []
        
        # This would analyze community structure changes
        # For now, create placeholder community events
        if time_index > 2 and np.random.random() > 0.8:
            vocabulary = list(self.embeddings.get_vocabulary(timestamp))
            if len(vocabulary) >= 3:
                community_words = vocabulary[:3]
                events.append(FlowEvent(
                    timestamp=timestamp,
                    event_type='community_formation',
                    primary_word=community_words[0],
                    secondary_words=community_words[1:],
                    magnitude=0.8,
                    confidence=0.6,
                    description=f"Community formed around {community_words}"
                ))
        
        return events
    
    def _detect_temporal_correlations(self, events: List[FlowEvent]) -> List[CorrelatedEvent]:
        """Detect temporal correlations between events"""
        correlations = []
        max_lag = self.correlation_config['max_time_lag']
        
        # Group events by timestamp
        events_by_time = defaultdict(list)
        for event in events:
            events_by_time[event.timestamp].append(event)
        
        timestamps = sorted(events_by_time.keys())
        
        # Check for temporal correlations
        for i, timestamp1 in enumerate(timestamps):
            for j in range(i + 1, min(i + max_lag + 1, len(timestamps))):
                timestamp2 = timestamps[j]
                time_lag = j - i
                
                events1 = events_by_time[timestamp1]
                events2 = events_by_time[timestamp2]
                
                # Check all pairs of events
                for event1 in events1:
                    for event2 in events2:
                        correlation = self._compute_temporal_correlation(event1, event2, time_lag)
                        
                        if correlation and correlation.correlation_strength >= self.correlation_config['min_correlation_strength']:
                            correlations.append(correlation)
        
        return correlations
    
    def _compute_temporal_correlation(self, event1: FlowEvent, event2: FlowEvent, 
                                    time_lag: int) -> Optional[CorrelatedEvent]:
        """Compute temporal correlation between two events"""
        # Check for temporal patterns
        correlation_strength = 0.0
        mechanism = ""
        
        # Same event type correlation
        if event1.event_type == event2.event_type:
            correlation_strength += 0.4
            mechanism += f"Same event type ({event1.event_type}); "
        
        # Word overlap correlation
        words1 = {event1.primary_word} | set(event1.secondary_words)
        words2 = {event2.primary_word} | set(event2.secondary_words)
        word_overlap = len(words1 & words2) / len(words1 | words2)
        correlation_strength += 0.3 * word_overlap
        if word_overlap > 0:
            mechanism += f"Word overlap ({word_overlap:.2f}); "
        
        # Magnitude correlation
        magnitude_correlation = 1.0 - abs(event1.magnitude - event2.magnitude)
        correlation_strength += 0.2 * magnitude_correlation
        
        # Temporal proximity bonus
        temporal_bonus = max(0, 1.0 - time_lag / self.correlation_config['max_time_lag'])
        correlation_strength += 0.1 * temporal_bonus
        
        if correlation_strength >= self.correlation_config['min_correlation_strength']:
            confidence = min(event1.confidence, event2.confidence) * correlation_strength
            
            return CorrelatedEvent(
                event1=event1,
                event2=event2,
                correlation_type='temporal',
                correlation_strength=correlation_strength,
                time_lag=time_lag,
                confidence=confidence,
                mechanism=mechanism.strip('; ')
            )
        
        return None
    
    def _detect_semantic_correlations(self, events: List[FlowEvent]) -> List[CorrelatedEvent]:
        """Detect semantic correlations between events"""
        correlations = []
        
        # Compare all pairs of events for semantic similarity
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                correlation = self._compute_semantic_correlation(event1, event2)
                
                if correlation and correlation.correlation_strength >= self.correlation_config['min_correlation_strength']:
                    correlations.append(correlation)
        
        return correlations
    
    def _compute_semantic_correlation(self, event1: FlowEvent, event2: FlowEvent) -> Optional[CorrelatedEvent]:
        """Compute semantic correlation between two events"""
        # Get embeddings for primary words
        emb1 = self.embeddings.get_embedding(event1.primary_word, event1.timestamp)
        emb2 = self.embeddings.get_embedding(event2.primary_word, event2.timestamp)
        
        if emb1 is None or emb2 is None:
            return None
        
        # Compute semantic similarity
        from scipy.spatial.distance import cosine
        semantic_similarity = 1 - cosine(emb1, emb2)
        
        if semantic_similarity >= self.correlation_config['semantic_similarity_threshold']:
            # Compute time lag
            timestamps = sorted([event1.timestamp, event2.timestamp])
            time_lag = abs(hash(event2.timestamp) - hash(event1.timestamp)) % 10  # Simplified
            
            correlation_strength = semantic_similarity
            confidence = min(event1.confidence, event2.confidence) * semantic_similarity
            
            mechanism = f"Semantic similarity between '{event1.primary_word}' and '{event2.primary_word}' ({semantic_similarity:.3f})"
            
            return CorrelatedEvent(
                event1=event1,
                event2=event2,
                correlation_type='semantic',
                correlation_strength=correlation_strength,
                time_lag=time_lag,
                confidence=confidence,
                mechanism=mechanism
            )
        
        return None
    
    def _detect_cascade_correlations(self, events: List[FlowEvent]) -> List[CorrelatedEvent]:
        """Detect cascade correlations between events"""
        correlations = []
        cascade_window = self.correlation_config['cascade_window']
        
        # Group events by timestamp
        events_by_time = defaultdict(list)
        for event in events:
            events_by_time[event.timestamp].append(event)
        
        timestamps = sorted(events_by_time.keys())
        
        # Look for cascade patterns
        for i, timestamp in enumerate(timestamps):
            primary_events = events_by_time[timestamp]
            
            # Check subsequent time windows for cascading events
            for j in range(i + 1, min(i + cascade_window + 1, len(timestamps))):
                secondary_timestamp = timestamps[j]
                secondary_events = events_by_time[secondary_timestamp]
                
                cascade_correlations = self._identify_cascade_patterns(
                    primary_events, secondary_events, j - i
                )
                correlations.extend(cascade_correlations)
        
        return correlations
    
    def _identify_cascade_patterns(self, primary_events: List[FlowEvent], 
                                 secondary_events: List[FlowEvent], 
                                 time_lag: int) -> List[CorrelatedEvent]:
        """Identify cascade patterns between event groups"""
        correlations = []
        
        for primary_event in primary_events:
            for secondary_event in secondary_events:
                # Check for cascade indicators
                cascade_strength = self._compute_cascade_strength(primary_event, secondary_event)
                
                if cascade_strength >= self.correlation_config['min_correlation_strength']:
                    confidence = min(primary_event.confidence, secondary_event.confidence) * cascade_strength
                    
                    mechanism = f"Cascade effect from {primary_event.event_type} to {secondary_event.event_type}"
                    
                    correlations.append(CorrelatedEvent(
                        event1=primary_event,
                        event2=secondary_event,
                        correlation_type='cascade',
                        correlation_strength=cascade_strength,
                        time_lag=time_lag,
                        confidence=confidence,
                        mechanism=mechanism
                    ))
        
        return correlations
    
    def _compute_cascade_strength(self, primary_event: FlowEvent, secondary_event: FlowEvent) -> float:
        """Compute strength of cascade relationship"""
        cascade_strength = 0.0
        
        # Event type cascade patterns
        cascade_patterns = {
            ('burst', 'semantic_shift'): 0.7,
            ('semantic_shift', 'word_emergence'): 0.6,
            ('word_emergence', 'community_formation'): 0.5,
            ('community_formation', 'burst'): 0.4
        }
        
        pattern = (primary_event.event_type, secondary_event.event_type)
        if pattern in cascade_patterns:
            cascade_strength += cascade_patterns[pattern]
        
        # Magnitude amplification
        if secondary_event.magnitude > primary_event.magnitude:
            amplification_factor = secondary_event.magnitude / primary_event.magnitude
            cascade_strength += 0.2 * min(1.0, amplification_factor - 1.0)
        
        # Word relationship
        if primary_event.primary_word in secondary_event.secondary_words:
            cascade_strength += 0.3
        
        return min(1.0, cascade_strength)
    
    def _detect_causal_correlations(self, events: List[FlowEvent]) -> List[CorrelatedEvent]:
        """Detect causal correlations using causal inference methods"""
        correlations = []
        method = self.correlation_config['causal_inference_method']
        
        if method == 'granger':
            correlations = self._granger_causality_analysis(events)
        elif method == 'transfer_entropy':
            correlations = self._transfer_entropy_analysis(events)
        elif method == 'ccm':
            correlations = self._ccm_analysis(events)
        
        return correlations
    
    def _granger_causality_analysis(self, events: List[FlowEvent]) -> List[CorrelatedEvent]:
        """Perform Granger causality analysis"""
        correlations = []
        
        # Create time series for different event types
        event_types = set(event.event_type for event in events)
        time_series = {event_type: [] for event_type in event_types}
        
        # Group by timestamp and count events
        events_by_time = defaultdict(lambda: defaultdict(int))
        for event in events:
            events_by_time[event.timestamp][event.event_type] += 1
        
        timestamps = sorted(events_by_time.keys())
        
        # Build time series
        for timestamp in timestamps:
            for event_type in event_types:
                count = events_by_time[timestamp][event_type]
                time_series[event_type].append(count)
        
        # Perform pairwise Granger causality tests
        for type1 in event_types:
            for type2 in event_types:
                if type1 != type2:
                    causality_score = self._compute_granger_causality(
                        time_series[type1], time_series[type2]
                    )
                    
                    if causality_score >= self.correlation_config['min_correlation_strength']:
                        # Create representative events for this causal relationship
                        events_type1 = [e for e in events if e.event_type == type1]
                        events_type2 = [e for e in events if e.event_type == type2]
                        
                        if events_type1 and events_type2:
                            # Use first occurrence of each type as representative
                            event1 = events_type1[0]
                            event2 = events_type2[0]
                            
                            correlations.append(CorrelatedEvent(
                                event1=event1,
                                event2=event2,
                                correlation_type='causal',
                                correlation_strength=causality_score,
                                time_lag=1,  # Simplified
                                confidence=causality_score * 0.8,
                                mechanism=f"Granger causality: {type1} → {type2}"
                            ))
        
        return correlations
    
    def _compute_granger_causality(self, series1: List[float], series2: List[float]) -> float:
        """Compute Granger causality score (simplified implementation)"""
        if len(series1) < 3 or len(series2) < 3:
            return 0.0
        
        # Simplified Granger causality: correlation between series1[t-1] and series2[t]
        if len(series1) != len(series2):
            min_len = min(len(series1), len(series2))
            series1 = series1[:min_len]
            series2 = series2[:min_len]
        
        if len(series1) < 2:
            return 0.0
        
        # Compute lagged correlation
        lagged_series1 = series1[:-1]
        future_series2 = series2[1:]
        
        if len(lagged_series1) == 0 or len(future_series2) == 0:
            return 0.0
        
        # Simplified correlation calculation
        correlation = np.corrcoef(lagged_series1, future_series2)[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _transfer_entropy_analysis(self, events: List[FlowEvent]) -> List[CorrelatedEvent]:
        """Perform transfer entropy analysis (simplified)"""
        # For now, return empty list - full implementation would require
        # more sophisticated information-theoretic measures
        return []
    
    def _ccm_analysis(self, events: List[FlowEvent]) -> List[CorrelatedEvent]:
        """Perform Convergent Cross Mapping analysis (simplified)"""
        # For now, return empty list - full implementation would require
        # nonlinear dynamics and embedding methods
        return []
    
    def _build_correlation_network(self, all_correlations: List[CorrelatedEvent]) -> Dict[str, Any]:
        """Build network representation of event correlations"""
        network = {
            'nodes': [],
            'edges': [],
            'node_attributes': {},
            'edge_attributes': {},
            'network_metrics': {}
        }
        
        # Extract unique events as nodes
        unique_events = set()
        for correlation in all_correlations:
            unique_events.add(self._event_to_node_id(correlation.event1))
            unique_events.add(self._event_to_node_id(correlation.event2))
        
        network['nodes'] = list(unique_events)
        
        # Add correlations as edges
        for correlation in all_correlations:
            node1 = self._event_to_node_id(correlation.event1)
            node2 = self._event_to_node_id(correlation.event2)
            
            edge = {
                'source': node1,
                'target': node2,
                'correlation_type': correlation.correlation_type,
                'strength': correlation.correlation_strength,
                'time_lag': correlation.time_lag,
                'confidence': correlation.confidence
            }
            
            network['edges'].append(edge)
        
        # Compute network metrics
        network['network_metrics'] = self._compute_network_metrics(network)
        
        return network
    
    def _event_to_node_id(self, event: FlowEvent) -> str:
        """Convert event to unique node identifier"""
        return f"{event.timestamp}_{event.event_type}_{event.primary_word}"
    
    def _compute_network_metrics(self, network: Dict[str, Any]) -> Dict[str, float]:
        """Compute basic network metrics"""
        nodes = network['nodes']
        edges = network['edges']
        
        if not nodes:
            return {}
        
        num_nodes = len(nodes)
        num_edges = len(edges)
        
        metrics = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0,
            'average_degree': 2 * num_edges / num_nodes if num_nodes > 0 else 0.0
        }
        
        # Average correlation strength
        if edges:
            strengths = [edge['strength'] for edge in edges]
            metrics['average_correlation_strength'] = np.mean(strengths)
            metrics['max_correlation_strength'] = np.max(strengths)
        
        return metrics
    
    def _identify_correlation_patterns(self, correlation_network: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in the correlation network"""
        patterns = {
            'correlation_type_distribution': {},
            'temporal_patterns': {},
            'strength_patterns': {},
            'hub_events': [],
            'correlation_chains': []
        }
        
        edges = correlation_network.get('edges', [])
        
        if not edges:
            return patterns
        
        # Correlation type distribution
        type_counts = defaultdict(int)
        for edge in edges:
            type_counts[edge['correlation_type']] += 1
        patterns['correlation_type_distribution'] = dict(type_counts)
        
        # Temporal patterns
        time_lags = [edge['time_lag'] for edge in edges]
        patterns['temporal_patterns'] = {
            'average_time_lag': np.mean(time_lags),
            'most_common_lag': max(set(time_lags), key=time_lags.count),
            'lag_distribution': dict(zip(*np.unique(time_lags, return_counts=True)))
        }
        
        # Strength patterns
        strengths = [edge['strength'] for edge in edges]
        patterns['strength_patterns'] = {
            'average_strength': np.mean(strengths),
            'strength_variance': np.var(strengths),
            'strong_correlations': len([s for s in strengths if s > 0.7])
        }
        
        # Identify hub events (events with many correlations)
        node_degrees = defaultdict(int)
        for edge in edges:
            node_degrees[edge['source']] += 1
            node_degrees[edge['target']] += 1
        
        if node_degrees:
            max_degree = max(node_degrees.values())
            hub_threshold = max_degree * 0.7
            patterns['hub_events'] = [
                node for node, degree in node_degrees.items() 
                if degree >= hub_threshold
            ]
        
        return patterns
    
    def _generate_correlation_summary(self, temporal_correlations: List[CorrelatedEvent],
                                    semantic_correlations: List[CorrelatedEvent],
                                    cascade_correlations: List[CorrelatedEvent],
                                    causal_correlations: List[CorrelatedEvent]) -> Dict[str, Any]:
        """Generate summary of correlation analysis"""
        all_correlations = (temporal_correlations + semantic_correlations + 
                          cascade_correlations + causal_correlations)
        
        if not all_correlations:
            return {'error': 'No correlations found'}
        
        summary = {
            'total_correlations': len(all_correlations),
            'correlation_breakdown': {
                'temporal': len(temporal_correlations),
                'semantic': len(semantic_correlations),
                'cascade': len(cascade_correlations),
                'causal': len(causal_correlations)
            },
            'average_correlation_strength': np.mean([c.correlation_strength for c in all_correlations]),
            'average_confidence': np.mean([c.confidence for c in all_correlations]),
            'average_time_lag': np.mean([c.time_lag for c in all_correlations]),
            'strongest_correlation_type': max(
                ['temporal', 'semantic', 'cascade', 'causal'],
                key=lambda t: len([c for c in all_correlations if c.correlation_type == t])
            ),
            'correlation_density': len(all_correlations) / max(1, len(set(
                [c.event1 for c in all_correlations] + [c.event2 for c in all_correlations]
            )))
        }
        
        return summary
    
    def _serialize_event(self, event: FlowEvent) -> Dict[str, Any]:
        """Serialize event for export"""
        return {
            'timestamp': event.timestamp,
            'event_type': event.event_type,
            'primary_word': event.primary_word,
            'secondary_words': event.secondary_words,
            'magnitude': event.magnitude,
            'confidence': event.confidence,
            'description': event.description
        }
    
    def _serialize_correlation(self, correlation: CorrelatedEvent) -> Dict[str, Any]:
        """Serialize correlation for export"""
        return {
            'event1': self._serialize_event(correlation.event1),
            'event2': self._serialize_event(correlation.event2),
            'correlation_type': correlation.correlation_type,
            'correlation_strength': correlation.correlation_strength,
            'time_lag': correlation.time_lag,
            'confidence': correlation.confidence,
            'mechanism': correlation.mechanism
        }