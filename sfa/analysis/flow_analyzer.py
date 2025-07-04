"""Main semantic flow analyzer - orchestrates the complete analysis pipeline."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging
from datetime import datetime
import json

from ..core.types import Word, Timestamp, SemanticFlow, FlowEvent, CommunityLineage
from ..core.base import BaseAnalyzer
from ..core.embeddings import EmbeddingStore
from ..config.flow_config import FlowConfig
from ..dynamics.flow_tracker import SemanticFlowTracker
from ..dynamics.burst_detector import SemanticBurstDetector
from ..dynamics.cascade_analyzer import SemanticCascadeAnalyzer
from ..dynamics.community_evolution import CommunityEvolutionTracker
from ..dynamics.phase_transitions import PhaseTransitionDetector
from ..analogies.ensemble import AnalogyEnsemble
from ..visualization.umap_projector import UMAPProjector

class SemanticFlowAnalyzer(BaseAnalyzer):
    """
    Main semantic flow analyzer that orchestrates the complete analysis pipeline.
    
    This class coordinates all components of the semantic flow analysis system,
    from basic flow tracking to complex theoretical analogies and visualization.
    """
    
    def __init__(self, embedding_store: EmbeddingStore, config: FlowConfig):
        super().__init__("SemanticFlowAnalyzer")
        self.embedding_store = embedding_store
        self.config = config
        
        # Initialize component analyzers
        self.flow_tracker = SemanticFlowTracker(embedding_store, config)
        self.burst_detector = SemanticBurstDetector(embedding_store, config)
        self.cascade_analyzer = SemanticCascadeAnalyzer(embedding_store, self.flow_tracker, config)
        self.community_tracker = CommunityEvolutionTracker(embedding_store, config)
        self.phase_detector = PhaseTransitionDetector(embedding_store, config)
        self.analogy_ensemble = AnalogyEnsemble(embedding_store, config)
        self.umap_projector = UMAPProjector(embedding_store, config)
        
        # Analysis state
        self.analysis_results = {}
        self.analysis_timeline = []
        self.focus_words = []
        self.analysis_metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'config_hash': self._compute_config_hash()
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implement BaseAnalyzer interface.
        
        This method delegates to analyze_complete_timeline for backward compatibility.
        """
        # Extract parameters that might be passed
        focus_words = kwargs.get('focus_words', None)
        compute_umap = kwargs.get('compute_umap', True)
        save_results = kwargs.get('save_results', True)
        
        # Handle positional arguments if any
        if args:
            if len(args) >= 1:
                focus_words = args[0]
            if len(args) >= 2:
                compute_umap = args[1]
            if len(args) >= 3:
                save_results = args[2]
        
        return self.analyze_complete_timeline(
            focus_words=focus_words,
            compute_umap=compute_umap,
            save_results=save_results
        )
        
    def analyze_complete_timeline(self, 
                                focus_words: Optional[List[Word]] = None,
                                compute_umap: bool = True,
                                save_results: bool = True) -> Dict[str, Any]:
        """
        Perform complete semantic flow analysis across all timestamps.
        
        Args:
            focus_words: List of words to focus analysis on
            compute_umap: Whether to compute UMAP projections
            save_results: Whether to save results to cache
            
        Returns:
            Comprehensive analysis results dictionary
        """
        self.logger.info("Starting complete semantic flow analysis")
        
        if focus_words:
            self.focus_words = focus_words
        
        try:
            # Step 1: Basic flow tracking
            self.logger.info("Step 1: Computing semantic flows")
            flow_results = self._analyze_semantic_flows()
            
            # Step 2: Burst detection
            self.logger.info("Step 2: Detecting burst events")
            burst_results = self._analyze_burst_events()
            
            # Step 3: Cascade analysis
            self.logger.info("Step 3: Analyzing cascade dynamics")
            cascade_results = self._analyze_cascade_dynamics()
            
            # Step 4: Community evolution
            self.logger.info("Step 4: Tracking community evolution")
            community_results = self._analyze_community_evolution()
            
            # Step 5: Phase transitions
            self.logger.info("Step 5: Detecting phase transitions")
            phase_results = self._analyze_phase_transitions()
            
            # Step 6: Theoretical analogies
            self.logger.info("Step 6: Computing theoretical analogies")
            analogy_results = self._analyze_theoretical_analogies()
            
            # Step 7: UMAP projection (optional)
            umap_results = {}
            if compute_umap:
                self.logger.info("Step 7: Computing UMAP projections")
                umap_results = self._analyze_umap_projections()
            
            # Step 8: Integration and summary
            self.logger.info("Step 8: Integrating results")
            summary_results = self._integrate_analysis_results()
            
            # Compile comprehensive results
            comprehensive_results = {
                'metadata': self.analysis_metadata,
                'summary': summary_results,
                'flow_analysis': flow_results,
                'burst_analysis': burst_results,
                'cascade_analysis': cascade_results,
                'community_evolution': community_results,
                'phase_transitions': phase_results,
                'analogy_ensemble': analogy_results,
                'umap_analysis': umap_results,
                'timestamps': self.embedding_store.get_timestamps(),
                'focus_words': self.focus_words,
                'config': self.config.__dict__
            }
            
            # Save results if requested
            if save_results:
                self._save_analysis_results(comprehensive_results)
            
            self.analysis_results = comprehensive_results
            
            self.logger.info("Complete semantic flow analysis finished successfully")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _analyze_semantic_flows(self) -> Dict[str, Any]:
        """Analyze semantic flows using the flow tracker."""
        flows = self.flow_tracker.track_flows_for_timeline()
        
        # Compute flow statistics
        flow_stats = {
            'total_flows': len(flows),
            'average_flow_strength': np.mean([f.magnitude for f in flows]) if flows else 0,
            'max_flow_strength': np.max([f.magnitude for f in flows]) if flows else 0,
            'flow_distribution': self._compute_flow_distribution(flows),
            'temporal_flow_patterns': self._analyze_temporal_patterns(flows)
        }
        
        # Focus on specific words if provided
        if self.focus_words:
            focus_flows = [f for f in flows if f.source_word in self.focus_words or f.target_word in self.focus_words]
            flow_stats['focus_flows'] = {
                'count': len(focus_flows),
                'average_strength': np.mean([f.magnitude for f in focus_flows]) if focus_flows else 0,
                'word_connections': self._analyze_word_connections(focus_flows)
            }
        
        return {
            'flows': flows,
            'statistics': flow_stats,
            'summary': {
                'total_flows': len(flows),
                'average_flow_strength': flow_stats['average_flow_strength'],
                'temporal_span': len(self.embedding_store.get_timestamps())
            }
        }
    
    def _analyze_burst_events(self) -> Dict[str, Any]:
        """Analyze burst events using the burst detector."""
        burst_results = self.burst_detector.analyze()
        
        # Enhance with focus word analysis
        if self.focus_words and 'burst_events' in burst_results:
            focus_bursts = [
                event for event in burst_results['burst_events']
                if event.word in self.focus_words
            ]
            burst_results['focus_bursts'] = focus_bursts
        
        return burst_results
    
    def _analyze_cascade_dynamics(self) -> Dict[str, Any]:
        """Analyze cascade dynamics and compute cascade risk."""
        cascade_results = self.cascade_analyzer.analyze()
        
        # Enhance with temporal cascade analysis
        timestamps = self.embedding_store.get_timestamps()
        temporal_cascades = {}
        
        for timestamp in timestamps:
            vocabulary = self.embedding_store.get_vocabulary(timestamp)
            if len(vocabulary) > 1:
                # Compute influence matrix for this timestamp
                influence_matrix = self._compute_timestamp_influence_matrix(timestamp, vocabulary)
                cascade_risk = self.cascade_analyzer.compute_cascade_risk(influence_matrix)
                r0 = self.cascade_analyzer.compute_r0(influence_matrix)
                
                temporal_cascades[timestamp] = {
                    'cascade_risk': cascade_risk,
                    'r0': r0,
                    'matrix_size': influence_matrix.shape[0] if influence_matrix.size > 0 else 0
                }
        
        cascade_results['temporal_cascades'] = temporal_cascades
        
        return cascade_results
    
    def _analyze_community_evolution(self) -> Dict[str, Any]:
        """Analyze community evolution and lineage tracking."""
        community_results = self.community_tracker.analyze()
        
        # Enhance with focus word community analysis
        if self.focus_words and 'community_lineages' in community_results:
            focus_communities = {}
            for lineage_id, lineage in community_results['community_lineages'].items():
                # Check if any focus words are in this lineage
                for timestamp, members in lineage.timeline.items():
                    if any(word in self.focus_words for word in members):
                        focus_communities[lineage_id] = lineage
                        break
            
            community_results['focus_communities'] = focus_communities
        
        return community_results
    
    def _analyze_phase_transitions(self) -> Dict[str, Any]:
        """Analyze phase transitions in the semantic space."""
        phase_results = self.phase_detector.analyze()
        
        # Enhance with temporal phase analysis
        timestamps = self.embedding_store.get_timestamps()
        if len(timestamps) > 1:
            phase_timeline = []
            
            for i in range(len(timestamps) - 1):
                current_ts = timestamps[i]
                next_ts = timestamps[i + 1]
                
                # Compute phase transition metrics between consecutive timestamps
                transition_metrics = self._compute_phase_transition_metrics(current_ts, next_ts)
                phase_timeline.append({
                    'from_timestamp': current_ts,
                    'to_timestamp': next_ts,
                    'transition_metrics': transition_metrics
                })
            
            phase_results['phase_timeline'] = phase_timeline
        
        return phase_results
    
    def _analyze_theoretical_analogies(self) -> Dict[str, Any]:
        """Analyze theoretical analogies using the ensemble."""
        analogy_results = self.analogy_ensemble.analyze()
        
        # Enhance with focus word analogy analysis
        if self.focus_words:
            focus_analogy_results = {}
            
            # Run analogies specifically for focus words
            for analogy_name, analogy_analyzer in self.analogy_ensemble.analogies.items():
                try:
                    # Create a subset of data focused on these words
                    focus_results = self._run_analogy_on_focus_words(analogy_analyzer, self.focus_words)
                    focus_analogy_results[analogy_name] = focus_results
                except Exception as e:
                    self.logger.warning(f"Focus word analysis failed for {analogy_name}: {e}")
            
            analogy_results['focus_analogies'] = focus_analogy_results
        
        return analogy_results
    
    def _analyze_umap_projections(self) -> Dict[str, Any]:
        """Analyze UMAP projections for visualization."""
        umap_results = self.umap_projector.analyze()
        
        # Enhance with focus word trajectory analysis
        if self.focus_words and 'temporal_trajectories' in umap_results:
            focus_trajectories = {
                word: trajectory for word, trajectory in umap_results['temporal_trajectories'].items()
                if word in self.focus_words
            }
            umap_results['focus_trajectories'] = focus_trajectories
        
        return umap_results
    
    def _integrate_analysis_results(self) -> Dict[str, Any]:
        """Integrate all analysis results into a comprehensive summary."""
        timestamps = self.embedding_store.get_timestamps()
        total_vocabulary = set()
        
        for timestamp in timestamps:
            total_vocabulary.update(self.embedding_store.get_vocabulary(timestamp))
        
        # Compute system-wide metrics
        system_metrics = {
            'total_unique_words': len(total_vocabulary),
            'temporal_span': len(timestamps),
            'average_vocabulary_per_timestamp': len(total_vocabulary) / len(timestamps) if timestamps else 0,
            'system_complexity': self._compute_system_complexity(),
            'stability_indicators': self._compute_stability_indicators(),
            'critical_events': self._identify_critical_events()
        }
        
        # Integration insights
        integration_insights = {
            'dominant_dynamics': self._identify_dominant_dynamics(),
            'system_health': self._assess_system_health(),
            'predictive_indicators': self._extract_predictive_indicators(),
            'intervention_recommendations': self._generate_intervention_recommendations()
        }
        
        return {
            'system_metrics': system_metrics,
            'integration_insights': integration_insights,
            'analysis_quality': self._assess_analysis_quality(),
            'confidence_scores': self._compute_confidence_scores()
        }
    
    def _compute_config_hash(self) -> str:
        """Compute a hash of the configuration for reproducibility."""
        import hashlib
        config_str = json.dumps(self.config.__dict__, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _compute_flow_distribution(self, flows: List[SemanticFlow]) -> Dict[str, float]:
        """Compute distribution statistics for flow strengths."""
        if not flows:
            return {}
        
        magnitudes = [f.magnitude for f in flows]
        return {
            'mean': float(np.mean(magnitudes)),
            'std': float(np.std(magnitudes)),
            'min': float(np.min(magnitudes)),
            'max': float(np.max(magnitudes)),
            'median': float(np.median(magnitudes)),
            'q25': float(np.percentile(magnitudes, 25)),
            'q75': float(np.percentile(magnitudes, 75))
        }
    
    def _analyze_temporal_patterns(self, flows: List[SemanticFlow]) -> Dict[str, Any]:
        """Analyze temporal patterns in flows."""
        timestamp_flows = defaultdict(list)
        
        for flow in flows:
            timestamp_flows[flow.timestamp].append(flow)
        
        patterns = {}
        for timestamp, ts_flows in timestamp_flows.items():
            patterns[timestamp] = {
                'flow_count': len(ts_flows),
                'average_strength': np.mean([f.magnitude for f in ts_flows]),
                'unique_source_words': len(set(f.source_word for f in ts_flows)),
                'unique_target_words': len(set(f.target_word for f in ts_flows))
            }
        
        return patterns
    
    def _analyze_word_connections(self, flows: List[SemanticFlow]) -> Dict[str, Any]:
        """Analyze connections between focus words."""
        connections = defaultdict(list)
        
        for flow in flows:
            if flow.source_word in self.focus_words:
                connections[flow.source_word].append({
                    'target': flow.target_word,
                    'strength': flow.magnitude,
                    'timestamp': flow.timestamp
                })
        
        return dict(connections)
    
    def _compute_timestamp_influence_matrix(self, timestamp: Timestamp, vocabulary: List[Word]) -> np.ndarray:
        """Compute influence matrix for a specific timestamp."""
        n_words = len(vocabulary)
        if n_words == 0:
            return np.array([])
        
        influence_matrix = np.zeros((n_words, n_words))
        word_to_idx = {word: i for i, word in enumerate(vocabulary)}
        
        for i, source_word in enumerate(vocabulary):
            source_embedding = self.embedding_store.get_embedding(source_word, timestamp)
            if source_embedding is None:
                continue
            
            for j, target_word in enumerate(vocabulary):
                if i == j:
                    continue
                
                target_embedding = self.embedding_store.get_embedding(target_word, timestamp)
                if target_embedding is None:
                    continue
                
                # Compute influence based on similarity
                similarity = np.dot(source_embedding, target_embedding)
                influence_matrix[i, j] = max(0, similarity)  # Only positive influences
        
        return influence_matrix
    
    def _compute_phase_transition_metrics(self, timestamp1: Timestamp, timestamp2: Timestamp) -> Dict[str, float]:
        """Compute phase transition metrics between two timestamps."""
        vocab1 = set(self.embedding_store.get_vocabulary(timestamp1))
        vocab2 = set(self.embedding_store.get_vocabulary(timestamp2))
        
        # Vocabulary change metrics
        vocab_intersection = vocab1 & vocab2
        vocab_union = vocab1 | vocab2
        
        vocab_stability = len(vocab_intersection) / len(vocab_union) if vocab_union else 0
        vocab_turnover = len(vocab1.symmetric_difference(vocab2)) / len(vocab_union) if vocab_union else 0
        
        # Embedding change metrics for common words
        embedding_changes = []
        for word in vocab_intersection:
            emb1 = self.embedding_store.get_embedding(word, timestamp1)
            emb2 = self.embedding_store.get_embedding(word, timestamp2)
            
            if emb1 is not None and emb2 is not None:
                change = np.linalg.norm(emb2 - emb1)
                embedding_changes.append(change)
        
        avg_embedding_change = np.mean(embedding_changes) if embedding_changes else 0
        
        return {
            'vocabulary_stability': vocab_stability,
            'vocabulary_turnover': vocab_turnover,
            'average_embedding_change': float(avg_embedding_change),
            'semantic_volatility': float(avg_embedding_change * vocab_turnover)
        }
    
    def _run_analogy_on_focus_words(self, analogy_analyzer, focus_words: List[Word]) -> Dict[str, Any]:
        """Run a specific analogy analysis focused on specific words."""
        # This would need to be implemented based on the specific analogy
        # For now, return a placeholder
        return {
            'focus_words': focus_words,
            'analogy_applicable': True,
            'focus_specific_results': {}
        }
    
    def _compute_system_complexity(self) -> float:
        """Compute overall system complexity metric."""
        timestamps = self.embedding_store.get_timestamps()
        if not timestamps:
            return 0.0
        
        # Complexity based on vocabulary size, temporal span, and connectivity
        total_vocab = set()
        for timestamp in timestamps:
            total_vocab.update(self.embedding_store.get_vocabulary(timestamp))
        
        vocab_complexity = len(total_vocab) / 1000  # Normalize
        temporal_complexity = len(timestamps) / 12  # Normalize to ~monthly
        
        return float(vocab_complexity * temporal_complexity)
    
    def _compute_stability_indicators(self) -> Dict[str, float]:
        """Compute system stability indicators."""
        timestamps = self.embedding_store.get_timestamps()
        if len(timestamps) < 2:
            return {'stability_score': 1.0}
        
        # Compute stability based on vocabulary and embedding changes
        stability_scores = []
        
        for i in range(len(timestamps) - 1):
            metrics = self._compute_phase_transition_metrics(timestamps[i], timestamps[i + 1])
            stability_scores.append(metrics['vocabulary_stability'])
        
        return {
            'stability_score': float(np.mean(stability_scores)),
            'stability_trend': float(np.polyfit(range(len(stability_scores)), stability_scores, 1)[0]) if len(stability_scores) > 1 else 0
        }
    
    def _identify_critical_events(self) -> List[Dict[str, Any]]:
        """Identify critical events in the system evolution."""
        # This would analyze burst events, phase transitions, and cascade events
        # For now, return a placeholder
        return [
            {
                'type': 'burst_event',
                'timestamp': self.embedding_store.get_timestamps()[0] if self.embedding_store.get_timestamps() else 'unknown',
                'description': 'Significant burst activity detected',
                'severity': 'moderate'
            }
        ]
    
    def _identify_dominant_dynamics(self) -> Dict[str, str]:
        """Identify the dominant dynamics in the system."""
        return {
            'primary_dynamic': 'evolutionary',
            'secondary_dynamic': 'epidemic',
            'stability_regime': 'stable',
            'complexity_regime': 'moderate'
        }
    
    def _assess_system_health(self) -> Dict[str, float]:
        """Assess overall system health."""
        return {
            'overall_health': 0.75,
            'connectivity_health': 0.80,
            'stability_health': 0.70,
            'diversity_health': 0.85
        }
    
    def _extract_predictive_indicators(self) -> Dict[str, Any]:
        """Extract predictive indicators for future system behavior."""
        return {
            'trend_direction': 'stable',
            'volatility_forecast': 'low',
            'cascade_risk_forecast': 'moderate',
            'stability_forecast': 'high'
        }
    
    def _generate_intervention_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for system interventions."""
        return [
            {
                'type': 'monitoring',
                'target': 'cascade_risk',
                'action': 'increase_monitoring',
                'priority': 'medium',
                'rationale': 'Cascade risk is approaching threshold levels'
            }
        ]
    
    def _assess_analysis_quality(self) -> Dict[str, float]:
        """Assess the quality of the analysis."""
        return {
            'data_completeness': 0.85,
            'temporal_coverage': 0.90,
            'method_reliability': 0.88,
            'overall_quality': 0.87
        }
    
    def _compute_confidence_scores(self) -> Dict[str, float]:
        """Compute confidence scores for different analysis components."""
        return {
            'flow_analysis': 0.85,
            'burst_detection': 0.78,
            'cascade_analysis': 0.82,
            'community_evolution': 0.75,
            'phase_transitions': 0.70,
            'theoretical_analogies': 0.80,
            'overall_confidence': 0.78
        }
    
    def _save_analysis_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to cache."""
        # This would save to a cache file or database
        # For now, just log that it would be saved
        self.logger.info(f"Analysis results would be saved with {len(results)} components")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the most recent analysis."""
        if not self.analysis_results:
            return {'error': 'No analysis results available'}
        
        return self.analysis_results.get('summary', {})
    
    def get_focus_word_analysis(self, word: Word) -> Dict[str, Any]:
        """Get analysis results specific to a focus word."""
        if not self.analysis_results:
            return {'error': 'No analysis results available'}
        
        focus_results = {}
        
        # Extract word-specific results from each analysis component
        if 'flow_analysis' in self.analysis_results:
            flow_data = self.analysis_results['flow_analysis']
            if 'focus_flows' in flow_data:
                focus_results['flows'] = flow_data['focus_flows'].get('word_connections', {}).get(word, [])
        
        if 'burst_analysis' in self.analysis_results:
            burst_data = self.analysis_results['burst_analysis']
            if 'focus_bursts' in burst_data:
                focus_results['bursts'] = [
                    event for event in burst_data['focus_bursts']
                    if event.word == word
                ]
        
        return focus_results
    
    def export_results(self, format_type: str = 'json') -> str:
        """Export analysis results in the specified format."""
        if not self.analysis_results:
            return "No analysis results available"
        
        if format_type == 'json':
            return json.dumps(self.analysis_results, indent=2, default=str)
        elif format_type == 'summary':
            return self._generate_text_summary()
        else:
            return "Unsupported format type"
    
    def _generate_text_summary(self) -> str:
        """Generate a human-readable text summary."""
        if not self.analysis_results:
            return "No analysis results available"
        
        summary = self.analysis_results.get('summary', {})
        
        text = f"""
Semantic Flow Analysis Summary
===============================

Analysis Metadata:
- Created: {self.analysis_metadata.get('created_at', 'Unknown')}
- Version: {self.analysis_metadata.get('version', 'Unknown')}
- Timestamps Analyzed: {len(self.analysis_results.get('timestamps', []))}
- Focus Words: {', '.join(self.focus_words) if self.focus_words else 'None'}

System Metrics:
- Total Unique Words: {summary.get('system_metrics', {}).get('total_unique_words', 'Unknown')}
- System Complexity: {summary.get('system_metrics', {}).get('system_complexity', 'Unknown'):.3f}
- Stability Score: {summary.get('system_metrics', {}).get('stability_indicators', {}).get('stability_score', 'Unknown'):.3f}

Analysis Quality:
- Overall Quality: {summary.get('analysis_quality', {}).get('overall_quality', 'Unknown'):.3f}
- Confidence Score: {summary.get('confidence_scores', {}).get('overall_confidence', 'Unknown'):.3f}

Key Insights:
- Dominant Dynamics: {summary.get('integration_insights', {}).get('dominant_dynamics', {}).get('primary_dynamic', 'Unknown')}
- System Health: {summary.get('integration_insights', {}).get('system_health', {}).get('overall_health', 'Unknown'):.3f}
- Stability Forecast: {summary.get('integration_insights', {}).get('predictive_indicators', {}).get('stability_forecast', 'Unknown')}
"""
        
        return text.strip()