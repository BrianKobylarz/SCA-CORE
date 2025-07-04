"""Intervention advisor for semantic flow systems."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

from ..core.types import Word, Timestamp, SemanticFlow
from ..core.base import BaseAnalyzer
from ..analogies.ensemble import AnalogyEnsemble

class InterventionType(Enum):
    """Types of interventions for semantic flow systems"""
    AMPLIFICATION = "amplification"
    DAMPENING = "dampening"  
    REDIRECTION = "redirection"
    STABILIZATION = "stabilization"
    DIVERSIFICATION = "diversification"
    CONSENSUS_BUILDING = "consensus_building"
    FRAGMENTATION_PREVENTION = "fragmentation_prevention"
    CRITICAL_TRANSITION_MANAGEMENT = "critical_transition_management"

@dataclass
class InterventionRecommendation:
    """Recommendation for system intervention"""
    intervention_type: InterventionType
    target_words: List[Word]
    target_timestamps: List[Timestamp] 
    strength: float  # 0-1, strength of intervention needed
    urgency: float   # 0-1, urgency of intervention
    confidence: float  # 0-1, confidence in recommendation
    mechanism: str   # Description of intervention mechanism
    expected_outcome: str  # Expected result of intervention
    risks: List[str]  # Potential risks or side effects
    success_metrics: List[str]  # How to measure success

class InterventionAdvisor(BaseAnalyzer):
    """Provides intervention recommendations for semantic flow systems"""
    
    def __init__(self, flow_analyzer, analogy_ensemble: AnalogyEnsemble, config):
        super().__init__("InterventionAdvisor")
        self.flow_analyzer = flow_analyzer
        self.analogy_ensemble = analogy_ensemble
        self.config = config
        
        # Intervention parameters
        self.intervention_config = {
            'intervention_threshold': 0.6,  # Threshold for recommending intervention
            'urgency_threshold': 0.8,       # Threshold for urgent interventions
            'confidence_threshold': 0.5,    # Minimum confidence for recommendations
            'max_recommendations': 10,      # Maximum number of recommendations
            'time_horizon': 5,              # Time steps to consider for predictions
            'risk_tolerance': 0.3,          # Tolerance for intervention risks
            'multi_objective_optimization': True,
            'consider_uncertainty': True,
            'adaptive_thresholds': True
        }
        
        # Intervention knowledge base
        self.intervention_strategies = {
            InterventionType.AMPLIFICATION: self._amplification_strategies,
            InterventionType.DAMPENING: self._dampening_strategies,
            InterventionType.REDIRECTION: self._redirection_strategies,
            InterventionType.STABILIZATION: self._stabilization_strategies,
            InterventionType.DIVERSIFICATION: self._diversification_strategies,
            InterventionType.CONSENSUS_BUILDING: self._consensus_building_strategies,
            InterventionType.FRAGMENTATION_PREVENTION: self._fragmentation_prevention_strategies,
            InterventionType.CRITICAL_TRANSITION_MANAGEMENT: self._critical_transition_strategies
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Generate comprehensive intervention analysis"""
        # Analyze current system state
        system_state = self._analyze_system_state()
        
        # Identify intervention opportunities
        intervention_opportunities = self._identify_intervention_opportunities(system_state)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(intervention_opportunities, system_state)
        
        # Prioritize recommendations
        prioritized_recommendations = self._prioritize_recommendations(recommendations)
        
        # Assess intervention feasibility
        feasibility_analysis = self._assess_intervention_feasibility(prioritized_recommendations)
        
        # Generate intervention timeline
        intervention_timeline = self._generate_intervention_timeline(prioritized_recommendations)
        
        return {
            'system_state': system_state,
            'intervention_opportunities': intervention_opportunities,
            'recommendations': [self._serialize_recommendation(r) for r in prioritized_recommendations],
            'feasibility_analysis': feasibility_analysis,
            'intervention_timeline': intervention_timeline,
            'summary': self._generate_intervention_summary(prioritized_recommendations)
        }
    
    def _analyze_system_state(self) -> Dict[str, Any]:
        """Analyze current state of the semantic flow system"""
        # Get latest analysis from flow analyzer
        flow_analysis = self.flow_analyzer.analyze()
        
        # Get analogy ensemble analysis
        analogy_analysis = self.analogy_ensemble.analyze()
        
        # Extract key indicators
        system_state = {
            'stability_indicators': self._extract_stability_indicators(flow_analysis),
            'critical_indicators': self._extract_critical_indicators(flow_analysis, analogy_analysis),
            'flow_patterns': self._extract_flow_patterns(flow_analysis),
            'consensus_state': self._extract_consensus_state(analogy_analysis),
            'risk_indicators': self._extract_risk_indicators(flow_analysis, analogy_analysis),
            'temporal_trends': self._extract_temporal_trends(flow_analysis)
        }
        
        return system_state
    
    def _extract_stability_indicators(self, flow_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Extract system stability indicators"""
        stability_indicators = {}
        
        # Flow stability
        if 'flow_metrics' in flow_analysis:
            flow_metrics = flow_analysis['flow_metrics']
            
            # Variance in flow magnitudes
            if 'magnitude_variance' in flow_metrics:
                stability_indicators['flow_magnitude_stability'] = 1.0 / (1.0 + flow_metrics['magnitude_variance'])
            
            # Network stability
            if 'network_metrics' in flow_metrics:
                network_metrics = flow_metrics['network_metrics']
                if 'modularity_variance' in network_metrics:
                    stability_indicators['network_stability'] = 1.0 / (1.0 + network_metrics['modularity_variance'])
        
        # Community stability
        if 'community_evolution' in flow_analysis:
            community_data = flow_analysis['community_evolution']
            if 'summary_statistics' in community_data:
                stats = community_data['summary_statistics']
                stability_indicators['community_stability'] = stats.get('stability_score', 0.0)
        
        return stability_indicators
    
    def _extract_critical_indicators(self, flow_analysis: Dict[str, Any], 
                                   analogy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract indicators of critical behavior"""
        critical_indicators = {}
        
        # Phase transition indicators
        if 'phase_transitions' in flow_analysis:
            phase_data = flow_analysis['phase_transitions']
            critical_indicators['phase_transition_risk'] = len(phase_data.get('phase_transitions', []))
        
        # Cascade risk
        if 'cascade_analysis' in flow_analysis:
            cascade_data = flow_analysis['cascade_analysis']
            critical_indicators['cascade_risk'] = cascade_data.get('cascade_risk', 0.0)
            critical_indicators['r0'] = cascade_data.get('r0', 1.0)
        
        # Analogy-based criticality
        if 'individual_results' in analogy_analysis:
            individual_results = analogy_analysis['individual_results']
            
            # Epidemic criticality
            if 'epidemic' in individual_results:
                epidemic_data = individual_results['epidemic']
                if 'parameters' in epidemic_data:
                    r0 = epidemic_data['parameters'].get('basic_reproduction_number', 1.0)
                    critical_indicators['epidemic_criticality'] = max(0.0, r0 - 1.0)
            
            # Ferromagnetic criticality
            if 'ferromagnetic' in individual_results:
                ferro_data = individual_results['ferromagnetic']
                if 'critical_behavior' in ferro_data:
                    critical_indicators['magnetic_criticality'] = ferro_data['critical_behavior'].get('reduced_temperature', 0.0)
        
        return critical_indicators
    
    def _extract_flow_patterns(self, flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract flow pattern characteristics"""
        flow_patterns = {}
        
        if 'flow_network' in flow_analysis:
            network_data = flow_analysis['flow_network']
            
            # Flow directionality
            flow_patterns['directionality'] = network_data.get('average_flow_direction', 0.0)
            
            # Flow concentration
            flow_patterns['concentration'] = network_data.get('flow_concentration', 0.0)
            
            # Temporal patterns
            if 'temporal_patterns' in network_data:
                temporal = network_data['temporal_patterns']
                flow_patterns['periodicity'] = temporal.get('dominant_period', 0.0)
                flow_patterns['trend'] = temporal.get('trend_strength', 0.0)
        
        return flow_patterns
    
    def _extract_consensus_state(self, analogy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract consensus state from analogy analysis"""
        consensus_state = {}
        
        if 'individual_results' in analogy_analysis:
            individual_results = analogy_analysis['individual_results']
            
            # Bounded confidence consensus
            if 'bounded_confidence' in individual_results:
                bc_data = individual_results['bounded_confidence']
                if 'consensus_evolution' in bc_data:
                    consensus_evolution = bc_data['consensus_evolution']
                    # Get latest consensus metrics
                    if consensus_evolution:
                        latest_timestamp = max(consensus_evolution.keys())
                        latest_metrics = consensus_evolution[latest_timestamp]
                        consensus_state.update(latest_metrics)
        
        return consensus_state
    
    def _extract_risk_indicators(self, flow_analysis: Dict[str, Any], 
                               analogy_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Extract risk indicators"""
        risk_indicators = {}
        
        # Cascade risk
        if 'cascade_analysis' in flow_analysis:
            cascade_data = flow_analysis['cascade_analysis']
            risk_indicators['cascade_risk'] = cascade_data.get('cascade_risk', 0.0)
        
        # Fragmentation risk
        if 'individual_results' in analogy_analysis:
            individual_results = analogy_analysis['individual_results']
            
            if 'bounded_confidence' in individual_results:
                bc_data = individual_results['bounded_confidence']
                if 'consensus_evolution' in bc_data:
                    consensus_data = bc_data['consensus_evolution']
                    if consensus_data:
                        latest_timestamp = max(consensus_data.keys())
                        fragmentation = consensus_data[latest_timestamp].get('fragmentation_level', 0.0)
                        risk_indicators['fragmentation_risk'] = fragmentation
        
        # Extinction risk (evolutionary analogy)
        if 'individual_results' in analogy_analysis:
            individual_results = analogy_analysis['individual_results']
            
            if 'evolutionary' in individual_results:
                evo_data = individual_results['evolutionary']
                if 'phylogenetic_analysis' in evo_data:
                    phylo = evo_data['phylogenetic_analysis']
                    extinction_rate = phylo.get('extinct_lineages', 0) / max(phylo.get('total_lineages', 1), 1)
                    risk_indicators['extinction_risk'] = extinction_rate
        
        return risk_indicators
    
    def _extract_temporal_trends(self, flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal trends"""
        temporal_trends = {}
        
        # Flow trends
        if 'flow_metrics' in flow_analysis:
            flow_metrics = flow_analysis['flow_metrics']
            if 'temporal_analysis' in flow_metrics:
                temporal_analysis = flow_metrics['temporal_analysis']
                temporal_trends['flow_trend'] = temporal_analysis.get('trend_direction', 'stable')
                temporal_trends['acceleration'] = temporal_analysis.get('acceleration', 0.0)
        
        return temporal_trends
    
    def _identify_intervention_opportunities(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for intervention"""
        opportunities = []
        
        # Check for critical transitions
        critical_indicators = system_state.get('critical_indicators', {})
        
        if critical_indicators.get('cascade_risk', 0.0) > 0.5:
            opportunities.append({
                'type': 'cascade_prevention',
                'urgency': critical_indicators['cascade_risk'],
                'description': 'High cascade risk detected'
            })
        
        if critical_indicators.get('epidemic_criticality', 0.0) > 0.2:
            opportunities.append({
                'type': 'epidemic_control',
                'urgency': critical_indicators['epidemic_criticality'],
                'description': 'Epidemic spread above critical threshold'
            })
        
        # Check stability indicators
        stability_indicators = system_state.get('stability_indicators', {})
        
        for indicator_name, stability in stability_indicators.items():
            if stability < 0.3:  # Low stability
                opportunities.append({
                    'type': 'stabilization',
                    'urgency': 1.0 - stability,
                    'description': f'Low {indicator_name} detected',
                    'target_metric': indicator_name
                })
        
        # Check consensus state
        consensus_state = system_state.get('consensus_state', {})
        
        if consensus_state.get('fragmentation_level', 0.0) > 0.7:
            opportunities.append({
                'type': 'fragmentation_prevention',
                'urgency': consensus_state['fragmentation_level'],
                'description': 'High fragmentation detected'
            })
        
        if consensus_state.get('polarization_index', 0.0) > 0.6:
            opportunities.append({
                'type': 'consensus_building',
                'urgency': consensus_state['polarization_index'],
                'description': 'High polarization detected'
            })
        
        # Check risk indicators
        risk_indicators = system_state.get('risk_indicators', {})
        
        for risk_type, risk_level in risk_indicators.items():
            if risk_level > 0.6:
                opportunities.append({
                    'type': 'risk_mitigation',
                    'urgency': risk_level,
                    'description': f'High {risk_type} detected',
                    'risk_type': risk_type
                })
        
        return opportunities
    
    def _generate_recommendations(self, opportunities: List[Dict[str, Any]], 
                                system_state: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Generate intervention recommendations"""
        recommendations = []
        
        for opportunity in opportunities:
            intervention_type = self._map_opportunity_to_intervention_type(opportunity)
            
            if intervention_type:
                # Generate specific recommendations for this intervention type
                strategy_generator = self.intervention_strategies.get(intervention_type)
                
                if strategy_generator:
                    strategies = strategy_generator(opportunity, system_state)
                    recommendations.extend(strategies)
        
        return recommendations
    
    def _map_opportunity_to_intervention_type(self, opportunity: Dict[str, Any]) -> Optional[InterventionType]:
        """Map intervention opportunity to intervention type"""
        opportunity_type = opportunity.get('type', '')
        
        mapping = {
            'cascade_prevention': InterventionType.DAMPENING,
            'epidemic_control': InterventionType.DAMPENING,
            'stabilization': InterventionType.STABILIZATION,
            'fragmentation_prevention': InterventionType.FRAGMENTATION_PREVENTION,
            'consensus_building': InterventionType.CONSENSUS_BUILDING,
            'risk_mitigation': InterventionType.CRITICAL_TRANSITION_MANAGEMENT
        }
        
        return mapping.get(opportunity_type)
    
    def _amplification_strategies(self, opportunity: Dict[str, Any], 
                                system_state: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Generate amplification intervention strategies"""
        strategies = []
        
        # Strategy 1: Boost influential words
        strategies.append(InterventionRecommendation(
            intervention_type=InterventionType.AMPLIFICATION,
            target_words=self._identify_influential_words(system_state),
            target_timestamps=self._identify_optimal_timing(system_state),
            strength=0.7,
            urgency=opportunity.get('urgency', 0.5),
            confidence=0.8,
            mechanism="Increase exposure and usage of high-influence words",
            expected_outcome="Accelerated spread of desired semantic patterns",
            risks=["Potential oversaturation", "Uncontrolled cascade effects"],
            success_metrics=["Increased flow magnitude", "Expanded semantic reach"]
        ))
        
        return strategies
    
    def _dampening_strategies(self, opportunity: Dict[str, Any], 
                            system_state: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Generate dampening intervention strategies"""
        strategies = []
        
        # Strategy 1: Reduce cascade risk
        if opportunity.get('type') == 'cascade_prevention':
            strategies.append(InterventionRecommendation(
                intervention_type=InterventionType.DAMPENING,
                target_words=self._identify_cascade_sources(system_state),
                target_timestamps=self._identify_critical_timestamps(system_state),
                strength=0.8,
                urgency=opportunity.get('urgency', 0.7),
                confidence=0.9,
                mechanism="Reduce transmission rate of high-risk semantic flows",
                expected_outcome="Prevention of harmful cascade effects",
                risks=["Reduced beneficial information spread"],
                success_metrics=["Decreased cascade risk", "Lower R0 values"]
            ))
        
        # Strategy 2: Control epidemic spread
        if opportunity.get('type') == 'epidemic_control':
            strategies.append(InterventionRecommendation(
                intervention_type=InterventionType.DAMPENING,
                target_words=self._identify_superspreaders(system_state),
                target_timestamps=self._identify_peak_transmission_times(system_state),
                strength=0.9,
                urgency=opportunity.get('urgency', 0.8),
                confidence=0.85,
                mechanism="Isolate or reduce activity of superspreader words",
                expected_outcome="Controlled semantic epidemic spread",
                risks=["Information bottlenecks", "Reduced system connectivity"],
                success_metrics=["R0 below threshold", "Flattened infection curve"]
            ))
        
        return strategies
    
    def _redirection_strategies(self, opportunity: Dict[str, Any], 
                              system_state: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Generate redirection intervention strategies"""
        strategies = []
        
        strategies.append(InterventionRecommendation(
            intervention_type=InterventionType.REDIRECTION,
            target_words=self._identify_flow_hubs(system_state),
            target_timestamps=self._identify_flow_peaks(system_state),
            strength=0.6,
            urgency=opportunity.get('urgency', 0.4),
            confidence=0.7,
            mechanism="Redirect semantic flows toward beneficial pathways",
            expected_outcome="Improved flow patterns and outcomes",
            risks=["Disruption of natural flow patterns"],
            success_metrics=["Improved flow efficiency", "Better semantic outcomes"]
        ))
        
        return strategies
    
    def _stabilization_strategies(self, opportunity: Dict[str, Any], 
                                system_state: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Generate stabilization intervention strategies"""
        strategies = []
        
        target_metric = opportunity.get('target_metric', 'overall_stability')
        
        strategies.append(InterventionRecommendation(
            intervention_type=InterventionType.STABILIZATION,
            target_words=self._identify_stability_anchors(system_state),
            target_timestamps=self._identify_instability_periods(system_state),
            strength=0.7,
            urgency=opportunity.get('urgency', 0.6),
            confidence=0.8,
            mechanism=f"Strengthen {target_metric} through anchor word reinforcement",
            expected_outcome="Increased system stability and predictability",
            risks=["Reduced system adaptability"],
            success_metrics=[f"Improved {target_metric}", "Lower variance in key metrics"]
        ))
        
        return strategies
    
    def _diversification_strategies(self, opportunity: Dict[str, Any], 
                                  system_state: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Generate diversification intervention strategies"""
        strategies = []
        
        strategies.append(InterventionRecommendation(
            intervention_type=InterventionType.DIVERSIFICATION,
            target_words=self._identify_diversity_gaps(system_state),
            target_timestamps=self._identify_low_diversity_periods(system_state),
            strength=0.5,
            urgency=opportunity.get('urgency', 0.3),
            confidence=0.6,
            mechanism="Introduce semantic diversity in underexplored regions",
            expected_outcome="Increased semantic richness and resilience",
            risks=["Potential fragmentation", "Reduced coherence"],
            success_metrics=["Increased semantic diversity", "Better coverage of semantic space"]
        ))
        
        return strategies
    
    def _consensus_building_strategies(self, opportunity: Dict[str, Any], 
                                     system_state: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Generate consensus building intervention strategies"""
        strategies = []
        
        strategies.append(InterventionRecommendation(
            intervention_type=InterventionType.CONSENSUS_BUILDING,
            target_words=self._identify_bridging_words(system_state),
            target_timestamps=self._identify_polarization_peaks(system_state),
            strength=0.8,
            urgency=opportunity.get('urgency', 0.7),
            confidence=0.75,
            mechanism="Promote bridging words that connect polarized groups",
            expected_outcome="Reduced polarization and increased consensus",
            risks=["Forced artificial consensus", "Suppression of legitimate diversity"],
            success_metrics=["Increased consensus strength", "Reduced polarization index"]
        ))
        
        return strategies
    
    def _fragmentation_prevention_strategies(self, opportunity: Dict[str, Any], 
                                           system_state: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Generate fragmentation prevention strategies"""
        strategies = []
        
        strategies.append(InterventionRecommendation(
            intervention_type=InterventionType.FRAGMENTATION_PREVENTION,
            target_words=self._identify_fragmentation_points(system_state),
            target_timestamps=self._identify_fragmentation_risks(system_state),
            strength=0.9,
            urgency=opportunity.get('urgency', 0.8),
            confidence=0.8,
            mechanism="Strengthen connections at fragmentation-prone points",
            expected_outcome="Maintained system coherence and connectivity",
            risks=["Suppression of natural differentiation"],
            success_metrics=["Reduced fragmentation level", "Maintained connectivity"]
        ))
        
        return strategies
    
    def _critical_transition_strategies(self, opportunity: Dict[str, Any], 
                                      system_state: Dict[str, Any]) -> List[InterventionRecommendation]:
        """Generate critical transition management strategies"""
        strategies = []
        
        risk_type = opportunity.get('risk_type', 'unknown')
        
        strategies.append(InterventionRecommendation(
            intervention_type=InterventionType.CRITICAL_TRANSITION_MANAGEMENT,
            target_words=self._identify_critical_nodes(system_state, risk_type),
            target_timestamps=self._identify_critical_periods(system_state, risk_type),
            strength=0.9,
            urgency=opportunity.get('urgency', 0.9),
            confidence=0.7,
            mechanism=f"Manage {risk_type} through targeted interventions at critical points",
            expected_outcome="Controlled or prevented critical transitions",
            risks=["Interference with natural system evolution"],
            success_metrics=[f"Reduced {risk_type}", "Stable system state"]
        ))
        
        return strategies
    
    def _identify_influential_words(self, system_state: Dict[str, Any]) -> List[Word]:
        """Identify most influential words in the system"""
        # This would analyze flow patterns, centrality measures, etc.
        # For now, return placeholder
        return ["innovation", "technology", "growth"]
    
    def _identify_cascade_sources(self, system_state: Dict[str, Any]) -> List[Word]:
        """Identify words that are sources of cascade risk"""
        return ["viral", "trending", "explosive"]
    
    def _identify_superspreaders(self, system_state: Dict[str, Any]) -> List[Word]:
        """Identify superspreader words in epidemic model"""
        return ["influencer", "popular", "mainstream"]
    
    def _identify_flow_hubs(self, system_state: Dict[str, Any]) -> List[Word]:
        """Identify central hub words in flow network"""
        return ["central", "core", "hub"]
    
    def _identify_stability_anchors(self, system_state: Dict[str, Any]) -> List[Word]:
        """Identify words that could serve as stability anchors"""
        return ["stable", "consistent", "reliable"]
    
    def _identify_bridging_words(self, system_state: Dict[str, Any]) -> List[Word]:
        """Identify words that bridge polarized groups"""
        return ["common", "shared", "universal"]
    
    def _identify_fragmentation_points(self, system_state: Dict[str, Any]) -> List[Word]:
        """Identify words at fragmentation-prone points"""
        return ["divisive", "controversial", "polarizing"]
    
    def _identify_critical_nodes(self, system_state: Dict[str, Any], risk_type: str) -> List[Word]:
        """Identify critical nodes for specific risk type"""
        return ["critical", "threshold", "tipping"]
    
    def _identify_optimal_timing(self, system_state: Dict[str, Any]) -> List[Timestamp]:
        """Identify optimal timing for interventions"""
        return ["2024-01", "2024-02", "2024-03"]
    
    def _identify_critical_timestamps(self, system_state: Dict[str, Any]) -> List[Timestamp]:
        """Identify critical timestamps requiring intervention"""
        return ["2024-01", "2024-02"]
    
    def _identify_peak_transmission_times(self, system_state: Dict[str, Any]) -> List[Timestamp]:
        """Identify times of peak transmission"""
        return ["2024-01", "2024-03"]
    
    def _identify_flow_peaks(self, system_state: Dict[str, Any]) -> List[Timestamp]:
        """Identify timestamps with peak flow activity"""
        return ["2024-02", "2024-03"]
    
    def _identify_instability_periods(self, system_state: Dict[str, Any]) -> List[Timestamp]:
        """Identify periods of system instability"""
        return ["2024-01", "2024-04"]
    
    def _identify_low_diversity_periods(self, system_state: Dict[str, Any]) -> List[Timestamp]:
        """Identify periods of low semantic diversity"""
        return ["2024-02"]
    
    def _identify_polarization_peaks(self, system_state: Dict[str, Any]) -> List[Timestamp]:
        """Identify times of peak polarization"""
        return ["2024-01", "2024-03"]
    
    def _identify_fragmentation_risks(self, system_state: Dict[str, Any]) -> List[Timestamp]:
        """Identify timestamps with fragmentation risk"""
        return ["2024-02", "2024-04"]
    
    def _identify_critical_periods(self, system_state: Dict[str, Any], risk_type: str) -> List[Timestamp]:
        """Identify critical periods for specific risk type"""
        return ["2024-01", "2024-02"]
    
    def _identify_diversity_gaps(self, system_state: Dict[str, Any]) -> List[Word]:
        """Identify gaps in semantic diversity"""
        return ["niche", "specialized", "underexplored"]
    
    def _prioritize_recommendations(self, recommendations: List[InterventionRecommendation]) -> List[InterventionRecommendation]:
        """Prioritize recommendations by urgency, confidence, and impact"""
        def priority_score(rec: InterventionRecommendation) -> float:
            return (
                0.4 * rec.urgency +
                0.3 * rec.confidence +
                0.3 * rec.strength
            )
        
        # Sort by priority score
        sorted_recs = sorted(recommendations, key=priority_score, reverse=True)
        
        # Limit to maximum recommendations
        max_recs = self.intervention_config['max_recommendations']
        return sorted_recs[:max_recs]
    
    def _assess_intervention_feasibility(self, recommendations: List[InterventionRecommendation]) -> Dict[str, Any]:
        """Assess feasibility of intervention recommendations"""
        feasibility_analysis = {
            'overall_feasibility': 0.0,
            'individual_feasibility': {},
            'resource_requirements': {},
            'implementation_challenges': [],
            'success_probability': 0.0
        }
        
        individual_feasibilities = []
        
        for i, rec in enumerate(recommendations):
            # Assess various feasibility factors
            technical_feasibility = self._assess_technical_feasibility(rec)
            resource_feasibility = self._assess_resource_feasibility(rec)
            timing_feasibility = self._assess_timing_feasibility(rec)
            
            overall_feasibility = np.mean([technical_feasibility, resource_feasibility, timing_feasibility])
            individual_feasibilities.append(overall_feasibility)
            
            feasibility_analysis['individual_feasibility'][f'recommendation_{i}'] = {
                'technical': technical_feasibility,
                'resource': resource_feasibility,
                'timing': timing_feasibility,
                'overall': overall_feasibility
            }
        
        if individual_feasibilities:
            feasibility_analysis['overall_feasibility'] = np.mean(individual_feasibilities)
            feasibility_analysis['success_probability'] = np.mean([
                rec.confidence * feasibility for rec, feasibility 
                in zip(recommendations, individual_feasibilities)
            ])
        
        return feasibility_analysis
    
    def _assess_technical_feasibility(self, recommendation: InterventionRecommendation) -> float:
        """Assess technical feasibility of intervention"""
        # Assess based on intervention type complexity
        complexity_scores = {
            InterventionType.AMPLIFICATION: 0.8,
            InterventionType.DAMPENING: 0.7,
            InterventionType.REDIRECTION: 0.6,
            InterventionType.STABILIZATION: 0.7,
            InterventionType.DIVERSIFICATION: 0.5,
            InterventionType.CONSENSUS_BUILDING: 0.4,
            InterventionType.FRAGMENTATION_PREVENTION: 0.6,
            InterventionType.CRITICAL_TRANSITION_MANAGEMENT: 0.3
        }
        
        return complexity_scores.get(recommendation.intervention_type, 0.5)
    
    def _assess_resource_feasibility(self, recommendation: InterventionRecommendation) -> float:
        """Assess resource requirements feasibility"""
        # Consider number of target words and timestamps
        word_complexity = min(1.0, 10.0 / len(recommendation.target_words))
        time_complexity = min(1.0, 5.0 / len(recommendation.target_timestamps))
        strength_factor = 1.0 - recommendation.strength * 0.3
        
        return np.mean([word_complexity, time_complexity, strength_factor])
    
    def _assess_timing_feasibility(self, recommendation: InterventionRecommendation) -> float:
        """Assess timing feasibility of intervention"""
        # Higher urgency means timing is more critical
        timing_pressure = 1.0 - recommendation.urgency * 0.5
        return timing_pressure
    
    def _generate_intervention_timeline(self, recommendations: List[InterventionRecommendation]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate intervention implementation timeline"""
        timeline = {}
        
        for rec in recommendations:
            for timestamp in rec.target_timestamps:
                if timestamp not in timeline:
                    timeline[timestamp] = []
                
                timeline[timestamp].append({
                    'intervention_type': rec.intervention_type.value,
                    'target_words': rec.target_words,
                    'strength': rec.strength,
                    'urgency': rec.urgency,
                    'mechanism': rec.mechanism,
                    'expected_outcome': rec.expected_outcome
                })
        
        # Sort timeline by timestamp
        sorted_timeline = dict(sorted(timeline.items()))
        
        return sorted_timeline
    
    def _generate_intervention_summary(self, recommendations: List[InterventionRecommendation]) -> Dict[str, Any]:
        """Generate summary of intervention recommendations"""
        if not recommendations:
            return {'error': 'No recommendations generated'}
        
        # Count intervention types
        type_counts = {}
        for rec in recommendations:
            type_name = rec.intervention_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Compute aggregate metrics
        avg_urgency = np.mean([rec.urgency for rec in recommendations])
        avg_confidence = np.mean([rec.confidence for rec in recommendations])
        avg_strength = np.mean([rec.strength for rec in recommendations])
        
        # Identify most critical recommendations
        urgent_recs = [rec for rec in recommendations if rec.urgency > self.intervention_config['urgency_threshold']]
        
        return {
            'total_recommendations': len(recommendations),
            'intervention_types': type_counts,
            'average_urgency': avg_urgency,
            'average_confidence': avg_confidence,
            'average_strength': avg_strength,
            'urgent_recommendations': len(urgent_recs),
            'most_urgent_type': max(type_counts.keys(), key=lambda k: type_counts[k]) if type_counts else None,
            'overall_risk_level': 'high' if avg_urgency > 0.7 else 'medium' if avg_urgency > 0.4 else 'low'
        }
    
    def _serialize_recommendation(self, rec: InterventionRecommendation) -> Dict[str, Any]:
        """Serialize recommendation for export"""
        return {
            'intervention_type': rec.intervention_type.value,
            'target_words': rec.target_words,
            'target_timestamps': rec.target_timestamps,
            'strength': rec.strength,
            'urgency': rec.urgency,
            'confidence': rec.confidence,
            'mechanism': rec.mechanism,
            'expected_outcome': rec.expected_outcome,
            'risks': rec.risks,
            'success_metrics': rec.success_metrics
        }