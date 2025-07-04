"""Evolutionary dynamics analogy for semantic flow analysis."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from scipy.integrate import odeint
from collections import defaultdict

from ..core.types import Word, Timestamp, SemanticFlow
from .base_analogy import BaseAnalogy

class EvolutionaryAnalogy(BaseAnalogy):
    """Models semantic evolution using evolutionary dynamics principles"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("EvolutionaryAnalogy", embeddings_store, config)
        
        # Evolutionary parameters
        self.analogy_parameters = {
            'mutation_rate': 0.01,           # μ: rate of semantic mutation
            'selection_strength': 0.1,       # s: strength of selection
            'drift_coefficient': 0.05,       # genetic drift strength
            'migration_rate': 0.02,          # rate of semantic migration
            'population_size': 1000,         # effective population size
            'carrying_capacity': 10000,      # maximum semantic diversity
            'fitness_landscape_ruggedness': 0.3,  # complexity of fitness landscape
            'recombination_rate': 0.1,       # rate of semantic recombination
            'neutral_evolution_fraction': 0.7,  # fraction of neutral evolution
            'adaptive_evolution_fraction': 0.3,  # fraction of adaptive evolution
            'evolutionary_model': 'wright_fisher'  # wright_fisher, moran, etc.
        }
        
        # Population genetics structures
        self.semantic_populations = {}    # timestamp -> population state
        self.fitness_landscapes = {}     # timestamp -> fitness landscape
        self.evolutionary_trajectories = {}  # lineage tracking
        self.selection_coefficients = {}  # word -> selection coefficient
        
    def fit_model(self, flows: List[SemanticFlow]) -> Dict[str, Any]:
        """Fit evolutionary model to semantic flows"""
        if not flows:
            return {'parameters': self.analogy_parameters, 'fit_quality': 0.0}
        
        # Organize flows by timestamp
        flow_timeline = self._organize_flows_by_time(flows)
        
        # Map semantic states to population genetics
        self._map_semantic_to_populations(flow_timeline)
        
        # Compute fitness landscapes
        self._compute_fitness_landscapes(flow_timeline)
        
        # Track evolutionary trajectories
        self._track_evolutionary_trajectories(flow_timeline)
        
        # Fit evolutionary parameters
        fitted_parameters = self._fit_evolutionary_parameters(flow_timeline)
        
        # Analyze selection vs drift
        selection_drift_analysis = self._analyze_selection_vs_drift(fitted_parameters)
        
        # Detect evolutionary events
        evolutionary_events = self._detect_evolutionary_events(flow_timeline)
        
        return {
            'parameters': fitted_parameters,
            'population_states': self.semantic_populations,
            'fitness_landscapes': self.fitness_landscapes,
            'evolutionary_trajectories': self.evolutionary_trajectories,
            'selection_drift_analysis': selection_drift_analysis,
            'evolutionary_events': evolutionary_events,
            'regime': self._determine_evolutionary_regime(fitted_parameters),
            'phylogenetic_analysis': self._analyze_phylogenetics(flow_timeline),
            'interpretation': self._interpret_evolutionary_dynamics(fitted_parameters)
        }
    
    def _organize_flows_by_time(self, flows: List[SemanticFlow]) -> Dict[Timestamp, List[SemanticFlow]]:
        """Organize flows by timestamp"""
        flow_timeline = {}
        for flow in flows:
            timestamp = flow.timestamp
            if timestamp not in flow_timeline:
                flow_timeline[timestamp] = []
            flow_timeline[timestamp].append(flow)
        return flow_timeline
    
    def _map_semantic_to_populations(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> None:
        """Map semantic states to population genetics framework"""
        for timestamp, flows in flow_timeline.items():
            # Get vocabulary as population
            vocabulary = list(self.embeddings.get_vocabulary(timestamp))
            
            # Compute population characteristics
            population_state = {
                'total_population': len(vocabulary),
                'active_variants': set(),
                'frequency_distribution': {},
                'diversity_metrics': {}
            }
            
            # Identify active semantic variants
            for flow in flows:
                population_state['active_variants'].add(flow.source_word)
                population_state['active_variants'].add(flow.target_word)
            
            # Compute frequency distribution
            flow_counts = defaultdict(int)
            for flow in flows:
                flow_counts[flow.source_word] += 1
                flow_counts[flow.target_word] += 1
            
            total_flow = sum(flow_counts.values())
            if total_flow > 0:
                for word, count in flow_counts.items():
                    population_state['frequency_distribution'][word] = count / total_flow
            
            # Compute diversity metrics
            population_state['diversity_metrics'] = self._compute_diversity_metrics(
                vocabulary, population_state['frequency_distribution']
            )
            
            self.semantic_populations[timestamp] = population_state
    
    def _compute_diversity_metrics(self, vocabulary: List[Word], 
                                  frequency_dist: Dict[Word, float]) -> Dict[str, float]:
        """Compute population diversity metrics"""
        if not vocabulary:
            return {}
        
        # Shannon diversity
        shannon_diversity = 0.0
        for word in vocabulary:
            freq = frequency_dist.get(word, 1.0 / len(vocabulary))
            if freq > 0:
                shannon_diversity -= freq * np.log(freq)
        
        # Simpson diversity
        simpson_diversity = 0.0
        for word in vocabulary:
            freq = frequency_dist.get(word, 1.0 / len(vocabulary))
            simpson_diversity += freq ** 2
        simpson_diversity = 1.0 - simpson_diversity
        
        # Effective population size
        effective_pop_size = len(vocabulary) * simpson_diversity
        
        return {
            'shannon_diversity': shannon_diversity,
            'simpson_diversity': simpson_diversity,
            'effective_population_size': effective_pop_size,
            'species_richness': len(vocabulary),
            'evenness': shannon_diversity / np.log(len(vocabulary)) if len(vocabulary) > 1 else 0.0
        }
    
    def _compute_fitness_landscapes(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> None:
        """Compute fitness landscapes for semantic variants"""
        from scipy.spatial.distance import cosine
        
        for timestamp, flows in flow_timeline.items():
            vocabulary = list(self.embeddings.get_vocabulary(timestamp))
            fitness_landscape = {}
            
            for word in vocabulary:
                # Fitness based on semantic connectivity and flow activity
                word_emb = self.embeddings.get_embedding(word, timestamp)
                if word_emb is None:
                    continue
                
                # Connectivity fitness
                connectivity = 0
                for other_word in vocabulary:
                    if other_word != word:
                        other_emb = self.embeddings.get_embedding(other_word, timestamp)
                        if other_emb is not None:
                            similarity = 1 - cosine(word_emb, other_emb)
                            if similarity > 0.3:
                                connectivity += similarity
                
                # Flow activity fitness
                flow_activity = 0
                for flow in flows:
                    if flow.source_word == word or flow.target_word == word:
                        flow_activity += flow.magnitude
                
                # Combined fitness
                fitness = (
                    0.5 * connectivity / len(vocabulary) +
                    0.5 * flow_activity / max(1, len(flows))
                )
                
                fitness_landscape[word] = fitness
            
            self.fitness_landscapes[timestamp] = fitness_landscape
    
    def _track_evolutionary_trajectories(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> None:
        """Track evolutionary trajectories of semantic variants"""
        timestamps = sorted(flow_timeline.keys())
        
        if len(timestamps) < 2:
            return
        
        # Initialize trajectories
        lineage_counter = 0
        active_lineages = {}
        
        for timestamp in timestamps:
            vocabulary = set(self.embeddings.get_vocabulary(timestamp))
            
            if not self.evolutionary_trajectories:
                # Initialize lineages
                for word in vocabulary:
                    lineage_id = f"lineage_{lineage_counter}"
                    lineage_counter += 1
                    
                    self.evolutionary_trajectories[lineage_id] = {
                        'word': word,
                        'birth_time': timestamp,
                        'death_time': None,
                        'trajectory': [(timestamp, 1.0)],  # (time, frequency)
                        'fitness_history': [],
                        'evolutionary_events': []
                    }
                    active_lineages[word] = lineage_id
            else:
                # Update existing lineages
                previous_vocab = set()
                for lineage in self.evolutionary_trajectories.values():
                    if lineage['death_time'] is None:
                        previous_vocab.add(lineage['word'])
                
                # Handle births and deaths
                new_words = vocabulary - previous_vocab
                lost_words = previous_vocab - vocabulary
                
                # Handle deaths
                for word in lost_words:
                    if word in active_lineages:
                        lineage_id = active_lineages[word]
                        self.evolutionary_trajectories[lineage_id]['death_time'] = timestamp
                        self.evolutionary_trajectories[lineage_id]['evolutionary_events'].append(
                            {'type': 'extinction', 'timestamp': timestamp}
                        )
                        del active_lineages[word]
                
                # Handle births
                for word in new_words:
                    lineage_id = f"lineage_{lineage_counter}"
                    lineage_counter += 1
                    
                    self.evolutionary_trajectories[lineage_id] = {
                        'word': word,
                        'birth_time': timestamp,
                        'death_time': None,
                        'trajectory': [(timestamp, 1.0)],
                        'fitness_history': [],
                        'evolutionary_events': [{'type': 'birth', 'timestamp': timestamp}]
                    }
                    active_lineages[word] = lineage_id
                
                # Update surviving lineages
                for word in vocabulary & previous_vocab:
                    if word in active_lineages:
                        lineage_id = active_lineages[word]
                        
                        # Update frequency
                        freq = self.semantic_populations[timestamp]['frequency_distribution'].get(word, 0.0)
                        self.evolutionary_trajectories[lineage_id]['trajectory'].append((timestamp, freq))
                        
                        # Update fitness
                        fitness = self.fitness_landscapes[timestamp].get(word, 0.0)
                        self.evolutionary_trajectories[lineage_id]['fitness_history'].append(fitness)
    
    def _fit_evolutionary_parameters(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> Dict[str, Any]:
        """Fit evolutionary parameters from data"""
        fitted_params = self.analogy_parameters.copy()
        
        # Estimate mutation rate from new word appearances
        timestamps = sorted(flow_timeline.keys())
        
        if len(timestamps) > 1:
            birth_rates = []
            death_rates = []
            
            for i in range(1, len(timestamps)):
                prev_vocab = set(self.embeddings.get_vocabulary(timestamps[i-1]))
                curr_vocab = set(self.embeddings.get_vocabulary(timestamps[i]))
                
                births = len(curr_vocab - prev_vocab)
                deaths = len(prev_vocab - curr_vocab)
                
                if len(prev_vocab) > 0:
                    birth_rates.append(births / len(prev_vocab))
                    death_rates.append(deaths / len(prev_vocab))
            
            if birth_rates:
                fitted_params['mutation_rate'] = np.mean(birth_rates)
            if death_rates:
                fitted_params['drift_coefficient'] = np.mean(death_rates)
        
        # Estimate selection strength from fitness variance
        if self.fitness_landscapes:
            all_fitness = []
            for fitness_landscape in self.fitness_landscapes.values():
                all_fitness.extend(fitness_landscape.values())
            
            if all_fitness:
                fitness_variance = np.var(all_fitness)
                fitted_params['selection_strength'] = fitness_variance
        
        # Estimate effective population size
        if self.semantic_populations:
            pop_sizes = []
            for pop_state in self.semantic_populations.values():
                eff_pop = pop_state['diversity_metrics'].get('effective_population_size', 0)
                if eff_pop > 0:
                    pop_sizes.append(eff_pop)
            
            if pop_sizes:
                fitted_params['population_size'] = np.mean(pop_sizes)
        
        return fitted_params
    
    def _analyze_selection_vs_drift(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relative importance of selection vs drift"""
        s = parameters['selection_strength']
        N = parameters['population_size']
        
        # Effective selection strength
        Ne_s = N * s
        
        # Determine dominant evolutionary force
        if Ne_s > 1:
            dominant_force = 'selection'
            regime = 'adaptive'
        elif Ne_s < 0.1:
            dominant_force = 'drift'
            regime = 'neutral'
        else:
            dominant_force = 'both'
            regime = 'mixed'
        
        return {
            'effective_selection': Ne_s,
            'dominant_force': dominant_force,
            'evolutionary_regime': regime,
            'selection_efficiency': s,
            'drift_strength': 1.0 / N,
            'selection_drift_ratio': Ne_s
        }
    
    def _detect_evolutionary_events(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> List[Dict[str, Any]]:
        """Detect major evolutionary events"""
        events = []
        timestamps = sorted(flow_timeline.keys())
        
        for i in range(1, len(timestamps)):
            prev_time = timestamps[i-1]
            curr_time = timestamps[i]
            
            # Detect population bottlenecks
            prev_pop = self.semantic_populations[prev_time]['total_population']
            curr_pop = self.semantic_populations[curr_time]['total_population']
            
            if curr_pop < prev_pop * 0.7:
                events.append({
                    'type': 'population_bottleneck',
                    'timestamp': curr_time,
                    'severity': 1.0 - (curr_pop / prev_pop),
                    'description': f'Population reduced from {prev_pop} to {curr_pop}'
                })
            
            # Detect adaptive radiations
            elif curr_pop > prev_pop * 1.3:
                events.append({
                    'type': 'adaptive_radiation',
                    'timestamp': curr_time,
                    'expansion_factor': curr_pop / prev_pop,
                    'description': f'Population expanded from {prev_pop} to {curr_pop}'
                })
            
            # Detect selective sweeps
            prev_diversity = self.semantic_populations[prev_time]['diversity_metrics'].get('shannon_diversity', 0)
            curr_diversity = self.semantic_populations[curr_time]['diversity_metrics'].get('shannon_diversity', 0)
            
            if curr_diversity < prev_diversity * 0.8:
                events.append({
                    'type': 'selective_sweep',
                    'timestamp': curr_time,
                    'diversity_loss': 1.0 - (curr_diversity / prev_diversity),
                    'description': f'Diversity reduced from {prev_diversity:.3f} to {curr_diversity:.3f}'
                })
        
        return events
    
    def _determine_evolutionary_regime(self, parameters: Dict[str, Any]) -> str:
        """Determine overall evolutionary regime"""
        Ne_s = parameters['population_size'] * parameters['selection_strength']
        mutation_rate = parameters['mutation_rate']
        
        if Ne_s > 10:
            return 'strong_selection'
        elif Ne_s > 1:
            return 'weak_selection'
        elif mutation_rate > 0.1:
            return 'high_mutation'
        else:
            return 'neutral_drift'
    
    def _analyze_phylogenetics(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> Dict[str, Any]:
        """Analyze phylogenetic relationships"""
        phylogenetic_analysis = {
            'total_lineages': len(self.evolutionary_trajectories),
            'extinct_lineages': 0,
            'living_lineages': 0,
            'average_lifespan': 0,
            'phylogenetic_diversity': 0
        }
        
        if not self.evolutionary_trajectories:
            return phylogenetic_analysis
        
        # Count extinct vs living
        lifespans = []
        for lineage in self.evolutionary_trajectories.values():
            if lineage['death_time'] is not None:
                phylogenetic_analysis['extinct_lineages'] += 1
                # Compute lifespan
                birth_time = lineage['birth_time']
                death_time = lineage['death_time']
                lifespan = self._compute_time_difference(birth_time, death_time)
                lifespans.append(lifespan)
            else:
                phylogenetic_analysis['living_lineages'] += 1
        
        if lifespans:
            phylogenetic_analysis['average_lifespan'] = np.mean(lifespans)
        
        # Compute phylogenetic diversity
        phylogenetic_analysis['phylogenetic_diversity'] = self._compute_phylogenetic_diversity()
        
        return phylogenetic_analysis
    
    def _compute_time_difference(self, time1: Timestamp, time2: Timestamp) -> float:
        """Compute time difference between timestamps"""
        # Simplified - assume sequential timestamps
        return 1.0
    
    def _compute_phylogenetic_diversity(self) -> float:
        """Compute phylogenetic diversity"""
        if not self.evolutionary_trajectories:
            return 0.0
        
        # Simplified phylogenetic diversity based on lineage overlap
        living_lineages = [
            lineage for lineage in self.evolutionary_trajectories.values()
            if lineage['death_time'] is None
        ]
        
        if len(living_lineages) < 2:
            return 0.0
        
        # Average pairwise distance (simplified)
        total_distance = 0
        count = 0
        
        for i, lineage1 in enumerate(living_lineages):
            for lineage2 in living_lineages[i+1:]:
                # Compute semantic distance
                word1 = lineage1['word']
                word2 = lineage2['word']
                
                # Use last available timestamp
                timestamps = sorted(self.fitness_landscapes.keys())
                if timestamps:
                    last_timestamp = timestamps[-1]
                    emb1 = self.embeddings.get_embedding(word1, last_timestamp)
                    emb2 = self.embeddings.get_embedding(word2, last_timestamp)
                    
                    if emb1 is not None and emb2 is not None:
                        from scipy.spatial.distance import cosine
                        distance = cosine(emb1, emb2)
                        total_distance += distance
                        count += 1
        
        if count > 0:
            return total_distance / count
        else:
            return 0.0
    
    def _interpret_evolutionary_dynamics(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Interpret evolutionary dynamics in semantic terms"""
        regime = self._determine_evolutionary_regime(parameters)
        
        interpretations = {
            'evolutionary_analogy': 'Semantic change follows evolutionary principles',
            'population_meaning': 'Vocabulary represents the population of semantic variants',
            'fitness_meaning': 'Semantic fitness based on connectivity and usage',
            'selection_meaning': 'Beneficial semantic variants increase in frequency',
            'drift_meaning': 'Random semantic changes due to sampling effects',
            'mutation_meaning': 'New semantic variants arise through innovation',
            'regime_meaning': f'System operates in {regime} regime'
        }
        
        if regime == 'strong_selection':
            interpretations['dynamics'] = 'Adaptive semantic evolution dominates'
        elif regime == 'weak_selection':
            interpretations['dynamics'] = 'Selection and drift both important'
        elif regime == 'neutral_drift':
            interpretations['dynamics'] = 'Random semantic drift dominates'
        elif regime == 'high_mutation':
            interpretations['dynamics'] = 'Rapid semantic innovation and change'
        
        return interpretations
    
    def predict_dynamics(self, current_state: Dict[str, Any], 
                        time_horizon: int) -> Dict[str, Any]:
        """Predict future evolutionary dynamics"""
        # Get current parameters
        s = self.analogy_parameters['selection_strength']
        N = self.analogy_parameters['population_size']
        mu = self.analogy_parameters['mutation_rate']
        
        # Current population state
        current_diversity = current_state.get('diversity', 0.5)
        current_pop_size = current_state.get('population_size', N)
        
        # Predict using Wright-Fisher model
        predictions = self._simulate_wright_fisher(current_diversity, current_pop_size, s, mu, time_horizon)
        
        return {
            'diversity_trajectory': predictions['diversity'],
            'population_size_trajectory': predictions['population_size'],
            'selection_events': predictions['selection_events'],
            'extinction_risk': predictions['extinction_risk'],
            'adaptive_potential': predictions['adaptive_potential'],
            'long_term_equilibrium': predictions['equilibrium_state']
        }
    
    def _simulate_wright_fisher(self, initial_diversity: float, initial_pop_size: int,
                               selection_strength: float, mutation_rate: float,
                               time_horizon: int) -> Dict[str, Any]:
        """Simulate Wright-Fisher evolutionary dynamics"""
        diversity_trajectory = [initial_diversity]
        population_trajectory = [initial_pop_size]
        selection_events = []
        
        for t in range(time_horizon):
            current_diversity = diversity_trajectory[-1]
            current_pop = population_trajectory[-1]
            
            # Selection effect
            diversity_change = selection_strength * current_diversity * (1 - current_diversity)
            
            # Drift effect
            if current_pop > 0:
                drift_variance = current_diversity * (1 - current_diversity) / current_pop
                drift_change = np.random.normal(0, np.sqrt(drift_variance))
            else:
                drift_change = 0
            
            # Mutation effect
            mutation_change = mutation_rate * (0.5 - current_diversity)
            
            # Total change
            total_change = diversity_change + drift_change + mutation_change
            new_diversity = max(0.0, min(1.0, current_diversity + total_change))
            
            # Population size change (simplified)
            pop_change = np.random.normal(0, 0.1 * current_pop)
            new_pop = max(10, int(current_pop + pop_change))
            
            diversity_trajectory.append(new_diversity)
            population_trajectory.append(new_pop)
            
            # Detect selection events
            if abs(diversity_change) > 0.1:
                selection_events.append({
                    'time': t,
                    'type': 'positive_selection' if diversity_change > 0 else 'negative_selection',
                    'strength': abs(diversity_change)
                })
        
        # Assess extinction risk
        extinction_risk = 1.0 - np.mean(diversity_trajectory[-5:])
        
        # Assess adaptive potential
        adaptive_potential = np.mean(diversity_trajectory[-5:])
        
        # Determine equilibrium state
        if diversity_trajectory[-1] > 0.7:
            equilibrium_state = 'high_diversity'
        elif diversity_trajectory[-1] > 0.3:
            equilibrium_state = 'moderate_diversity'
        else:
            equilibrium_state = 'low_diversity'
        
        return {
            'diversity': diversity_trajectory,
            'population_size': population_trajectory,
            'selection_events': selection_events,
            'extinction_risk': extinction_risk,
            'adaptive_potential': adaptive_potential,
            'equilibrium_state': equilibrium_state
        }
    
    def compute_analogy_metrics(self, flows: List[SemanticFlow]) -> Dict[str, float]:
        """Compute evolutionary-specific metrics"""
        if not flows:
            return {}
        
        metrics = {}
        
        # Diversity metrics
        if self.semantic_populations:
            diversity_values = []
            for pop_state in self.semantic_populations.values():
                shannon_div = pop_state['diversity_metrics'].get('shannon_diversity', 0)
                diversity_values.append(shannon_div)
            
            if diversity_values:
                metrics['mean_diversity'] = np.mean(diversity_values)
                metrics['diversity_variance'] = np.var(diversity_values)
        
        # Evolutionary rates
        if self.evolutionary_trajectories:
            birth_rate = 0
            death_rate = 0
            total_lineages = len(self.evolutionary_trajectories)
            
            for lineage in self.evolutionary_trajectories.values():
                if lineage['death_time'] is not None:
                    death_rate += 1
                    
                if len(lineage['evolutionary_events']) > 0:
                    birth_events = [e for e in lineage['evolutionary_events'] if e['type'] == 'birth']
                    birth_rate += len(birth_events)
            
            if total_lineages > 0:
                metrics['birth_rate'] = birth_rate / total_lineages
                metrics['death_rate'] = death_rate / total_lineages
                metrics['turnover_rate'] = (birth_rate + death_rate) / total_lineages
        
        # Selection strength
        if self.fitness_landscapes:
            all_fitness = []
            for fitness_landscape in self.fitness_landscapes.values():
                all_fitness.extend(fitness_landscape.values())
            
            if all_fitness:
                metrics['mean_fitness'] = np.mean(all_fitness)
                metrics['fitness_variance'] = np.var(all_fitness)
                metrics['selection_differential'] = max(all_fitness) - min(all_fitness)
        
        # Phylogenetic metrics
        phylo_analysis = self._analyze_phylogenetics({})
        metrics['phylogenetic_diversity'] = phylo_analysis['phylogenetic_diversity']
        
        return metrics
    
    def interpret_results(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret evolutionary model results"""
        interpretation = model_results.get('interpretation', {})
        
        # Add evolutionary-specific interpretations
        parameters = model_results.get('parameters', {})
        regime = model_results.get('regime', 'unknown')
        
        semantic_interpretation = {
            'evolutionary_analogy': 'Semantic change follows evolutionary principles of variation, selection, and inheritance',
            'population_genetics': 'Vocabulary evolves like a population of genetic variants',
            'fitness_landscapes': 'Semantic fitness determines survival and reproduction of word meanings',
            'selection_vs_drift': 'Interplay between adaptive selection and random drift shapes evolution',
            'phylogenetic_relationships': 'Semantic lineages form branching evolutionary trees',
            'adaptive_evolution': 'Beneficial semantic changes spread through population'
        }
        
        semantic_interpretation.update(interpretation)
        return semantic_interpretation