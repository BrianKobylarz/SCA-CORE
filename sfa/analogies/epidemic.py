"""Epidemic dynamics analogy for semantic flow analysis."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy.integrate import odeint
from scipy.optimize import minimize
import networkx as nx

from ..core.types import Word, Timestamp, SemanticFlow
from .base_analogy import BaseAnalogy

class EpidemicAnalogy(BaseAnalogy):
    """Models semantic spread using epidemic dynamics (SIR, SEIR models)"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("EpidemicAnalogy", embeddings_store, config)
        
        # Epidemic model parameters
        self.analogy_parameters = {
            'transmission_rate': 0.1,    # β: rate of infection spread
            'recovery_rate': 0.05,       # γ: rate of recovery/forgetting
            'incubation_rate': 0.2,      # σ: rate of becoming infectious (SEIR)
            'birth_rate': 0.01,          # μ: rate of new susceptible individuals
            'death_rate': 0.01,          # δ: rate of removal from system
            'basic_reproduction_number': 2.0,  # R₀ = β/γ
            'herd_immunity_threshold': 0.5,    # 1 - 1/R₀
            'model_type': 'SIR'          # SIR, SEIR, or SIRS
        }
        
        # State variables
        self.population_states = {}  # timestamp -> {S, I, R, E}
        self.semantic_networks = {}  # timestamp -> network
        
    def fit_model(self, flows: List[SemanticFlow]) -> Dict[str, Any]:
        """Fit epidemic model to semantic flows"""
        if not flows:
            return {'parameters': self.analogy_parameters, 'fit_quality': 0.0}
        
        # Organize flows by timestamp
        flow_timeline = self._organize_flows_by_time(flows)
        
        # Build semantic networks for each timestamp
        self._build_semantic_networks(flow_timeline)
        
        # Estimate population states
        self._estimate_population_states(flow_timeline)
        
        # Fit epidemic parameters
        fitted_parameters = self._fit_epidemic_parameters(flow_timeline)
        
        # Compute model predictions
        predictions = self._compute_model_predictions(fitted_parameters)
        
        # Assess fit quality
        fit_quality = self._assess_fit_quality(predictions, flow_timeline)
        
        return {
            'parameters': fitted_parameters,
            'population_states': self.population_states,
            'predictions': predictions,
            'fit_quality': fit_quality,
            'regime': self._determine_epidemic_regime(fitted_parameters),
            'critical_points': self._identify_critical_points(fitted_parameters),
            'interpretation': self._interpret_epidemic_dynamics(fitted_parameters)
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
    
    def _build_semantic_networks(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> None:
        """Build semantic networks for epidemic modeling"""
        from scipy.spatial.distance import cosine
        
        for timestamp, flows in flow_timeline.items():
            G = nx.Graph()
            
            # Get vocabulary and embeddings
            vocabulary = list(self.embeddings.get_vocabulary(timestamp))
            
            # Add nodes
            G.add_nodes_from(vocabulary)
            
            # Add edges based on semantic similarity
            threshold = 0.3
            for i, w1 in enumerate(vocabulary):
                emb1 = self.embeddings.get_embedding(w1, timestamp)
                if emb1 is None:
                    continue
                
                for w2 in vocabulary[i+1:]:
                    emb2 = self.embeddings.get_embedding(w2, timestamp)
                    if emb2 is not None:
                        similarity = 1 - cosine(emb1, emb2)
                        if similarity > threshold:
                            G.add_edge(w1, w2, weight=similarity)
            
            self.semantic_networks[timestamp] = G
    
    def _estimate_population_states(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> None:
        """Estimate SIR population states from semantic data"""
        for timestamp, flows in flow_timeline.items():
            network = self.semantic_networks.get(timestamp)
            if not network:
                continue
            
            total_population = network.number_of_nodes()
            if total_population == 0:
                continue
            
            # Estimate infected: words with high flow activity
            high_flow_words = set()
            for flow in flows:
                if flow.magnitude > np.percentile([f.magnitude for f in flows], 80):
                    high_flow_words.add(flow.source_word)
                    high_flow_words.add(flow.target_word)
            
            infected_count = len(high_flow_words)
            
            # Estimate susceptible: words with low activity but high connectivity
            susceptible_count = 0
            for node in network.nodes():
                if node not in high_flow_words:
                    degree = network.degree(node)
                    if degree > np.mean([network.degree(n) for n in network.nodes()]):
                        susceptible_count += 1
            
            # Estimate recovered: remaining words
            recovered_count = total_population - infected_count - susceptible_count
            
            # Normalize to fractions
            self.population_states[timestamp] = {
                'S': susceptible_count / total_population,
                'I': infected_count / total_population,
                'R': max(0, recovered_count / total_population),
                'total_population': total_population
            }
    
    def _fit_epidemic_parameters(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> Dict[str, Any]:
        """Fit epidemic parameters using optimization"""
        timestamps = sorted(flow_timeline.keys())
        
        if len(timestamps) < 3:
            return self.analogy_parameters
        
        # Extract time series data
        time_series = []
        for timestamp in timestamps:
            states = self.population_states.get(timestamp, {'S': 0, 'I': 0, 'R': 0})
            time_series.append([states['S'], states['I'], states['R']])
        
        time_series = np.array(time_series)
        
        # Objective function for parameter fitting
        def objective(params):
            beta, gamma = params
            
            # Solve SIR model
            def sir_model(y, t):
                S, I, R = y
                dSdt = -beta * S * I
                dIdt = beta * S * I - gamma * I
                dRdt = gamma * I
                return [dSdt, dIdt, dRdt]
            
            # Initial conditions
            y0 = time_series[0]
            t = np.arange(len(timestamps))
            
            try:
                solution = odeint(sir_model, y0, t)
                # Compute MSE
                mse = np.mean((solution - time_series) ** 2)
                return mse
            except:
                return 1e6  # Large penalty for invalid parameters
        
        # Optimize parameters
        try:
            result = minimize(objective, [0.1, 0.05], 
                            bounds=[(0.01, 1.0), (0.01, 1.0)],
                            method='L-BFGS-B')
            
            if result.success:
                fitted_params = self.analogy_parameters.copy()
                fitted_params['transmission_rate'] = result.x[0]
                fitted_params['recovery_rate'] = result.x[1]
                fitted_params['basic_reproduction_number'] = result.x[0] / result.x[1]
                fitted_params['herd_immunity_threshold'] = 1 - 1/fitted_params['basic_reproduction_number']
                return fitted_params
        except:
            pass
        
        return self.analogy_parameters
    
    def _compute_model_predictions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute epidemic model predictions"""
        timestamps = sorted(self.population_states.keys())
        
        if len(timestamps) < 2:
            return {}
        
        # Get parameters
        beta = parameters['transmission_rate']
        gamma = parameters['recovery_rate']
        
        # SIR model
        def sir_model(y, t):
            S, I, R = y
            dSdt = -beta * S * I
            dIdt = beta * S * I - gamma * I
            dRdt = gamma * I
            return [dSdt, dIdt, dRdt]
        
        # Initial conditions
        initial_states = self.population_states[timestamps[0]]
        y0 = [initial_states['S'], initial_states['I'], initial_states['R']]
        
        # Time points
        t = np.arange(len(timestamps))
        
        try:
            solution = odeint(sir_model, y0, t)
            
            predictions = {}
            for i, timestamp in enumerate(timestamps):
                predictions[timestamp] = {
                    'S': solution[i, 0],
                    'I': solution[i, 1],
                    'R': solution[i, 2]
                }
            
            # Predict future states
            future_t = np.arange(len(timestamps), len(timestamps) + 5)
            future_solution = odeint(sir_model, solution[-1], future_t)
            
            return {
                'historical_predictions': predictions,
                'future_predictions': future_solution,
                'peak_infection_time': self._find_peak_infection(solution),
                'final_outbreak_size': solution[-1, 2],  # Final R
                'predicted_values': solution[:, 1]  # Infected fraction for validation
            }
        except:
            return {}
    
    def _find_peak_infection(self, solution: np.ndarray) -> int:
        """Find time of peak infection"""
        infected_series = solution[:, 1]  # I column
        return np.argmax(infected_series)
    
    def _assess_fit_quality(self, predictions: Dict[str, Any], 
                           flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> float:
        """Assess quality of epidemic model fit"""
        if not predictions or 'historical_predictions' not in predictions:
            return 0.0
        
        # Compare predicted vs actual infected fractions
        predicted_infected = []
        actual_infected = []
        
        for timestamp in sorted(flow_timeline.keys()):
            if timestamp in predictions['historical_predictions']:
                predicted_infected.append(predictions['historical_predictions'][timestamp]['I'])
                actual_infected.append(self.population_states[timestamp]['I'])
        
        if len(predicted_infected) < 2:
            return 0.0
        
        # Compute correlation
        correlation = np.corrcoef(predicted_infected, actual_infected)[0, 1]
        return max(0.0, correlation if not np.isnan(correlation) else 0.0)
    
    def _determine_epidemic_regime(self, parameters: Dict[str, Any]) -> str:
        """Determine epidemic regime based on R₀"""
        r0 = parameters['basic_reproduction_number']
        
        if r0 < 1.0:
            return 'sub-critical'
        elif r0 > 1.0 and r0 < 2.0:
            return 'critical'
        elif r0 >= 2.0 and r0 < 5.0:
            return 'super-critical'
        else:
            return 'explosive'
    
    def _identify_critical_points(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical points in epidemic dynamics"""
        critical_points = []
        
        r0 = parameters['basic_reproduction_number']
        
        # Epidemic threshold
        if abs(r0 - 1.0) < 0.1:
            critical_points.append({
                'type': 'epidemic_threshold',
                'r0': r0,
                'description': 'Near epidemic threshold (R₀ ≈ 1)'
            })
        
        # Herd immunity threshold
        herd_threshold = parameters['herd_immunity_threshold']
        critical_points.append({
            'type': 'herd_immunity_threshold',
            'threshold': herd_threshold,
            'description': f'Herd immunity at {herd_threshold:.2f} population fraction'
        })
        
        return critical_points
    
    def _interpret_epidemic_dynamics(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Interpret epidemic dynamics in semantic terms"""
        r0 = parameters['basic_reproduction_number']
        regime = self._determine_epidemic_regime(parameters)
        
        interpretations = {
            'basic_reproduction_number': f'Each semantic innovation spreads to {r0:.2f} other concepts on average',
            'regime': f'System is in {regime} regime',
            'transmission_mechanism': 'Semantic spread through similarity networks',
            'recovery_mechanism': 'Concepts return to baseline usage patterns',
            'herd_immunity': f'Semantic saturation at {parameters["herd_immunity_threshold"]:.2f} population fraction'
        }
        
        if regime == 'sub-critical':
            interpretations['dynamics'] = 'Semantic innovations die out quickly'
        elif regime == 'critical':
            interpretations['dynamics'] = 'Semantic innovations spread slowly but persistently'
        elif regime == 'super-critical':
            interpretations['dynamics'] = 'Semantic innovations spread rapidly through population'
        else:
            interpretations['dynamics'] = 'Explosive semantic change events'
        
        return interpretations
    
    def predict_dynamics(self, current_state: Dict[str, Any], 
                        time_horizon: int) -> Dict[str, Any]:
        """Predict future semantic dynamics using epidemic model"""
        # Get current parameters
        beta = self.analogy_parameters['transmission_rate']
        gamma = self.analogy_parameters['recovery_rate']
        
        # Current state
        S0 = current_state.get('susceptible_fraction', 0.7)
        I0 = current_state.get('infected_fraction', 0.2)
        R0 = current_state.get('recovered_fraction', 0.1)
        
        # SIR model
        def sir_model(y, t):
            S, I, R = y
            dSdt = -beta * S * I
            dIdt = beta * S * I - gamma * I
            dRdt = gamma * I
            return [dSdt, dIdt, dRdt]
        
        # Predict future
        t = np.arange(0, time_horizon)
        y0 = [S0, I0, R0]
        
        try:
            solution = odeint(sir_model, y0, t)
            
            return {
                'predicted_trajectory': {
                    'susceptible': solution[:, 0],
                    'infected': solution[:, 1],
                    'recovered': solution[:, 2]
                },
                'peak_infection_time': self._find_peak_infection(solution),
                'final_outbreak_size': solution[-1, 2],
                'extinction_time': self._find_extinction_time(solution),
                'intervention_recommendations': self._suggest_interventions(solution)
            }
        except:
            return {}
    
    def _find_extinction_time(self, solution: np.ndarray) -> int:
        """Find time when infection effectively dies out"""
        infected_series = solution[:, 1]
        extinction_threshold = 0.01
        
        for i, infected in enumerate(infected_series):
            if infected < extinction_threshold:
                return i
        
        return len(infected_series)
    
    def _suggest_interventions(self, solution: np.ndarray) -> List[str]:
        """Suggest interventions based on epidemic dynamics"""
        interventions = []
        
        peak_infection = np.max(solution[:, 1])
        peak_time = np.argmax(solution[:, 1])
        
        if peak_infection > 0.5:
            interventions.append("High peak infection - consider reducing transmission rate")
        
        if peak_time < 5:
            interventions.append("Early peak - epidemic spreads rapidly")
        
        final_size = solution[-1, 2]
        if final_size > 0.8:
            interventions.append("Large final outbreak size - consider preventive measures")
        
        return interventions
    
    def compute_analogy_metrics(self, flows: List[SemanticFlow]) -> Dict[str, float]:
        """Compute epidemic-specific metrics"""
        if not flows:
            return {}
        
        # Organize flows
        flow_timeline = self._organize_flows_by_time(flows)
        
        # Compute metrics
        metrics = {}
        
        # Attack rate (final outbreak size)
        if self.population_states:
            final_recovered = max(state['R'] for state in self.population_states.values())
            metrics['attack_rate'] = final_recovered
        
        # Epidemic duration
        infected_timeline = [state['I'] for state in self.population_states.values()]
        if infected_timeline:
            threshold = 0.01
            duration = len([i for i in infected_timeline if i > threshold])
            metrics['epidemic_duration'] = duration
        
        # Peak infection rate
        if infected_timeline:
            metrics['peak_infection_rate'] = max(infected_timeline)
        
        # Doubling time (early exponential growth)
        doubling_time = self._compute_doubling_time(infected_timeline)
        if doubling_time:
            metrics['doubling_time'] = doubling_time
        
        # Effective reproduction number
        r_eff = self._compute_effective_reproduction_number()
        if r_eff:
            metrics['effective_reproduction_number'] = r_eff
        
        return metrics
    
    def _compute_doubling_time(self, infected_timeline: List[float]) -> Optional[float]:
        """Compute doubling time during exponential growth phase"""
        if len(infected_timeline) < 3:
            return None
        
        # Find exponential growth phase (first few time points)
        growth_phase = infected_timeline[:min(5, len(infected_timeline))]
        
        if len(growth_phase) < 2:
            return None
        
        # Compute growth rate
        log_infected = np.log(np.array(growth_phase) + 1e-10)
        t = np.arange(len(growth_phase))
        
        try:
            growth_rate = np.polyfit(t, log_infected, 1)[0]
            if growth_rate > 0:
                return np.log(2) / growth_rate
        except:
            pass
        
        return None
    
    def _compute_effective_reproduction_number(self) -> Optional[float]:
        """Compute effective reproduction number over time"""
        if not self.population_states:
            return None
        
        # R_eff = R₀ * S(t)
        r0 = self.analogy_parameters['basic_reproduction_number']
        
        # Average susceptible fraction
        susceptible_fractions = [state['S'] for state in self.population_states.values()]
        avg_susceptible = np.mean(susceptible_fractions)
        
        return r0 * avg_susceptible
    
    def interpret_results(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret epidemic model results"""
        interpretation = model_results.get('interpretation', {})
        
        # Add epidemic-specific interpretations
        r0 = model_results.get('parameters', {}).get('basic_reproduction_number', 1.0)
        regime = model_results.get('regime', 'unknown')
        
        semantic_interpretation = {
            'epidemic_analogy': 'Semantic innovations spread like infectious diseases',
            'r0_meaning': f'Each semantic innovation influences {r0:.2f} other concepts',
            'regime_meaning': f'System exhibits {regime} dynamics',
            'transmission_meaning': 'Ideas spread through semantic similarity networks',
            'recovery_meaning': 'Concepts return to baseline usage after peak activity',
            'population_meaning': 'Vocabulary serves as the population of "hosts"'
        }
        
        semantic_interpretation.update(interpretation)
        return semantic_interpretation