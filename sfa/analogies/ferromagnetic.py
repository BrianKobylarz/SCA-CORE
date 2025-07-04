"""Ferromagnetic phase transition analogy for semantic flow analysis."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy.optimize import minimize, curve_fit
import networkx as nx

from ..core.types import Word, Timestamp, SemanticFlow
from .base_analogy import BaseAnalogy

class FerromagneticAnalogy(BaseAnalogy):
    """Models semantic coherence using ferromagnetic phase transitions"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("FerromagneticAnalogy", embeddings_store, config)
        
        # Ferromagnetic model parameters
        self.analogy_parameters = {
            'coupling_strength': 1.0,      # J: interaction strength between spins
            'external_field': 0.0,         # H: external magnetic field
            'temperature': 1.0,             # T: thermal energy/noise
            'critical_temperature': 2.27,   # T_c: Curie temperature (2D Ising)
            'magnetization': 0.0,           # M: order parameter
            'susceptibility': 1.0,          # χ: response to external field
            'correlation_length': 1.0,      # ξ: correlation length
            'beta_critical': 0.125,         # β: critical exponent for magnetization
            'gamma_critical': 1.75,         # γ: critical exponent for susceptibility
            'nu_critical': 1.0,             # ν: critical exponent for correlation length
            'model_type': 'ising_2d'        # ising_2d, ising_3d, xy_model, etc.
        }
        
        # Spin configurations
        self.spin_configurations = {}  # timestamp -> spin configuration
        self.magnetization_timeline = {}  # timestamp -> magnetization
        self.energy_timeline = {}  # timestamp -> energy
        
    def fit_model(self, flows: List[SemanticFlow]) -> Dict[str, Any]:
        """Fit ferromagnetic model to semantic flows"""
        if not flows:
            return {'parameters': self.analogy_parameters, 'fit_quality': 0.0}
        
        # Organize flows by timestamp
        flow_timeline = self._organize_flows_by_time(flows)
        
        # Map semantic states to spin configurations
        self._map_semantic_to_spin_states(flow_timeline)
        
        # Compute order parameters
        self._compute_order_parameters()
        
        # Fit model parameters
        fitted_parameters = self._fit_magnetic_parameters()
        
        # Analyze phase transitions
        phase_analysis = self._analyze_phase_transitions(fitted_parameters)
        
        # Compute critical behavior
        critical_behavior = self._analyze_critical_behavior(fitted_parameters)
        
        return {
            'parameters': fitted_parameters,
            'spin_configurations': self.spin_configurations,
            'magnetization_timeline': self.magnetization_timeline,
            'energy_timeline': self.energy_timeline,
            'phase_analysis': phase_analysis,
            'critical_behavior': critical_behavior,
            'regime': self._determine_magnetic_regime(fitted_parameters),
            'phase_diagram': self._construct_phase_diagram(fitted_parameters),
            'interpretation': self._interpret_magnetic_dynamics(fitted_parameters)
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
    
    def _map_semantic_to_spin_states(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> None:
        """Map semantic states to spin configurations"""
        from scipy.spatial.distance import cosine
        
        for timestamp, flows in flow_timeline.items():
            # Get vocabulary
            vocabulary = list(self.embeddings.get_vocabulary(timestamp))
            
            if len(vocabulary) < 2:
                continue
            
            # Create spin configuration
            spin_config = {}
            
            # Compute semantic coherence for each word
            for word in vocabulary:
                word_emb = self.embeddings.get_embedding(word, timestamp)
                if word_emb is None:
                    continue
                
                # Compute alignment with local semantic field
                local_coherence = self._compute_local_coherence(word, word_emb, vocabulary, timestamp)
                
                # Map coherence to spin (-1 to +1)
                # High coherence -> spin up (+1)
                # Low coherence -> spin down (-1)
                spin_config[word] = 2 * local_coherence - 1
            
            self.spin_configurations[timestamp] = spin_config
    
    def _compute_local_coherence(self, word: Word, word_emb: np.ndarray, 
                                vocabulary: List[Word], timestamp: Timestamp) -> float:
        """Compute local semantic coherence for a word"""
        from scipy.spatial.distance import cosine
        
        # Find semantic neighbors
        neighbors = []
        for other_word in vocabulary:
            if other_word != word:
                other_emb = self.embeddings.get_embedding(other_word, timestamp)
                if other_emb is not None:
                    similarity = 1 - cosine(word_emb, other_emb)
                    if similarity > 0.3:  # Threshold for semantic neighborhood
                        neighbors.append((other_word, similarity))
        
        if not neighbors:
            return 0.5  # Neutral coherence
        
        # Compute weighted average coherence
        total_coherence = 0
        total_weight = 0
        
        for neighbor_word, similarity in neighbors:
            # Coherence based on consistency of semantic direction
            neighbor_emb = self.embeddings.get_embedding(neighbor_word, timestamp)
            if neighbor_emb is not None:
                # Compute semantic alignment
                dot_product = np.dot(word_emb, neighbor_emb)
                norm_product = np.linalg.norm(word_emb) * np.linalg.norm(neighbor_emb)
                
                if norm_product > 0:
                    alignment = dot_product / norm_product
                    coherence = (alignment + 1) / 2  # Map to [0, 1]
                    
                    total_coherence += coherence * similarity
                    total_weight += similarity
        
        if total_weight > 0:
            return total_coherence / total_weight
        else:
            return 0.5
    
    def _compute_order_parameters(self) -> None:
        """Compute magnetic order parameters"""
        for timestamp, spin_config in self.spin_configurations.items():
            if not spin_config:
                continue
            
            spins = list(spin_config.values())
            
            # Magnetization (order parameter)
            magnetization = np.mean(spins)
            self.magnetization_timeline[timestamp] = magnetization
            
            # Energy (based on nearest neighbor interactions)
            energy = self._compute_magnetic_energy(spin_config, timestamp)
            self.energy_timeline[timestamp] = energy
    
    def _compute_magnetic_energy(self, spin_config: Dict[Word, float], timestamp: Timestamp) -> float:
        """Compute magnetic energy from spin configuration"""
        from scipy.spatial.distance import cosine
        
        # Hamiltonian: H = -J Σ s_i s_j - H Σ s_i
        
        J = self.analogy_parameters['coupling_strength']
        H = self.analogy_parameters['external_field']
        
        energy = 0.0
        
        # Interaction energy
        words = list(spin_config.keys())
        for i, word1 in enumerate(words):
            emb1 = self.embeddings.get_embedding(word1, timestamp)
            if emb1 is None:
                continue
            
            for word2 in words[i+1:]:
                emb2 = self.embeddings.get_embedding(word2, timestamp)
                if emb2 is not None:
                    # Coupling strength based on semantic similarity
                    similarity = 1 - cosine(emb1, emb2)
                    if similarity > 0.3:  # Only nearest neighbors
                        coupling = J * similarity
                        energy -= coupling * spin_config[word1] * spin_config[word2]
        
        # External field energy
        for word, spin in spin_config.items():
            energy -= H * spin
        
        return energy
    
    def _fit_magnetic_parameters(self) -> Dict[str, Any]:
        """Fit magnetic model parameters"""
        if not self.magnetization_timeline:
            return self.analogy_parameters
        
        timestamps = sorted(self.magnetization_timeline.keys())
        magnetizations = [self.magnetization_timeline[t] for t in timestamps]
        
        # Fit temperature from magnetization curve
        fitted_params = self.analogy_parameters.copy()
        
        # Estimate critical temperature
        critical_temp = self._estimate_critical_temperature(magnetizations)
        if critical_temp:
            fitted_params['critical_temperature'] = critical_temp
        
        # Estimate coupling strength from energy fluctuations
        if len(self.energy_timeline) > 1:
            energies = list(self.energy_timeline.values())
            energy_fluctuation = np.var(energies)
            # Coupling strength inversely related to energy fluctuation
            fitted_params['coupling_strength'] = 1.0 / (1.0 + energy_fluctuation)
        
        # Compute susceptibility
        susceptibility = self._compute_magnetic_susceptibility(magnetizations)
        fitted_params['susceptibility'] = susceptibility
        
        return fitted_params
    
    def _estimate_critical_temperature(self, magnetizations: List[float]) -> Optional[float]:
        """Estimate critical temperature from magnetization data"""
        if len(magnetizations) < 5:
            return None
        
        # Look for rapid change in magnetization (phase transition)
        derivatives = np.diff(magnetizations)
        
        if len(derivatives) < 2:
            return None
        
        # Find point of maximum derivative (steepest change)
        max_change_idx = np.argmax(np.abs(derivatives))
        
        # Estimate temperature scale
        # In real units, this would be calibrated
        estimated_temp = 2.0 + 0.5 * max_change_idx / len(magnetizations)
        
        return estimated_temp
    
    def _compute_magnetic_susceptibility(self, magnetizations: List[float]) -> float:
        """Compute magnetic susceptibility"""
        if len(magnetizations) < 2:
            return 1.0
        
        # Susceptibility ~ variance of magnetization
        susceptibility = np.var(magnetizations)
        
        # Normalize
        return min(10.0, max(0.1, susceptibility * 10))
    
    def _analyze_phase_transitions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze magnetic phase transitions"""
        T = parameters.get('temperature', 1.0)
        T_c = parameters.get('critical_temperature', 2.27)
        
        # Determine phase
        if T < T_c:
            phase = 'ferromagnetic'
            order = 'ordered'
        else:
            phase = 'paramagnetic'
            order = 'disordered'
        
        # Transition characteristics
        transition_analysis = {
            'current_phase': phase,
            'order_state': order,
            'temperature_ratio': T / T_c,
            'phase_transition_type': 'second_order',
            'critical_distance': abs(T - T_c) / T_c
        }
        
        # Identify phase transition signatures
        if len(self.magnetization_timeline) > 1:
            magnetizations = list(self.magnetization_timeline.values())
            
            # Look for hysteresis
            hysteresis = self._detect_hysteresis(magnetizations)
            transition_analysis['hysteresis'] = hysteresis
            
            # Look for critical opalescence (fluctuations)
            critical_fluctuations = self._detect_critical_fluctuations(magnetizations)
            transition_analysis['critical_fluctuations'] = critical_fluctuations
        
        return transition_analysis
    
    def _detect_hysteresis(self, magnetizations: List[float]) -> Dict[str, Any]:
        """Detect hysteresis in magnetization curve"""
        if len(magnetizations) < 5:
            return {'present': False, 'area': 0.0}
        
        # Simple hysteresis detection based on path dependence
        # This is a simplified implementation
        
        # Look for loops in magnetization
        reversals = 0
        for i in range(1, len(magnetizations) - 1):
            if ((magnetizations[i] > magnetizations[i-1] and magnetizations[i] > magnetizations[i+1]) or
                (magnetizations[i] < magnetizations[i-1] and magnetizations[i] < magnetizations[i+1])):
                reversals += 1
        
        hysteresis_strength = reversals / len(magnetizations)
        
        return {
            'present': hysteresis_strength > 0.1,
            'strength': hysteresis_strength,
            'area': np.std(magnetizations)  # Proxy for hysteresis area
        }
    
    def _detect_critical_fluctuations(self, magnetizations: List[float]) -> Dict[str, Any]:
        """Detect critical fluctuations near phase transition"""
        if len(magnetizations) < 3:
            return {'present': False, 'amplitude': 0.0}
        
        # Compute fluctuation metrics
        variance = np.var(magnetizations)
        mean_abs_derivative = np.mean(np.abs(np.diff(magnetizations)))
        
        # Critical fluctuations characterized by high variance and rapid changes
        fluctuation_amplitude = variance + mean_abs_derivative
        
        return {
            'present': fluctuation_amplitude > 0.1,
            'amplitude': fluctuation_amplitude,
            'variance': variance,
            'mean_change_rate': mean_abs_derivative
        }
    
    def _analyze_critical_behavior(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze critical behavior near phase transition"""
        T = parameters.get('temperature', 1.0)
        T_c = parameters.get('critical_temperature', 2.27)
        
        # Reduced temperature
        t = (T - T_c) / T_c
        
        critical_behavior = {
            'reduced_temperature': t,
            'critical_region': abs(t) < 0.1,
            'universality_class': '2D Ising'
        }
        
        # Compute critical exponents if near critical point
        if abs(t) < 0.2 and len(self.magnetization_timeline) > 3:
            exponents = self._compute_critical_exponents(t)
            critical_behavior['critical_exponents'] = exponents
        
        # Scaling behavior
        scaling_analysis = self._analyze_scaling_behavior(t)
        critical_behavior['scaling_behavior'] = scaling_analysis
        
        return critical_behavior
    
    def _compute_critical_exponents(self, reduced_temperature: float) -> Dict[str, float]:
        """Compute critical exponents from data"""
        exponents = {}
        
        # Magnetization exponent β
        if len(self.magnetization_timeline) > 2:
            magnetizations = list(self.magnetization_timeline.values())
            
            if reduced_temperature < 0:  # Below critical temperature
                # M ~ |t|^β
                nonzero_mag = [abs(m) for m in magnetizations if abs(m) > 0.01]
                if nonzero_mag:
                    # Fit power law (simplified)
                    log_mag = np.log(np.mean(nonzero_mag))
                    log_t = np.log(abs(reduced_temperature))
                    beta_exp = log_mag / log_t
                    exponents['beta'] = min(0.5, max(0.1, beta_exp))
        
        # Susceptibility exponent γ
        susceptibility = self.analogy_parameters['susceptibility']
        if abs(reduced_temperature) > 0.01:
            # χ ~ |t|^(-γ)
            log_chi = np.log(susceptibility)
            log_t = np.log(abs(reduced_temperature))
            gamma_exp = -log_chi / log_t
            exponents['gamma'] = min(3.0, max(0.5, gamma_exp))
        
        return exponents
    
    def _analyze_scaling_behavior(self, reduced_temperature: float) -> Dict[str, Any]:
        """Analyze scaling behavior near critical point"""
        scaling_behavior = {
            'scaling_region': abs(reduced_temperature) < 0.1,
            'finite_size_effects': 'minimal',  # Assuming large system
            'universality_class': '2D Ising'
        }
        
        # Data collapse analysis (simplified)
        if len(self.magnetization_timeline) > 5:
            magnetizations = list(self.magnetization_timeline.values())
            
            # Check for scaling collapse
            scaling_quality = self._assess_scaling_collapse(magnetizations, reduced_temperature)
            scaling_behavior['scaling_quality'] = scaling_quality
        
        return scaling_behavior
    
    def _assess_scaling_collapse(self, magnetizations: List[float], reduced_temperature: float) -> float:
        """Assess quality of scaling collapse"""
        # Simplified scaling collapse assessment
        # In practice, this would involve detailed data collapse analysis
        
        # Look for self-similarity in magnetization data
        if len(magnetizations) < 5:
            return 0.0
        
        # Compute correlation between different length scales
        half_length = len(magnetizations) // 2
        first_half = magnetizations[:half_length]
        second_half = magnetizations[half_length:half_length*2]
        
        if len(first_half) == len(second_half):
            correlation = np.corrcoef(first_half, second_half)[0, 1]
            return max(0.0, correlation if not np.isnan(correlation) else 0.0)
        
        return 0.0
    
    def _determine_magnetic_regime(self, parameters: Dict[str, Any]) -> str:
        """Determine magnetic regime"""
        T = parameters.get('temperature', 1.0)
        T_c = parameters.get('critical_temperature', 2.27)
        
        if T < T_c * 0.5:
            return 'deep_ferromagnetic'
        elif T < T_c:
            return 'ferromagnetic'
        elif T < T_c * 1.1:
            return 'critical'
        elif T < T_c * 2.0:
            return 'paramagnetic'
        else:
            return 'high_temperature'
    
    def _construct_phase_diagram(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Construct magnetic phase diagram"""
        T_c = parameters.get('critical_temperature', 2.27)
        
        # Temperature range
        T_range = np.linspace(0.1, 3.0, 100)
        
        # Theoretical magnetization curve
        magnetization_curve = []
        for T in T_range:
            if T < T_c:
                # Below critical temperature: M ~ (T_c - T)^β
                M = (T_c - T) ** 0.125  # β = 1/8 for 2D Ising
            else:
                M = 0.0
            magnetization_curve.append(M)
        
        return {
            'temperature_range': T_range.tolist(),
            'magnetization_curve': magnetization_curve,
            'critical_temperature': T_c,
            'phase_boundaries': [
                {'T': T_c, 'phase_transition': 'ferromagnetic_to_paramagnetic'}
            ]
        }
    
    def _interpret_magnetic_dynamics(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Interpret magnetic dynamics in semantic terms"""
        T = parameters.get('temperature', 1.0)
        T_c = parameters.get('critical_temperature', 2.27)
        regime = self._determine_magnetic_regime(parameters)
        
        interpretations = {
            'magnetic_analogy': 'Semantic coherence behaves like magnetic order',
            'spins_meaning': 'Words are like magnetic spins with semantic orientations',
            'coupling_meaning': 'Semantic similarity creates coupling between word-spins',
            'temperature_meaning': 'Noise and variability act like thermal fluctuations',
            'magnetization_meaning': 'Overall semantic coherence is the order parameter',
            'phase_meaning': f'System is in {regime} phase'
        }
        
        if regime == 'ferromagnetic':
            interpretations['dynamics'] = 'Strong semantic coherence and alignment'
        elif regime == 'paramagnetic':
            interpretations['dynamics'] = 'Disordered semantic states, weak coherence'
        elif regime == 'critical':
            interpretations['dynamics'] = 'Critical semantic fluctuations, phase transition'
        
        # Critical temperature interpretation
        interpretations['critical_temperature'] = f'Semantic coherence breaks down above T = {T_c:.2f}'
        
        return interpretations
    
    def predict_dynamics(self, current_state: Dict[str, Any], 
                        time_horizon: int) -> Dict[str, Any]:
        """Predict future magnetic dynamics"""
        # Get current parameters
        T = current_state.get('temperature', self.analogy_parameters['temperature'])
        T_c = self.analogy_parameters['critical_temperature']
        J = self.analogy_parameters['coupling_strength']
        
        # Current magnetization
        M0 = current_state.get('magnetization', 0.0)
        
        # Predict evolution using Langevin dynamics
        predictions = self._simulate_langevin_dynamics(M0, T, T_c, J, time_horizon)
        
        return {
            'magnetization_trajectory': predictions['magnetization'],
            'energy_trajectory': predictions['energy'],
            'phase_transitions': predictions['phase_transitions'],
            'critical_phenomena': predictions['critical_phenomena'],
            'long_term_behavior': predictions['long_term_state']
        }
    
    def _simulate_langevin_dynamics(self, M0: float, T: float, T_c: float, 
                                   J: float, time_horizon: int) -> Dict[str, Any]:
        """Simulate Langevin dynamics for magnetization"""
        # Simplified Langevin equation: dM/dt = -dF/dM + noise
        
        dt = 0.1
        noise_strength = np.sqrt(2 * T * dt)
        
        magnetization = [M0]
        energy = []
        
        for t in range(time_horizon):
            M = magnetization[-1]
            
            # Free energy derivative (mean field approximation)
            # F = -J*M^2/2 + T*M*log(M) for M > 0
            if M > 0.001:
                dF_dM = -J * M + T * (np.log(M) + 1)
            else:
                dF_dM = -J * M
            
            # Langevin equation
            noise = np.random.normal(0, noise_strength)
            dM_dt = -dF_dM + noise
            
            # Update magnetization
            M_new = M + dM_dt * dt
            
            # Keep magnetization in valid range
            M_new = max(-1.0, min(1.0, M_new))
            
            magnetization.append(M_new)
            
            # Compute energy
            E = -J * M_new**2 / 2
            energy.append(E)
        
        # Analyze trajectory
        phase_transitions = self._detect_phase_transitions_in_trajectory(magnetization)
        critical_phenomena = self._analyze_critical_phenomena_in_trajectory(magnetization, T, T_c)
        
        return {
            'magnetization': magnetization,
            'energy': energy,
            'phase_transitions': phase_transitions,
            'critical_phenomena': critical_phenomena,
            'long_term_state': self._assess_long_term_behavior(magnetization[-10:])
        }
    
    def _detect_phase_transitions_in_trajectory(self, magnetization: List[float]) -> List[Dict[str, Any]]:
        """Detect phase transitions in trajectory"""
        transitions = []
        
        # Look for sudden changes in magnetization
        for i in range(1, len(magnetization) - 1):
            change = abs(magnetization[i] - magnetization[i-1])
            if change > 0.3:  # Threshold for significant change
                transitions.append({
                    'time': i,
                    'type': 'magnetization_jump',
                    'magnitude': change,
                    'from_state': magnetization[i-1],
                    'to_state': magnetization[i]
                })
        
        return transitions
    
    def _analyze_critical_phenomena_in_trajectory(self, magnetization: List[float], 
                                                 T: float, T_c: float) -> Dict[str, Any]:
        """Analyze critical phenomena in trajectory"""
        critical_phenomena = {
            'near_critical_point': abs(T - T_c) < 0.1,
            'critical_fluctuations': np.var(magnetization),
            'correlation_time': self._compute_correlation_time(magnetization)
        }
        
        if critical_phenomena['near_critical_point']:
            critical_phenomena['diverging_quantities'] = [
                'susceptibility', 'correlation_length', 'specific_heat'
            ]
        
        return critical_phenomena
    
    def _compute_correlation_time(self, magnetization: List[float]) -> float:
        """Compute correlation time from magnetization trajectory"""
        if len(magnetization) < 10:
            return 1.0
        
        # Compute autocorrelation function
        M = np.array(magnetization)
        M_centered = M - np.mean(M)
        
        # Autocorrelation
        autocorr = np.correlate(M_centered, M_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 1:
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find correlation time (1/e decay)
            for i, corr in enumerate(autocorr):
                if corr < 1/np.e:
                    return float(i)
        
        return 1.0
    
    def _assess_long_term_behavior(self, recent_magnetization: List[float]) -> Dict[str, Any]:
        """Assess long-term behavior from recent trajectory"""
        if not recent_magnetization:
            return {}
        
        mean_M = np.mean(recent_magnetization)
        std_M = np.std(recent_magnetization)
        
        if std_M < 0.1:
            if abs(mean_M) > 0.5:
                state = 'ordered'
            else:
                state = 'disordered'
        else:
            state = 'fluctuating'
        
        return {
            'equilibrium_state': state,
            'mean_magnetization': mean_M,
            'fluctuation_amplitude': std_M,
            'stability': 'stable' if std_M < 0.2 else 'unstable'
        }
    
    def compute_analogy_metrics(self, flows: List[SemanticFlow]) -> Dict[str, float]:
        """Compute ferromagnetic-specific metrics"""
        if not flows:
            return {}
        
        metrics = {}
        
        # Order parameter statistics
        if self.magnetization_timeline:
            magnetizations = list(self.magnetization_timeline.values())
            metrics['mean_magnetization'] = np.mean(magnetizations)
            metrics['magnetization_variance'] = np.var(magnetizations)
            metrics['order_parameter_range'] = max(magnetizations) - min(magnetizations)
        
        # Energy statistics
        if self.energy_timeline:
            energies = list(self.energy_timeline.values())
            metrics['mean_energy'] = np.mean(energies)
            metrics['energy_variance'] = np.var(energies)
            metrics['specific_heat'] = np.var(energies) / self.analogy_parameters['temperature']
        
        # Susceptibility
        metrics['magnetic_susceptibility'] = self.analogy_parameters['susceptibility']
        
        # Phase coherence
        if self.magnetization_timeline:
            phase_coherence = self._compute_phase_coherence()
            metrics['phase_coherence'] = phase_coherence
        
        return metrics
    
    def _compute_phase_coherence(self) -> float:
        """Compute phase coherence metric"""
        if not self.magnetization_timeline:
            return 0.0
        
        magnetizations = list(self.magnetization_timeline.values())
        
        # Coherence based on consistency of magnetization
        if len(magnetizations) < 2:
            return 0.0
        
        # Compute temporal correlation
        correlations = []
        for i in range(len(magnetizations) - 1):
            corr = magnetizations[i] * magnetizations[i+1]
            correlations.append(corr)
        
        return np.mean(correlations)
    
    def interpret_results(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret ferromagnetic model results"""
        interpretation = model_results.get('interpretation', {})
        
        # Add ferromagnetic-specific interpretations
        parameters = model_results.get('parameters', {})
        regime = model_results.get('regime', 'unknown')
        
        semantic_interpretation = {
            'ferromagnetic_analogy': 'Semantic coherence exhibits magnetic-like phase behavior',
            'order_parameter': 'Magnetization represents overall semantic alignment',
            'coupling_strength': f'Semantic coupling strength: {parameters.get("coupling_strength", 1.0):.3f}',
            'temperature_effects': 'Noise and randomness act like thermal fluctuations',
            'phase_transitions': 'Sharp transitions between coherent and incoherent states',
            'critical_phenomena': 'Universal scaling behavior near semantic phase transitions'
        }
        
        semantic_interpretation.update(interpretation)
        return semantic_interpretation