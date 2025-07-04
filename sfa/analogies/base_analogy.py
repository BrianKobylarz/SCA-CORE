"""Base class for theoretical analogies in semantic flow analysis."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..core.types import Word, Timestamp, SemanticFlow
from ..core.base import BaseAnalyzer

class BaseAnalogy(BaseAnalyzer, ABC):
    """Abstract base class for theoretical analogies"""
    
    def __init__(self, name: str, embeddings_store, config):
        super().__init__(name)
        self.embeddings = embeddings_store
        self.config = config
        self.analogy_parameters = {}
        
    @abstractmethod
    def fit_model(self, flows: List[SemanticFlow]) -> Dict[str, Any]:
        """Fit the analogy model to semantic flows"""
        pass
    
    @abstractmethod
    def predict_dynamics(self, current_state: Dict[str, Any], 
                        time_horizon: int) -> Dict[str, Any]:
        """Predict future dynamics using the analogy"""
        pass
    
    @abstractmethod
    def compute_analogy_metrics(self, flows: List[SemanticFlow]) -> Dict[str, float]:
        """Compute metrics specific to this analogy"""
        pass
    
    @abstractmethod
    def interpret_results(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret model results in terms of the analogy"""
        pass
    
    def validate_analogy_fit(self, flows: List[SemanticFlow]) -> Dict[str, float]:
        """Validate how well the analogy fits the data"""
        if not flows:
            return {'goodness_of_fit': 0.0, 'confidence': 0.0}
        
        # Compute model predictions
        model_results = self.fit_model(flows)
        
        # Basic validation metrics
        validation_metrics = {
            'goodness_of_fit': self._compute_goodness_of_fit(flows, model_results),
            'confidence': self._compute_confidence(model_results),
            'model_complexity': self._compute_model_complexity(model_results),
            'interpretability': self._compute_interpretability_score(model_results)
        }
        
        return validation_metrics
    
    def _compute_goodness_of_fit(self, flows: List[SemanticFlow], 
                               model_results: Dict[str, Any]) -> float:
        """Compute goodness of fit metric"""
        # Default implementation based on prediction accuracy
        try:
            predicted_values = model_results.get('predicted_values', [])
            actual_values = [flow.magnitude for flow in flows]
            
            if len(predicted_values) == len(actual_values) and len(actual_values) > 0:
                mse = np.mean((np.array(predicted_values) - np.array(actual_values)) ** 2)
                variance = np.var(actual_values)
                
                if variance > 0:
                    r_squared = 1 - (mse / variance)
                    return max(0.0, min(1.0, r_squared))
            
            return 0.5  # Neutral score if can't compute
        except Exception:
            return 0.0
    
    def _compute_confidence(self, model_results: Dict[str, Any]) -> float:
        """Compute confidence in model results"""
        # Default implementation based on model stability
        try:
            # Look for uncertainty indicators in results
            uncertainty_indicators = [
                'parameter_uncertainty',
                'prediction_intervals',
                'bootstrap_confidence',
                'model_stability'
            ]
            
            confidence_scores = []
            for indicator in uncertainty_indicators:
                if indicator in model_results:
                    # Convert uncertainty to confidence
                    uncertainty = model_results[indicator]
                    if isinstance(uncertainty, (int, float)):
                        confidence_scores.append(1.0 - min(1.0, max(0.0, uncertainty)))
            
            if confidence_scores:
                return np.mean(confidence_scores)
            else:
                return 0.7  # Default moderate confidence
        except Exception:
            return 0.5
    
    def _compute_model_complexity(self, model_results: Dict[str, Any]) -> float:
        """Compute model complexity score"""
        # Default implementation
        try:
            # Count parameters
            parameter_count = len(model_results.get('parameters', {}))
            
            # Normalize to 0-1 scale (assuming max ~20 parameters)
            complexity = min(1.0, parameter_count / 20.0)
            
            return complexity
        except Exception:
            return 0.5
    
    def _compute_interpretability_score(self, model_results: Dict[str, Any]) -> float:
        """Compute interpretability score"""
        # Default implementation based on available interpretations
        try:
            interpretation = model_results.get('interpretation', {})
            
            if not interpretation:
                return 0.3  # Low interpretability
            
            # Count interpretable components
            interpretable_components = [
                'parameter_meanings',
                'mechanism_description',
                'predictions_explanation',
                'analogical_mapping'
            ]
            
            available_components = sum(1 for comp in interpretable_components 
                                     if comp in interpretation)
            
            return available_components / len(interpretable_components)
        except Exception:
            return 0.3
    
    def analyze(self) -> Dict[str, Any]:
        """Standard analyze method for BaseAnalyzer compatibility"""
        # Collect all flows for this analogy
        flows = self._collect_flows()
        
        # Fit model
        model_results = self.fit_model(flows)
        
        # Validate fit
        validation = self.validate_analogy_fit(flows)
        
        # Compute metrics
        metrics = self.compute_analogy_metrics(flows)
        
        # Interpret results
        interpretation = self.interpret_results(model_results)
        
        return {
            'analogy_name': self.name,
            'model_results': model_results,
            'validation_metrics': validation,
            'analogy_metrics': metrics,
            'interpretation': interpretation,
            'summary': self._generate_summary(model_results, validation, metrics)
        }
    
    def _collect_flows(self) -> List[SemanticFlow]:
        """Collect semantic flows for analysis"""
        # This would typically be implemented by the specific analogy
        # For now, return empty list
        return []
    
    def _generate_summary(self, model_results: Dict[str, Any],
                         validation: Dict[str, float],
                         metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary of analogy analysis"""
        return {
            'analogy_applicability': validation.get('goodness_of_fit', 0.0),
            'model_confidence': validation.get('confidence', 0.0),
            'key_insights': self._extract_key_insights(model_results),
            'predictive_power': metrics.get('predictive_accuracy', 0.0),
            'theoretical_significance': self._assess_theoretical_significance(model_results)
        }
    
    def _extract_key_insights(self, model_results: Dict[str, Any]) -> List[str]:
        """Extract key insights from model results"""
        insights = []
        
        # Look for significant parameters
        parameters = model_results.get('parameters', {})
        for param_name, param_value in parameters.items():
            if isinstance(param_value, (int, float)):
                if param_value > 1.0:
                    insights.append(f"High {param_name}: {param_value:.3f}")
                elif param_value < 0.1:
                    insights.append(f"Low {param_name}: {param_value:.3f}")
        
        # Look for regime information
        regime = model_results.get('regime', '')
        if regime:
            insights.append(f"System in {regime} regime")
        
        return insights
    
    def _assess_theoretical_significance(self, model_results: Dict[str, Any]) -> float:
        """Assess theoretical significance of results"""
        # Default implementation
        significance_indicators = [
            'critical_points',
            'phase_transitions',
            'universal_behaviors',
            'emergent_properties'
        ]
        
        significance_count = sum(1 for indicator in significance_indicators
                               if indicator in model_results and model_results[indicator])
        
        return significance_count / len(significance_indicators)
    
    def get_analogy_parameters(self) -> Dict[str, Any]:
        """Get current analogy parameters"""
        return self.analogy_parameters.copy()
    
    def set_analogy_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set analogy parameters"""
        self.analogy_parameters.update(parameters)
    
    def export_model(self) -> Dict[str, Any]:
        """Export model for external use"""
        return {
            'analogy_name': self.name,
            'parameters': self.analogy_parameters,
            'config': self.config,
            'model_type': self.__class__.__name__
        }
    
    def load_model(self, model_data: Dict[str, Any]) -> None:
        """Load model from external data"""
        if 'parameters' in model_data:
            self.analogy_parameters = model_data['parameters']
        if 'config' in model_data:
            self.config.update(model_data['config'])