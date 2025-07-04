"""Ensemble methods for combining multiple theoretical analogies."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from ..core.types import Word, Timestamp, SemanticFlow
from .base_analogy import BaseAnalogy
from .epidemic import EpidemicAnalogy
from .ferromagnetic import FerromagneticAnalogy
from .evolutionary import EvolutionaryAnalogy
from .bounded_confidence import BoundedConfidenceAnalogy

class AnalogyEnsemble(BaseAnalogy):
    """Ensemble of multiple theoretical analogies for comprehensive analysis"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("AnalogyEnsemble", embeddings_store, config)
        
        # Ensemble parameters
        self.analogy_parameters = {
            'ensemble_method': 'weighted_voting',  # weighted_voting, stacking, averaging
            'weight_by_performance': True,
            'cross_validation_folds': 3,
            'consensus_threshold': 0.6,
            'uncertainty_quantification': True,
            'model_selection_criterion': 'goodness_of_fit',
            'adaptive_weighting': True,
            'meta_learning': False
        }
        
        # Initialize individual analogies
        self.analogies = {
            'epidemic': EpidemicAnalogy(embeddings_store, config),
            'ferromagnetic': FerromagneticAnalogy(embeddings_store, config),
            'evolutionary': EvolutionaryAnalogy(embeddings_store, config),
            'bounded_confidence': BoundedConfidenceAnalogy(embeddings_store, config)
        }
        
        # Ensemble state
        self.analogy_weights = {}
        self.analogy_performance = {}
        self.ensemble_predictions = {}
        self.uncertainty_estimates = {}
        self.model_agreements = {}
        
    def fit_model(self, flows: List[SemanticFlow]) -> Dict[str, Any]:
        """Fit ensemble of analogies to semantic flows"""
        if not flows:
            return {'parameters': self.analogy_parameters, 'fit_quality': 0.0}
        
        # Fit individual analogies
        individual_results = self._fit_individual_analogies(flows)
        
        # Compute performance metrics for each analogy
        performance_metrics = self._compute_performance_metrics(individual_results, flows)
        
        # Determine ensemble weights
        ensemble_weights = self._compute_ensemble_weights(performance_metrics)
        
        # Generate ensemble predictions
        ensemble_predictions = self._generate_ensemble_predictions(individual_results, ensemble_weights)
        
        # Quantify uncertainty
        uncertainty_analysis = self._quantify_uncertainty(individual_results)
        
        # Analyze model agreements and disagreements
        agreement_analysis = self._analyze_model_agreements(individual_results)
        
        # Meta-analysis across analogies
        meta_analysis = self._perform_meta_analysis(individual_results, performance_metrics)
        
        return {
            'individual_results': individual_results,
            'performance_metrics': performance_metrics,
            'ensemble_weights': ensemble_weights,
            'ensemble_predictions': ensemble_predictions,
            'uncertainty_analysis': uncertainty_analysis,
            'agreement_analysis': agreement_analysis,
            'meta_analysis': meta_analysis,
            'regime': self._determine_ensemble_regime(individual_results),
            'consensus_insights': self._extract_consensus_insights(individual_results),
            'interpretation': self._interpret_ensemble_results(individual_results, ensemble_weights)
        }
    
    def _fit_individual_analogies(self, flows: List[SemanticFlow]) -> Dict[str, Dict[str, Any]]:
        """Fit each individual analogy"""
        individual_results = {}
        
        for analogy_name, analogy in self.analogies.items():
            try:
                result = analogy.fit_model(flows)
                individual_results[analogy_name] = result
            except Exception as e:
                # Handle analogy fitting failures gracefully
                individual_results[analogy_name] = {
                    'error': str(e),
                    'parameters': analogy.analogy_parameters,
                    'fit_quality': 0.0
                }
        
        return individual_results
    
    def _compute_performance_metrics(self, individual_results: Dict[str, Dict[str, Any]], 
                                   flows: List[SemanticFlow]) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics for each analogy"""
        performance_metrics = {}
        
        for analogy_name, result in individual_results.items():
            if 'error' in result:
                performance_metrics[analogy_name] = {
                    'goodness_of_fit': 0.0,
                    'interpretability': 0.0,
                    'predictive_power': 0.0,
                    'robustness': 0.0,
                    'overall_score': 0.0
                }
                continue
            
            # Validate analogy fit
            analogy = self.analogies[analogy_name]
            validation_metrics = analogy.validate_analogy_fit(flows)
            
            # Extract performance indicators
            goodness_of_fit = result.get('fit_quality', validation_metrics.get('goodness_of_fit', 0.0))
            interpretability = validation_metrics.get('interpretability', 0.0)
            
            # Estimate predictive power
            predictive_power = self._estimate_predictive_power(result, flows)
            
            # Estimate robustness
            robustness = self._estimate_robustness(result)
            
            # Compute overall score
            overall_score = (
                0.3 * goodness_of_fit +
                0.2 * interpretability +
                0.3 * predictive_power +
                0.2 * robustness
            )
            
            performance_metrics[analogy_name] = {
                'goodness_of_fit': goodness_of_fit,
                'interpretability': interpretability,
                'predictive_power': predictive_power,
                'robustness': robustness,
                'overall_score': overall_score
            }
        
        return performance_metrics
    
    def _estimate_predictive_power(self, result: Dict[str, Any], flows: List[SemanticFlow]) -> float:
        """Estimate predictive power of an analogy"""
        # Look for prediction-related metrics in results
        predictions = result.get('predictions', {})
        
        if not predictions:
            return 0.5  # Default moderate score
        
        # Check if predictions contain accuracy measures
        predicted_values = predictions.get('predicted_values', [])
        if predicted_values and len(predicted_values) > 0:
            # Compare with actual flow magnitudes
            actual_values = [flow.magnitude for flow in flows]
            
            if len(predicted_values) == len(actual_values):
                # Compute correlation as predictive power proxy
                if len(actual_values) > 1:
                    correlation = np.corrcoef(predicted_values, actual_values)[0, 1]
                    return max(0.0, correlation if not np.isnan(correlation) else 0.0)
        
        # Fallback: look for other prediction quality indicators
        prediction_quality_indicators = [
            'prediction_accuracy',
            'forecast_quality',
            'model_performance',
            'validation_score'
        ]
        
        for indicator in prediction_quality_indicators:
            if indicator in result:
                value = result[indicator]
                if isinstance(value, (int, float)):
                    return max(0.0, min(1.0, value))
        
        return 0.5
    
    def _estimate_robustness(self, result: Dict[str, Any]) -> float:
        """Estimate robustness of an analogy"""
        robustness_indicators = []
        
        # Parameter stability
        parameters = result.get('parameters', {})
        if parameters:
            # Check for reasonable parameter values (not extreme)
            param_values = [v for v in parameters.values() if isinstance(v, (int, float))]
            if param_values:
                # Robustness inversely related to parameter extremeness
                param_scores = []
                for value in param_values:
                    if 0.01 <= abs(value) <= 100:  # Reasonable range
                        param_scores.append(1.0)
                    else:
                        param_scores.append(0.0)
                robustness_indicators.append(np.mean(param_scores))
        
        # Model complexity (simpler models often more robust)
        complexity_indicators = ['model_complexity', 'parameter_count']
        for indicator in complexity_indicators:
            if indicator in result:
                complexity = result[indicator]
                if isinstance(complexity, (int, float)):
                    # Robustness inversely related to complexity
                    robustness_indicators.append(1.0 / (1.0 + complexity))
        
        # Convergence indicators
        convergence_indicators = ['convergence', 'stability', 'convergence_analysis']
        for indicator in convergence_indicators:
            if indicator in result:
                convergence_info = result[indicator]
                if isinstance(convergence_info, dict):
                    if 'converged' in convergence_info:
                        robustness_indicators.append(1.0 if convergence_info['converged'] else 0.0)
                elif isinstance(convergence_info, (int, float)):
                    robustness_indicators.append(max(0.0, min(1.0, convergence_info)))
        
        if robustness_indicators:
            return np.mean(robustness_indicators)
        else:
            return 0.6  # Default moderate robustness
    
    def _compute_ensemble_weights(self, performance_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute weights for ensemble combination"""
        if self.analogy_parameters['weight_by_performance']:
            # Weight by overall performance
            total_score = sum(metrics['overall_score'] for metrics in performance_metrics.values())
            
            if total_score > 0:
                weights = {
                    analogy: metrics['overall_score'] / total_score
                    for analogy, metrics in performance_metrics.items()
                }
            else:
                # Equal weights if no valid scores
                n_analogies = len(performance_metrics)
                weights = {analogy: 1.0 / n_analogies for analogy in performance_metrics.keys()}
        else:
            # Equal weights
            n_analogies = len(performance_metrics)
            weights = {analogy: 1.0 / n_analogies for analogy in performance_metrics.keys()}
        
        # Store weights
        self.analogy_weights = weights
        
        return weights
    
    def _generate_ensemble_predictions(self, individual_results: Dict[str, Dict[str, Any]], 
                                     ensemble_weights: Dict[str, float]) -> Dict[str, Any]:
        """Generate ensemble predictions by combining individual predictions"""
        ensemble_predictions = {}
        
        # Collect all prediction types
        prediction_types = set()
        for result in individual_results.values():
            if 'predictions' in result and isinstance(result['predictions'], dict):
                prediction_types.update(result['predictions'].keys())
        
        # Combine predictions for each type
        for pred_type in prediction_types:
            predictions_for_type = {}
            weights_for_type = {}
            
            for analogy_name, result in individual_results.items():
                if ('predictions' in result and 
                    isinstance(result['predictions'], dict) and 
                    pred_type in result['predictions']):
                    
                    predictions_for_type[analogy_name] = result['predictions'][pred_type]
                    weights_for_type[analogy_name] = ensemble_weights.get(analogy_name, 0.0)
            
            # Combine predictions
            if predictions_for_type:
                combined_prediction = self._combine_predictions(predictions_for_type, weights_for_type)
                ensemble_predictions[pred_type] = combined_prediction
        
        # Generate meta-predictions
        meta_predictions = self._generate_meta_predictions(individual_results, ensemble_weights)
        ensemble_predictions.update(meta_predictions)
        
        return ensemble_predictions
    
    def _combine_predictions(self, predictions: Dict[str, Any], weights: Dict[str, float]) -> Any:
        """Combine predictions from multiple analogies"""
        # Handle different prediction types
        sample_prediction = next(iter(predictions.values()))
        
        if isinstance(sample_prediction, (int, float)):
            # Scalar predictions - weighted average
            total_weight = sum(weights.values())
            if total_weight > 0:
                weighted_sum = sum(weight * predictions[analogy] for analogy, weight in weights.items())
                return weighted_sum / total_weight
            else:
                return np.mean(list(predictions.values()))
        
        elif isinstance(sample_prediction, list):
            # List predictions - element-wise weighted average
            if all(isinstance(pred, list) and len(pred) == len(sample_prediction) 
                   for pred in predictions.values()):
                
                combined = []
                total_weight = sum(weights.values())
                
                for i in range(len(sample_prediction)):
                    if total_weight > 0:
                        weighted_sum = sum(weight * predictions[analogy][i] 
                                         for analogy, weight in weights.items())
                        combined.append(weighted_sum / total_weight)
                    else:
                        values_at_i = [predictions[analogy][i] for analogy in predictions.keys()]
                        combined.append(np.mean(values_at_i))
                
                return combined
        
        elif isinstance(sample_prediction, dict):
            # Dictionary predictions - combine recursively
            combined_dict = {}
            all_keys = set()
            for pred in predictions.values():
                if isinstance(pred, dict):
                    all_keys.update(pred.keys())
            
            for key in all_keys:
                key_predictions = {}
                key_weights = {}
                
                for analogy, pred in predictions.items():
                    if isinstance(pred, dict) and key in pred:
                        key_predictions[analogy] = pred[key]
                        key_weights[analogy] = weights.get(analogy, 0.0)
                
                if key_predictions:
                    combined_dict[key] = self._combine_predictions(key_predictions, key_weights)
            
            return combined_dict
        
        # Fallback: return most weighted prediction
        if weights:
            best_analogy = max(weights.keys(), key=lambda k: weights[k])
            return predictions[best_analogy]
        else:
            return sample_prediction
    
    def _generate_meta_predictions(self, individual_results: Dict[str, Dict[str, Any]], 
                                 ensemble_weights: Dict[str, float]) -> Dict[str, Any]:
        """Generate meta-predictions from ensemble analysis"""
        meta_predictions = {}
        
        # Regime consensus
        regimes = {}
        for analogy_name, result in individual_results.items():
            if 'regime' in result:
                regime = result['regime']
                weight = ensemble_weights.get(analogy_name, 0.0)
                if regime in regimes:
                    regimes[regime] += weight
                else:
                    regimes[regime] = weight
        
        if regimes:
            consensus_regime = max(regimes.keys(), key=lambda k: regimes[k])
            regime_confidence = regimes[consensus_regime] / sum(regimes.values())
            
            meta_predictions['ensemble_regime'] = {
                'regime': consensus_regime,
                'confidence': regime_confidence,
                'alternatives': regimes
            }
        
        # Critical point consensus
        critical_points = []
        for analogy_name, result in individual_results.items():
            if 'critical_points' in result:
                points = result['critical_points']
                weight = ensemble_weights.get(analogy_name, 0.0)
                
                if isinstance(points, list):
                    for point in points:
                        critical_points.append({
                            'analogy': analogy_name,
                            'point': point,
                            'weight': weight
                        })
        
        if critical_points:
            meta_predictions['critical_points_consensus'] = critical_points
        
        # Parameter consensus
        parameter_consensus = self._compute_parameter_consensus(individual_results, ensemble_weights)
        if parameter_consensus:
            meta_predictions['parameter_consensus'] = parameter_consensus
        
        return meta_predictions
    
    def _compute_parameter_consensus(self, individual_results: Dict[str, Dict[str, Any]], 
                                   ensemble_weights: Dict[str, float]) -> Dict[str, Any]:
        """Compute consensus on parameter estimates"""
        parameter_consensus = {}
        
        # Collect all parameter names
        all_parameters = set()
        for result in individual_results.values():
            if 'parameters' in result and isinstance(result['parameters'], dict):
                all_parameters.update(result['parameters'].keys())
        
        # For each parameter, compute weighted consensus
        for param_name in all_parameters:
            param_values = {}
            param_weights = {}
            
            for analogy_name, result in individual_results.items():
                if ('parameters' in result and 
                    isinstance(result['parameters'], dict) and 
                    param_name in result['parameters']):
                    
                    value = result['parameters'][param_name]
                    if isinstance(value, (int, float)):
                        param_values[analogy_name] = value
                        param_weights[analogy_name] = ensemble_weights.get(analogy_name, 0.0)
            
            if param_values:
                # Weighted average
                total_weight = sum(param_weights.values())
                if total_weight > 0:
                    weighted_mean = sum(weight * param_values[analogy] 
                                      for analogy, weight in param_weights.items()) / total_weight
                else:
                    weighted_mean = np.mean(list(param_values.values()))
                
                # Compute uncertainty
                values_list = list(param_values.values())
                uncertainty = np.std(values_list) if len(values_list) > 1 else 0.0
                
                parameter_consensus[param_name] = {
                    'consensus_value': weighted_mean,
                    'uncertainty': uncertainty,
                    'individual_values': param_values,
                    'agreement_level': 1.0 / (1.0 + uncertainty)
                }
        
        return parameter_consensus
    
    def _quantify_uncertainty(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Quantify uncertainty across ensemble"""
        uncertainty_analysis = {
            'model_disagreement': {},
            'prediction_variance': {},
            'confidence_intervals': {},
            'epistemic_uncertainty': 0.0,
            'aleatoric_uncertainty': 0.0
        }
        
        # Model disagreement analysis
        regimes = [result.get('regime', 'unknown') for result in individual_results.values()]
        unique_regimes = set(regimes)
        regime_disagreement = len(unique_regimes) / len(regimes) if regimes else 0.0
        
        uncertainty_analysis['model_disagreement']['regime_disagreement'] = regime_disagreement
        
        # Prediction variance
        prediction_types = set()
        for result in individual_results.values():
            if 'predictions' in result and isinstance(result['predictions'], dict):
                prediction_types.update(result['predictions'].keys())
        
        for pred_type in prediction_types:
            pred_values = []
            for result in individual_results.values():
                if ('predictions' in result and 
                    isinstance(result['predictions'], dict) and 
                    pred_type in result['predictions']):
                    
                    pred = result['predictions'][pred_type]
                    if isinstance(pred, (int, float)):
                        pred_values.append(pred)
            
            if len(pred_values) > 1:
                uncertainty_analysis['prediction_variance'][pred_type] = np.var(pred_values)
        
        # Epistemic uncertainty (model uncertainty)
        performance_scores = []
        for analogy_name in individual_results.keys():
            if analogy_name in self.analogy_weights:
                performance_scores.append(self.analogy_weights[analogy_name])
        
        if performance_scores:
            uncertainty_analysis['epistemic_uncertainty'] = 1.0 - max(performance_scores)
        
        return uncertainty_analysis
    
    def _analyze_model_agreements(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze agreements and disagreements between models"""
        agreement_analysis = {
            'regime_agreement': {},
            'prediction_correlations': {},
            'parameter_correlations': {},
            'consensus_strength': 0.0
        }
        
        # Regime agreement
        regimes = {}
        for analogy_name, result in individual_results.items():
            regime = result.get('regime', 'unknown')
            regimes[analogy_name] = regime
        
        regime_counts = {}
        for regime in regimes.values():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if regime_counts:
            max_agreement = max(regime_counts.values())
            total_models = len(regimes)
            agreement_analysis['regime_agreement'] = {
                'regime_distribution': regime_counts,
                'max_agreement_fraction': max_agreement / total_models,
                'consensus_regime': max(regime_counts.keys(), key=lambda k: regime_counts[k])
            }
        
        # Prediction correlations
        prediction_correlations = self._compute_prediction_correlations(individual_results)
        agreement_analysis['prediction_correlations'] = prediction_correlations
        
        # Overall consensus strength
        consensus_indicators = []
        
        # Regime consensus
        if 'regime_agreement' in agreement_analysis:
            consensus_indicators.append(agreement_analysis['regime_agreement']['max_agreement_fraction'])
        
        # Prediction consensus
        if prediction_correlations:
            correlation_values = []
            for correlations in prediction_correlations.values():
                if isinstance(correlations, dict):
                    correlation_values.extend([v for v in correlations.values() 
                                             if isinstance(v, (int, float)) and not np.isnan(v)])
            
            if correlation_values:
                consensus_indicators.append(np.mean(correlation_values))
        
        if consensus_indicators:
            agreement_analysis['consensus_strength'] = np.mean(consensus_indicators)
        
        return agreement_analysis
    
    def _compute_prediction_correlations(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict]:
        """Compute correlations between model predictions"""
        prediction_correlations = {}
        
        # Find common prediction types
        prediction_types = set()
        for result in individual_results.values():
            if 'predictions' in result and isinstance(result['predictions'], dict):
                prediction_types.update(result['predictions'].keys())
        
        for pred_type in prediction_types:
            # Collect predictions of this type
            predictions_by_model = {}
            for analogy_name, result in individual_results.items():
                if ('predictions' in result and 
                    isinstance(result['predictions'], dict) and 
                    pred_type in result['predictions']):
                    
                    pred = result['predictions'][pred_type]
                    if isinstance(pred, (list, np.ndarray)):
                        predictions_by_model[analogy_name] = np.array(pred)
                    elif isinstance(pred, (int, float)):
                        predictions_by_model[analogy_name] = np.array([pred])
            
            # Compute pairwise correlations
            if len(predictions_by_model) >= 2:
                correlations = {}
                model_names = list(predictions_by_model.keys())
                
                for i, model1 in enumerate(model_names):
                    for model2 in model_names[i+1:]:
                        pred1 = predictions_by_model[model1]
                        pred2 = predictions_by_model[model2]
                        
                        # Ensure same length
                        min_len = min(len(pred1), len(pred2))
                        if min_len > 1:
                            corr = np.corrcoef(pred1[:min_len], pred2[:min_len])[0, 1]
                            correlations[f"{model1}_vs_{model2}"] = corr if not np.isnan(corr) else 0.0
                
                if correlations:
                    prediction_correlations[pred_type] = correlations
        
        return prediction_correlations
    
    def _perform_meta_analysis(self, individual_results: Dict[str, Dict[str, Any]], 
                             performance_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform meta-analysis across analogies"""
        meta_analysis = {
            'best_performing_analogy': '',
            'complementary_analogies': [],
            'emergent_insights': [],
            'theoretical_synthesis': {},
            'limitations_analysis': {}
        }
        
        # Best performing analogy
        if performance_metrics:
            best_analogy = max(performance_metrics.keys(), 
                             key=lambda k: performance_metrics[k]['overall_score'])
            meta_analysis['best_performing_analogy'] = best_analogy
        
        # Identify complementary analogies
        complementary_pairs = self._identify_complementary_analogies(
            individual_results, performance_metrics
        )
        meta_analysis['complementary_analogies'] = complementary_pairs
        
        # Extract emergent insights
        emergent_insights = self._extract_emergent_insights(individual_results)
        meta_analysis['emergent_insights'] = emergent_insights
        
        # Theoretical synthesis
        theoretical_synthesis = self._synthesize_theoretical_insights(individual_results)
        meta_analysis['theoretical_synthesis'] = theoretical_synthesis
        
        # Limitations analysis
        limitations = self._analyze_limitations(individual_results, performance_metrics)
        meta_analysis['limitations_analysis'] = limitations
        
        return meta_analysis
    
    def _identify_complementary_analogies(self, individual_results: Dict[str, Dict[str, Any]], 
                                        performance_metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Identify analogies that provide complementary insights"""
        complementary_pairs = []
        
        analogy_names = list(individual_results.keys())
        
        for i, analogy1 in enumerate(analogy_names):
            for analogy2 in analogy_names[i+1:]:
                # Check if analogies are complementary
                complementarity_score = self._compute_complementarity(
                    individual_results[analogy1], individual_results[analogy2]
                )
                
                if complementarity_score > 0.6:
                    complementary_pairs.append({
                        'analogy_pair': [analogy1, analogy2],
                        'complementarity_score': complementarity_score,
                        'complementary_aspects': self._identify_complementary_aspects(
                            individual_results[analogy1], individual_results[analogy2]
                        )
                    })
        
        # Sort by complementarity score
        complementary_pairs.sort(key=lambda x: x['complementarity_score'], reverse=True)
        
        return complementary_pairs
    
    def _compute_complementarity(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> float:
        """Compute complementarity score between two analogies"""
        complementarity_factors = []
        
        # Regime complementarity
        regime1 = result1.get('regime', '')
        regime2 = result2.get('regime', '')
        
        if regime1 and regime2 and regime1 != regime2:
            complementarity_factors.append(1.0)  # Different regimes are complementary
        else:
            complementarity_factors.append(0.0)
        
        # Parameter complementarity
        params1 = result1.get('parameters', {})
        params2 = result2.get('parameters', {})
        
        if params1 and params2:
            # Check if they focus on different aspects
            common_params = set(params1.keys()) & set(params2.keys())
            total_params = set(params1.keys()) | set(params2.keys())
            
            if total_params:
                param_overlap = len(common_params) / len(total_params)
                param_complementarity = 1.0 - param_overlap
                complementarity_factors.append(param_complementarity)
        
        # Prediction complementarity
        preds1 = result1.get('predictions', {})
        preds2 = result2.get('predictions', {})
        
        if preds1 and preds2:
            pred_overlap = len(set(preds1.keys()) & set(preds2.keys()))
            total_preds = len(set(preds1.keys()) | set(preds2.keys()))
            
            if total_preds > 0:
                pred_complementarity = 1.0 - (pred_overlap / total_preds)
                complementarity_factors.append(pred_complementarity)
        
        if complementarity_factors:
            return np.mean(complementarity_factors)
        else:
            return 0.0
    
    def _identify_complementary_aspects(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> List[str]:
        """Identify specific complementary aspects between analogies"""
        aspects = []
        
        # Regime differences
        regime1 = result1.get('regime', '')
        regime2 = result2.get('regime', '')
        
        if regime1 and regime2 and regime1 != regime2:
            aspects.append(f"Different regimes: {regime1} vs {regime2}")
        
        # Parameter focus differences
        params1 = set(result1.get('parameters', {}).keys())
        params2 = set(result2.get('parameters', {}).keys())
        
        unique_to_1 = params1 - params2
        unique_to_2 = params2 - params1
        
        if unique_to_1:
            aspects.append(f"Unique parameters in first analogy: {', '.join(unique_to_1)}")
        if unique_to_2:
            aspects.append(f"Unique parameters in second analogy: {', '.join(unique_to_2)}")
        
        # Prediction focus differences
        preds1 = set(result1.get('predictions', {}).keys())
        preds2 = set(result2.get('predictions', {}).keys())
        
        unique_preds_1 = preds1 - preds2
        unique_preds_2 = preds2 - preds1
        
        if unique_preds_1:
            aspects.append(f"Unique predictions in first analogy: {', '.join(unique_preds_1)}")
        if unique_preds_2:
            aspects.append(f"Unique predictions in second analogy: {', '.join(unique_preds_2)}")
        
        return aspects
    
    def _extract_emergent_insights(self, individual_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Extract insights that emerge from ensemble analysis"""
        insights = []
        
        # Look for patterns across analogies
        regimes = [result.get('regime', '') for result in individual_results.values()]
        unique_regimes = set(regimes)
        
        if len(unique_regimes) > 1:
            insights.append(f"System exhibits multiple dynamical regimes: {', '.join(unique_regimes)}")
        
        # Critical points consensus
        critical_points = []
        for result in individual_results.values():
            if 'critical_points' in result:
                critical_points.extend(result['critical_points'])
        
        if len(critical_points) > 2:
            insights.append("Multiple analogies identify critical points, suggesting system near transition")
        
        # Parameter scale analysis
        all_parameters = {}
        for analogy_name, result in individual_results.items():
            params = result.get('parameters', {})
            for param_name, param_value in params.items():
                if isinstance(param_value, (int, float)):
                    if param_name not in all_parameters:
                        all_parameters[param_name] = []
                    all_parameters[param_name].append((analogy_name, param_value))
        
        # Look for consistent parameter scales across analogies
        for param_name, param_data in all_parameters.items():
            if len(param_data) > 1:
                values = [value for _, value in param_data]
                if max(values) / min(values) < 2.0:  # Within factor of 2
                    insights.append(f"Parameter {param_name} shows consistent scale across analogies")
        
        return insights
    
    def _synthesize_theoretical_insights(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Synthesize theoretical insights across analogies"""
        synthesis = {}
        
        # Mechanism synthesis
        mechanisms = []
        for analogy_name, result in individual_results.items():
            interpretation = result.get('interpretation', {})
            for key, value in interpretation.items():
                if 'mechanism' in key.lower() or 'meaning' in key.lower():
                    mechanisms.append(f"{analogy_name}: {value}")
        
        if mechanisms:
            synthesis['unified_mechanisms'] = "; ".join(mechanisms)
        
        # Regime synthesis
        regimes = {}
        for analogy_name, result in individual_results.items():
            regime = result.get('regime', '')
            if regime:
                regimes[analogy_name] = regime
        
        if regimes:
            synthesis['regime_landscape'] = str(regimes)
        
        # Dynamic synthesis
        dynamics = []
        for analogy_name, result in individual_results.items():
            interpretation = result.get('interpretation', {})
            if 'dynamics' in interpretation:
                dynamics.append(f"{analogy_name}: {interpretation['dynamics']}")
        
        if dynamics:
            synthesis['unified_dynamics'] = "; ".join(dynamics)
        
        return synthesis
    
    def _analyze_limitations(self, individual_results: Dict[str, Dict[str, Any]], 
                           performance_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze limitations of individual analogies and ensemble"""
        limitations = {
            'individual_limitations': {},
            'ensemble_limitations': [],
            'gaps_in_coverage': [],
            'uncertainty_sources': []
        }
        
        # Individual analogy limitations
        for analogy_name, metrics in performance_metrics.items():
            analogy_limitations = []
            
            if metrics['goodness_of_fit'] < 0.5:
                analogy_limitations.append("Poor fit to data")
            if metrics['interpretability'] < 0.3:
                analogy_limitations.append("Limited interpretability")
            if metrics['predictive_power'] < 0.4:
                analogy_limitations.append("Low predictive power")
            if metrics['robustness'] < 0.4:
                analogy_limitations.append("Limited robustness")
            
            limitations['individual_limitations'][analogy_name] = analogy_limitations
        
        # Ensemble limitations
        if max(metrics['overall_score'] for metrics in performance_metrics.values()) < 0.6:
            limitations['ensemble_limitations'].append("No analogy achieves high overall performance")
        
        regimes = [result.get('regime', '') for result in individual_results.values()]
        if len(set(regimes)) == len(regimes):
            limitations['ensemble_limitations'].append("Complete disagreement on system regime")
        
        # Gaps in coverage
        all_prediction_types = set()
        for result in individual_results.values():
            if 'predictions' in result:
                all_prediction_types.update(result['predictions'].keys())
        
        prediction_coverage = {}
        for pred_type in all_prediction_types:
            coverage_count = 0
            for result in individual_results.values():
                if 'predictions' in result and pred_type in result['predictions']:
                    coverage_count += 1
            prediction_coverage[pred_type] = coverage_count / len(individual_results)
        
        for pred_type, coverage in prediction_coverage.items():
            if coverage < 0.5:
                limitations['gaps_in_coverage'].append(f"Limited coverage for {pred_type}")
        
        return limitations
    
    def _determine_ensemble_regime(self, individual_results: Dict[str, Dict[str, Any]]) -> str:
        """Determine overall ensemble regime"""
        regimes = [result.get('regime', 'unknown') for result in individual_results.values()]
        
        # Count regime occurrences
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if regime_counts:
            # Return most common regime
            consensus_regime = max(regime_counts.keys(), key=lambda k: regime_counts[k])
            max_count = regime_counts[consensus_regime]
            
            # Add confidence indicator
            confidence = max_count / len(regimes)
            
            if confidence >= 0.6:
                return f"{consensus_regime}_consensus"
            else:
                return f"{consensus_regime}_partial"
        
        return "unknown"
    
    def _extract_consensus_insights(self, individual_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Extract insights with high consensus across analogies"""
        consensus_insights = []
        
        # Find common interpretations
        all_interpretations = defaultdict(list)
        
        for analogy_name, result in individual_results.items():
            interpretation = result.get('interpretation', {})
            for key, value in interpretation.items():
                all_interpretations[key].append(value)
        
        # Identify high-consensus interpretations
        for key, values in all_interpretations.items():
            if len(values) >= 3:  # At least 3 analogies agree
                consensus_insights.append(f"High consensus on {key}: {values[0]}")
        
        return consensus_insights
    
    def _interpret_ensemble_results(self, individual_results: Dict[str, Dict[str, Any]], 
                                  ensemble_weights: Dict[str, float]) -> Dict[str, str]:
        """Interpret ensemble results"""
        ensemble_interpretation = {
            'ensemble_approach': 'Multiple theoretical analogies provide complementary perspectives',
            'weight_distribution': f"Weights: {ensemble_weights}",
            'consensus_level': self._assess_consensus_level(individual_results),
            'dominant_analogy': max(ensemble_weights.keys(), key=lambda k: ensemble_weights[k]),
            'theoretical_synthesis': 'Ensemble reveals multi-scale and multi-mechanism dynamics'
        }
        
        # Add dominant analogy interpretation
        dominant_analogy = ensemble_interpretation['dominant_analogy']
        if dominant_analogy in individual_results:
            dominant_interpretation = individual_results[dominant_analogy].get('interpretation', {})
            ensemble_interpretation['dominant_perspective'] = str(dominant_interpretation)
        
        return ensemble_interpretation
    
    def _assess_consensus_level(self, individual_results: Dict[str, Dict[str, Any]]) -> str:
        """Assess level of consensus across analogies"""
        regimes = [result.get('regime', '') for result in individual_results.values()]
        unique_regimes = set(regimes)
        
        consensus_ratio = (len(regimes) - len(unique_regimes) + 1) / len(regimes)
        
        if consensus_ratio >= 0.8:
            return "high_consensus"
        elif consensus_ratio >= 0.6:
            return "moderate_consensus"
        else:
            return "low_consensus"
    
    def predict_dynamics(self, current_state: Dict[str, Any], 
                        time_horizon: int) -> Dict[str, Any]:
        """Generate ensemble predictions for future dynamics"""
        # Get predictions from individual analogies
        individual_predictions = {}
        
        for analogy_name, analogy in self.analogies.items():
            try:
                prediction = analogy.predict_dynamics(current_state, time_horizon)
                individual_predictions[analogy_name] = prediction
            except Exception as e:
                individual_predictions[analogy_name] = {'error': str(e)}
        
        # Combine predictions using ensemble weights
        ensemble_prediction = self._combine_predictions(
            individual_predictions, self.analogy_weights
        )
        
        # Add ensemble-specific predictions
        ensemble_prediction['ensemble_uncertainty'] = self._predict_uncertainty(individual_predictions)
        ensemble_prediction['consensus_trajectory'] = self._predict_consensus_trajectory(individual_predictions)
        
        return ensemble_prediction
    
    def _predict_uncertainty(self, individual_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Predict uncertainty based on model disagreement"""
        uncertainty_metrics = {}
        
        # Collect prediction values for common types
        prediction_types = set()
        for pred in individual_predictions.values():
            if isinstance(pred, dict) and 'error' not in pred:
                prediction_types.update(pred.keys())
        
        for pred_type in prediction_types:
            values = []
            for pred in individual_predictions.values():
                if isinstance(pred, dict) and pred_type in pred:
                    value = pred[pred_type]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if len(values) > 1:
                uncertainty_metrics[pred_type] = np.std(values)
        
        return uncertainty_metrics
    
    def _predict_consensus_trajectory(self, individual_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Predict trajectory of ensemble consensus"""
        consensus_trajectory = {
            'initial_consensus': self._assess_consensus_level(individual_predictions),
            'predicted_consensus': 'stable',  # Default
            'consensus_confidence': 0.5
        }
        
        # Analyze prediction agreement
        agreement_scores = []
        for analogy1 in individual_predictions.keys():
            for analogy2 in individual_predictions.keys():
                if analogy1 != analogy2:
                    agreement = self._compute_prediction_agreement(
                        individual_predictions[analogy1], 
                        individual_predictions[analogy2]
                    )
                    agreement_scores.append(agreement)
        
        if agreement_scores:
            mean_agreement = np.mean(agreement_scores)
            consensus_trajectory['consensus_confidence'] = mean_agreement
            
            if mean_agreement > 0.7:
                consensus_trajectory['predicted_consensus'] = 'increasing'
            elif mean_agreement < 0.3:
                consensus_trajectory['predicted_consensus'] = 'decreasing'
        
        return consensus_trajectory
    
    def _compute_prediction_agreement(self, pred1: Dict[str, Any], pred2: Dict[str, Any]) -> float:
        """Compute agreement between two predictions"""
        if 'error' in pred1 or 'error' in pred2:
            return 0.0
        
        common_keys = set(pred1.keys()) & set(pred2.keys())
        
        if not common_keys:
            return 0.0
        
        agreements = []
        for key in common_keys:
            val1, val2 = pred1[key], pred2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1) + abs(val2) > 0:
                    agreement = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2))
                else:
                    agreement = 1.0
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.0
    
    def compute_analogy_metrics(self, flows: List[SemanticFlow]) -> Dict[str, float]:
        """Compute ensemble-specific metrics"""
        metrics = {}
        
        # Ensemble performance
        if self.analogy_weights:
            metrics['ensemble_diversity'] = 1.0 - max(self.analogy_weights.values())
            metrics['weight_entropy'] = -sum(w * np.log(w + 1e-10) for w in self.analogy_weights.values())
        
        # Consensus metrics
        if hasattr(self, 'model_agreements'):
            metrics['model_consensus'] = self.model_agreements.get('consensus_strength', 0.0)
        
        return metrics
    
    def interpret_results(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret ensemble model results"""
        interpretation = model_results.get('interpretation', {})
        
        # Add ensemble-specific interpretation
        ensemble_interpretation = {
            'ensemble_methodology': 'Combines insights from multiple theoretical analogies',
            'theoretical_diversity': 'Captures different aspects of semantic dynamics',
            'complementary_perspectives': 'Each analogy illuminates different mechanisms',
            'consensus_insights': 'High-confidence insights supported by multiple analogies',
            'uncertainty_quantification': 'Model disagreement quantifies epistemic uncertainty',
            'meta_theoretical_analysis': 'Reveals universal vs context-specific patterns'
        }
        
        ensemble_interpretation.update(interpretation)
        return ensemble_interpretation