#!/usr/bin/env python3
"""
Day 8: Bias Mitigation Suite - Advanced Mathematical Bias Reduction
================================================================

Implements mathematical algorithms to reduce bias in facial recognition systems.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class BiasMitigationSuite:
    """Comprehensive bias mitigation algorithms for facial recognition systems."""
    
    def __init__(self):
        self.mitigation_history = []
        self.validation_results = {}
        print("🛡️ Bias Mitigation Suite initialized!")
    
    # ========================================================================
    # 1. POST-PROCESSING APPROACHES
    # ========================================================================
    
    def threshold_optimization(self, scores: np.ndarray, demographics: np.ndarray, 
                             true_labels: np.ndarray, fairness_constraint: str = 'demographic_parity') -> Dict[str, Any]:
        """
        Optimize decision thresholds using constrained optimization for fairness.
        
        EXPLANATION FOR BEGINNERS:
        Instead of using the same threshold for everyone, we find different thresholds
        for each group that make the overall system more fair while keeping accuracy high.
        """
        
        unique_groups = np.unique(demographics)
        n_groups = len(unique_groups)
        
        # Define optimization objective
        def objective(thresholds):
            total_accuracy = 0
            fairness_violations = 0
            
            for i, group in enumerate(unique_groups):
                group_mask = demographics == group
                group_scores = scores[group_mask]
                group_labels = true_labels[group_mask]
                
                if len(group_scores) == 0:
                    continue
                
                # Apply threshold
                predictions = (group_scores >= thresholds[i]).astype(int)
                accuracy = np.mean(predictions == group_labels)
                total_accuracy += accuracy * len(group_scores)
                
                # Calculate fairness violation
                if fairness_constraint == 'demographic_parity':
                    positive_rate = np.mean(predictions)
                    fairness_violations += abs(positive_rate - 0.5)
            
            return -(total_accuracy / len(scores)) + 0.5 * fairness_violations
        
        # Optimize thresholds
        initial_thresholds = np.full(n_groups, 0.5)
        bounds = [(0.1, 0.9) for _ in range(n_groups)]
        
        result = optimize.minimize(objective, initial_thresholds, method='L-BFGS-B', bounds=bounds)
        optimized_thresholds = result.x
        
        # Calculate before/after metrics
        before_metrics = self._calculate_fairness_metrics(scores, demographics, true_labels, np.full(n_groups, 0.5))
        after_metrics = self._calculate_fairness_metrics(scores, demographics, true_labels, optimized_thresholds)
        
        return {
            'method': 'threshold_optimization',
            'optimized_thresholds': {str(group): thresh for group, thresh in zip(unique_groups, optimized_thresholds)},
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'improvement': {
                'accuracy_change': after_metrics['overall_accuracy'] - before_metrics['overall_accuracy'],
                'fairness_improvement': before_metrics['fairness_violation'] - after_metrics['fairness_violation']
            }
        }
    
    def lagrange_multiplier_fairness(self, scores: np.ndarray, demographics: np.ndarray, 
                                   true_labels: np.ndarray, fairness_weight: float = 0.5) -> Dict[str, Any]:
        """
        Apply Lagrange multiplier method for fairness constraints.
        
        EXPLANATION FOR BEGINNERS:
        Uses advanced calculus to find the best balance between accuracy and fairness.
        """
        
        unique_groups = np.unique(demographics)
        n_groups = len(unique_groups)
        
        def objective_function(adjustment_params):
            total_accuracy = 0
            fairness_violations = 0
            total_samples = 0
            
            for i, group in enumerate(unique_groups):
                group_mask = demographics == group
                group_scores = scores[group_mask]
                group_labels = true_labels[group_mask]
                
                if len(group_scores) == 0:
                    continue
                
                adjusted_group_scores = group_scores * adjustment_params[i]
                predictions = (adjusted_group_scores >= 0.5).astype(int)
                
                accuracy = np.mean(predictions == group_labels)
                total_accuracy += accuracy * len(group_scores)
                total_samples += len(group_scores)
                
                positive_rate = np.mean(predictions)
                fairness_violations += abs(positive_rate - 0.5)
            
            overall_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
            return -(overall_accuracy - fairness_weight * fairness_violations)
        
        # Optimize with constraints
        initial_params = np.ones(n_groups)
        bounds = [(0.5, 2.0) for _ in range(n_groups)]
        
        result = optimize.minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
        optimal_adjustments = result.x
        
        # Apply adjustments
        adjusted_scores = scores.copy()
        for i, group in enumerate(unique_groups):
            group_mask = demographics == group
            adjusted_scores[group_mask] *= optimal_adjustments[i]
        
        return {
            'method': 'lagrange_multiplier_fairness',
            'optimal_adjustments': {str(group): adj for group, adj in zip(unique_groups, optimal_adjustments)},
            'adjusted_scores': adjusted_scores,
            'optimization_success': result.success
        }
    
    # ========================================================================
    # 4. VALIDATION METHODS
    # ========================================================================
    
    def bootstrap_validation(self, scores: np.ndarray, demographics: np.ndarray, 
                           true_labels: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Bootstrap validation for bias metrics with confidence intervals.
        
        EXPLANATION FOR BEGINNERS:
        Bootstrap is like doing the same experiment many times with slightly
        different data to see how reliable our results are. It gives us
        confidence intervals - "we're 95% sure the bias is between X and Y".
        """
        
        bootstrap_results = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(scores)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            boot_scores = scores[bootstrap_indices]
            boot_demographics = demographics[bootstrap_indices]
            boot_labels = true_labels[bootstrap_indices]
            
            # Calculate bias metrics for bootstrap sample
            bias_score = self._calculate_overall_bias(boot_scores, boot_demographics, boot_labels)
            bootstrap_results.append(bias_score)
        
        # Calculate confidence intervals
        bootstrap_results = np.array(bootstrap_results)
        confidence_intervals = {
            '95%': (np.percentile(bootstrap_results, 2.5), np.percentile(bootstrap_results, 97.5)),
            '90%': (np.percentile(bootstrap_results, 5), np.percentile(bootstrap_results, 95)),
            '99%': (np.percentile(bootstrap_results, 0.5), np.percentile(bootstrap_results, 99.5))
        }
        
        return {
            'method': 'bootstrap_validation',
            'n_bootstrap_samples': n_bootstrap,
            'bias_score_distribution': {
                'mean': np.mean(bootstrap_results),
                'std': np.std(bootstrap_results),
                'median': np.median(bootstrap_results)
            },
            'confidence_intervals': confidence_intervals,
            'bootstrap_results': bootstrap_results.tolist()
        }
    
    def permutation_significance_test(self, scores: np.ndarray, demographics: np.ndarray, 
                                    true_labels: np.ndarray, n_permutations: int = 1000) -> Dict[str, Any]:
        """
        Permutation tests for statistical significance of bias.
        
        EXPLANATION FOR BEGINNERS:
        This tests if the bias we see is "real" or just random chance.
        We shuffle the demographic labels randomly many times and see
        if we still get the same bias patterns.
        """
        
        # Calculate observed bias
        observed_bias = self._calculate_overall_bias(scores, demographics, true_labels)
        
        # Permutation test
        permuted_biases = []
        
        for i in range(n_permutations):
            # Randomly shuffle demographic labels
            permuted_demographics = np.random.permutation(demographics)
            
            # Calculate bias with shuffled labels
            permuted_bias = self._calculate_overall_bias(scores, permuted_demographics, true_labels)
            permuted_biases.append(permuted_bias)
        
        # Calculate p-value
        permuted_biases = np.array(permuted_biases)
        p_value = np.mean(permuted_biases >= observed_bias)
        
        return {
            'method': 'permutation_significance_test',
            'observed_bias': observed_bias,
            'n_permutations': n_permutations,
            'permuted_bias_distribution': {
                'mean': np.mean(permuted_biases),
                'std': np.std(permuted_biases)
            },
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': (observed_bias - np.mean(permuted_biases)) / np.std(permuted_biases)
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _calculate_fairness_metrics(self, scores: np.ndarray, demographics: np.ndarray, 
                                  true_labels: np.ndarray, thresholds: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive fairness metrics."""
        
        unique_groups = np.unique(demographics)
        group_accuracies = []
        group_positive_rates = []
        
        for i, group in enumerate(unique_groups):
            group_mask = demographics == group
            group_scores = scores[group_mask]
            group_labels = true_labels[group_mask]
            
            if len(group_scores) == 0:
                continue
            
            threshold = thresholds[i] if len(thresholds) > i else 0.5
            predictions = (group_scores >= threshold).astype(int)
            
            accuracy = np.mean(predictions == group_labels)
            positive_rate = np.mean(predictions)
            
            group_accuracies.append(accuracy)
            group_positive_rates.append(positive_rate)
        
        return {
            'overall_accuracy': np.mean(group_accuracies),
            'accuracy_variance': np.var(group_accuracies),
            'positive_rate_variance': np.var(group_positive_rates),
            'fairness_violation': np.var(group_accuracies) + np.var(group_positive_rates)
        }
    
    def _calculate_overall_bias(self, scores: np.ndarray, demographics: np.ndarray, 
                              true_labels: np.ndarray) -> float:
        """Calculate overall bias score."""
        
        unique_groups = np.unique(demographics)
        group_accuracies = []
        
        for group in unique_groups:
            group_mask = demographics == group
            group_scores = scores[group_mask]
            group_labels = true_labels[group_mask]
            
            if len(group_scores) > 0:
                predictions = (group_scores >= 0.5).astype(int)
                accuracy = np.mean(predictions == group_labels)
                group_accuracies.append(accuracy)
        
        return np.var(group_accuracies) if len(group_accuracies) > 1 else 0.0
    
    def _find_pareto_knee_point(self, pareto_points: List[Dict]) -> int:
        """Find knee point in Pareto frontier."""
        
        if len(pareto_points) < 3:
            return 0
        
        # Calculate distances from line connecting endpoints
        accuracies = [p['accuracy'] for p in pareto_points]
        fairnesses = [p['fairness'] for p in pareto_points]
        
        # Find point with maximum distance from line
        max_distance = 0
        knee_idx = 0
        
        for i in range(1, len(pareto_points) - 1):
            # Calculate distance from line
            distance = abs((fairnesses[-1] - fairnesses[0]) * accuracies[i] - 
                          (accuracies[-1] - accuracies[0]) * fairnesses[i] + 
                          accuracies[-1] * fairnesses[0] - fairnesses[-1] * accuracies[0])
            
            if distance > max_distance:
                max_distance = distance
                knee_idx = i
        
        return knee_idx
    
    def run_comprehensive_mitigation(self, scores: np.ndarray, demographics: np.ndarray, 
                                   true_labels: np.ndarray) -> Dict[str, Any]:
        """
        Run all bias mitigation methods and compare results.
        
        EXPLANATION FOR BEGINNERS:
        This runs all our bias reduction methods and tells you which one
        works best for your specific data. It's like trying different
        medicines and seeing which one cures the bias most effectively.
        """
        
        print("🔬 Running Comprehensive Bias Mitigation Analysis")
        print("=" * 60)
        
        results = {
            'original_bias_score': self._calculate_overall_bias(scores, demographics, true_labels),
            'mitigation_methods': {}
        }
        
        # 1. Threshold Optimization
        print("🎯 Running threshold optimization...")
        threshold_result = self.threshold_optimization(scores, demographics, true_labels)
        results['mitigation_methods']['threshold_optimization'] = threshold_result
        
        # 2. Lagrange Multiplier Fairness
        print("📐 Running Lagrange multiplier optimization...")
        lagrange_result = self.lagrange_multiplier_fairness(scores, demographics, true_labels)
        results['mitigation_methods']['lagrange_multiplier'] = lagrange_result
        
        # 3. Bootstrap Validation
        print("🔄 Running bootstrap validation...")
        bootstrap_result = self.bootstrap_validation(scores, demographics, true_labels, n_bootstrap=100)
        results['validation'] = {'bootstrap': bootstrap_result}
        
        # 4. Permutation Test
        print("🎲 Running permutation significance test...")
        permutation_result = self.permutation_significance_test(scores, demographics, true_labels, n_permutations=100)
        results['validation']['permutation_test'] = permutation_result
        
        # Find best method
        best_method = self._find_best_mitigation_method(results['mitigation_methods'])
        results['best_method'] = best_method
        
        print("✅ Comprehensive mitigation analysis complete!")
        return results
    
    def _find_best_mitigation_method(self, methods: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing mitigation method."""
        
        best_score = float('inf')
        best_method_name = None
        
        for method_name, method_result in methods.items():
            if 'improvement' in method_result:
                # Score based on fairness improvement and accuracy preservation
                fairness_gain = method_result['improvement'].get('fairness_improvement', 0)
                accuracy_loss = abs(method_result['improvement'].get('accuracy_change', 0))
                
                score = -fairness_gain + 0.5 * accuracy_loss  # Lower is better
                
                if score < best_score:
                    best_score = score
                    best_method_name = method_name
        
        return {
            'name': best_method_name,
            'score': best_score,
            'recommendation': f"Use {best_method_name} for optimal bias reduction"
        }

# ============================================================================
# PERFORMANCE OPTIMIZATION SUITE
# ============================================================================

class PerformanceOptimizer:
    """Performance optimization for large-scale bias analysis."""
    
    def __init__(self):
        self.cache = {}
        self.computation_stats = {}
        print("⚡ Performance Optimizer initialized!")
    
    def vectorized_bias_computation(self, scores: np.ndarray, demographics: np.ndarray) -> Dict[str, Any]:
        """Vectorized operations for mathematical computations."""
        
        # Use numpy broadcasting for efficient computation
        unique_groups = np.unique(demographics)
        group_masks = demographics[:, np.newaxis] == unique_groups[np.newaxis, :]
        
        # Vectorized group statistics
        group_means = np.sum(scores[:, np.newaxis] * group_masks, axis=0) / np.sum(group_masks, axis=0)
        group_vars = np.sum((scores[:, np.newaxis] - group_means[np.newaxis, :]) ** 2 * group_masks, axis=0) / np.sum(group_masks, axis=0)
        
        # Overall bias metrics
        bias_score = np.var(group_means)
        disparity_score = np.max(group_means) - np.min(group_means)
        
        return {
            'vectorized_computation': True,
            'group_means': group_means.tolist(),
            'group_variances': group_vars.tolist(),
            'bias_score': float(bias_score),
            'disparity_score': float(disparity_score),
            'computation_method': 'numpy_broadcasting'
        }
    
    def parallel_chunk_processing(self, data: np.ndarray, chunk_size: int = 1000) -> Dict[str, Any]:
        """Parallel processing for batch analysis."""
        
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        n_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size > 0 else 0)
        chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
        
        def process_chunk(chunk):
            # Simulate expensive computation
            return {
                'chunk_size': len(chunk),
                'mean': np.mean(chunk),
                'std': np.std(chunk),
                'bias_estimate': np.var(chunk)
            }
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            chunk_results = list(executor.map(process_chunk, chunks))
        
        return {
            'parallel_processing': True,
            'n_chunks': n_chunks,
            'chunk_size': chunk_size,
            'chunk_results': chunk_results,
            'total_samples': len(data)
        }

# ============================================================================
# EXECUTION AND TESTING
# ============================================================================

def run_day8_bias_mitigation():
    """Execute Day 8 bias mitigation analysis."""
    
    print("🌅 DAY 8: Advanced Bias Mitigation & Optimization")
    print("=" * 60)
    
    # Generate synthetic biased data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Create biased synthetic dataset
    demographics = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Generate biased scores
    base_scores = np.random.beta(2, 2, n_samples)
    bias_factors = np.array([1.0, 0.8, 0.9, 0.7])[demographics]  # Different bias for each group
    biased_scores = base_scores * bias_factors
    biased_scores = np.clip(biased_scores, 0, 1)
    
    # Generate true labels
    true_labels = (base_scores > 0.6).astype(int)
    
    print(f"📊 Generated dataset: {n_samples} samples, {len(np.unique(demographics))} demographic groups")
    print(f"🎯 Original bias score: {BiasMitigationSuite()._calculate_overall_bias(biased_scores, demographics, true_labels):.4f}")
    
    # Initialize mitigation suite
    mitigation_suite = BiasMitigationSuite()
    
    # Run comprehensive mitigation
    results = mitigation_suite.run_comprehensive_mitigation(biased_scores, demographics, true_labels)
    
    # Performance optimization
    optimizer = PerformanceOptimizer()
    perf_results = optimizer.vectorized_bias_computation(biased_scores, demographics)
    
    print("\n📈 Mitigation Results Summary:")
    print(f"  Original Bias: {results['original_bias_score']:.4f}")
    
    if 'best_method' in results:
        print(f"  Best Method: {results['best_method']['name']}")
        print(f"  Recommendation: {results['best_method']['recommendation']}")
    
    return {
        'mitigation_results': results,
        'performance_optimization': perf_results,
        'dataset_info': {
            'n_samples': n_samples,
            'n_groups': len(np.unique(demographics)),
            'original_bias': results['original_bias_score']
        }
    }

if __name__ == "__main__":
    results = run_day8_bias_mitigation()
    print("✅ Day 8: Bias Mitigation Suite Complete!")
