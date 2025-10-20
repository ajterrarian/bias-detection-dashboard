"""
Mathematical bias metrics and differential geometry calculations for facial recognition analysis.
Implements statistical measures to quantify performance disparities across demographic groups.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BiasMetrics:
    """Class for calculating bias metrics and differential geometry measures."""
    
    def __init__(self):
        self.demographic_groups = [
            'young_male', 'young_female', 
            'middle_aged_male', 'middle_aged_female',
            'elderly_male', 'elderly_female',
            'light_skin', 'medium_skin', 'dark_skin'
        ]
    
    def calculate_accuracy_disparity(self, group_accuracies: Dict[str, float]) -> Dict[str, float]:
        """Calculate accuracy disparity metrics across demographic groups."""
        accuracies = list(group_accuracies.values())
        
        metrics = {
            'max_accuracy': max(accuracies),
            'min_accuracy': min(accuracies),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'accuracy_range': max(accuracies) - min(accuracies),
            'coefficient_of_variation': np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0
        }
        
        # Calculate pairwise disparities
        group_names = list(group_accuracies.keys())
        pairwise_disparities = {}
        
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i+1:], i+1):
                disparity = abs(group_accuracies[group1] - group_accuracies[group2])
                pairwise_disparities[f"{group1}_vs_{group2}"] = disparity
        
        metrics['pairwise_disparities'] = pairwise_disparities
        metrics['max_pairwise_disparity'] = max(pairwise_disparities.values()) if pairwise_disparities else 0
        
        return metrics
    
    def calculate_statistical_parity(self, predictions: Dict[str, List], true_labels: Dict[str, List]) -> Dict[str, float]:
        """Calculate statistical parity difference across groups."""
        group_rates = {}
        
        for group, preds in predictions.items():
            if group in true_labels and len(preds) > 0:
                positive_rate = sum(preds) / len(preds)
                group_rates[group] = positive_rate
        
        if len(group_rates) < 2:
            return {'statistical_parity_difference': 0.0}
        
        rates = list(group_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)
        
        return {
            'statistical_parity_difference': max_rate - min_rate,
            'group_rates': group_rates
        }
    
    def calculate_equalized_odds(self, predictions: Dict[str, List], true_labels: Dict[str, List]) -> Dict[str, float]:
        """Calculate equalized odds metrics across groups."""
        group_metrics = {}
        
        for group in predictions.keys():
            if group in true_labels and len(predictions[group]) > 0:
                y_true = true_labels[group]
                y_pred = predictions[group]
                
                # Calculate TPR and FPR
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
        
        # Calculate differences
        if len(group_metrics) < 2:
            return {'equalized_odds_difference': 0.0}
        
        tprs = [metrics['tpr'] for metrics in group_metrics.values()]
        fprs = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)
        
        return {
            'equalized_odds_difference': max(tpr_diff, fpr_diff),
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'group_metrics': group_metrics
        }
    
    def calculate_bias_gradient(self, accuracy_matrix: np.ndarray, demographic_coords: np.ndarray) -> Dict[str, float]:
        """
        Calculate bias gradient using differential geometry.
        
        Args:
            accuracy_matrix: NxM matrix where N is number of models, M is number of demographic groups
            demographic_coords: Mx2 matrix of demographic group coordinates in 2D space
        """
        try:
            # Calculate gradient for each model
            gradients = []
            
            for model_accuracies in accuracy_matrix:
                # Fit a surface to the accuracy data
                from scipy.interpolate import griddata
                
                # Create a regular grid
                xi = np.linspace(demographic_coords[:, 0].min(), demographic_coords[:, 0].max(), 10)
                yi = np.linspace(demographic_coords[:, 1].min(), demographic_coords[:, 1].max(), 10)
                xi_grid, yi_grid = np.meshgrid(xi, yi)
                
                # Interpolate accuracy values
                zi = griddata(demographic_coords, model_accuracies, (xi_grid, yi_grid), method='linear')
                
                # Calculate gradient magnitude
                grad_x, grad_y = np.gradient(zi)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                gradients.append(np.nanmean(gradient_magnitude))
            
            return {
                'mean_gradient_magnitude': np.mean(gradients),
                'max_gradient_magnitude': np.max(gradients),
                'gradient_std': np.std(gradients)
            }
        
        except Exception as e:
            logger.warning(f"Could not calculate bias gradient: {e}")
            return {
                'mean_gradient_magnitude': 0.0,
                'max_gradient_magnitude': 0.0,
                'gradient_std': 0.0
            }
    
    def calculate_demographic_parity(self, predictions: Dict[str, List]) -> Dict[str, float]:
        """Calculate demographic parity metrics."""
        group_positive_rates = {}
        
        for group, preds in predictions.items():
            if len(preds) > 0:
                positive_rate = sum(preds) / len(preds)
                group_positive_rates[group] = positive_rate
        
        if len(group_positive_rates) < 2:
            return {'demographic_parity_difference': 0.0}
        
        rates = list(group_positive_rates.values())
        return {
            'demographic_parity_difference': max(rates) - min(rates),
            'group_positive_rates': group_positive_rates
        }
    
    def calculate_individual_fairness(self, similarities: np.ndarray, outcome_differences: np.ndarray) -> float:
        """
        Calculate individual fairness metric.
        
        Args:
            similarities: Array of similarity scores between individuals
            outcome_differences: Array of outcome differences for similar individuals
        """
        if len(similarities) == 0 or len(outcome_differences) == 0:
            return 0.0
        
        # Individual fairness: similar individuals should have similar outcomes
        correlation, _ = stats.pearsonr(similarities, outcome_differences)
        
        # We want high similarity to correlate with low outcome differences
        # So we return the negative correlation (closer to -1 is better)
        return -correlation if not np.isnan(correlation) else 0.0
    
    def comprehensive_bias_analysis(self, results_data: Dict) -> Dict:
        """Perform comprehensive bias analysis on facial recognition results."""
        analysis = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': {}
        }
        
        # Extract accuracy data by demographic group
        group_accuracies = {}
        group_predictions = {}
        group_true_labels = {}
        
        for group in self.demographic_groups:
            if group in results_data:
                group_data = results_data[group]
                if 'accuracy' in group_data:
                    # Handle both dict and scalar accuracy values
                    if isinstance(group_data['accuracy'], dict):
                        # Use average of all services for overall accuracy
                        accuracies = [v for v in group_data['accuracy'].values() if isinstance(v, (int, float))]
                        if accuracies:
                            group_accuracies[group] = sum(accuracies) / len(accuracies)
                    else:
                        group_accuracies[group] = group_data['accuracy']
                if 'predictions' in group_data:
                    group_predictions[group] = group_data['predictions']
                if 'true_labels' in group_data:
                    group_true_labels[group] = group_data['true_labels']
        
        # Calculate various bias metrics
        if group_accuracies:
            analysis['metrics']['accuracy_disparity'] = self.calculate_accuracy_disparity(group_accuracies)
        
        if group_predictions and group_true_labels:
            analysis['metrics']['statistical_parity'] = self.calculate_statistical_parity(
                group_predictions, group_true_labels
            )
            analysis['metrics']['equalized_odds'] = self.calculate_equalized_odds(
                group_predictions, group_true_labels
            )
            analysis['metrics']['demographic_parity'] = self.calculate_demographic_parity(group_predictions)
        
        # Calculate overall bias score
        bias_score = 0.0
        if 'accuracy_disparity' in analysis['metrics']:
            bias_score += analysis['metrics']['accuracy_disparity'].get('max_pairwise_disparity', 0)
        if 'statistical_parity' in analysis['metrics']:
            bias_score += analysis['metrics']['statistical_parity'].get('statistical_parity_difference', 0)
        
        analysis['overall_bias_score'] = bias_score
        analysis['bias_level'] = self._classify_bias_level(bias_score)
        
        return analysis
    
    def _classify_bias_level(self, bias_score: float) -> str:
        """Classify bias level based on overall bias score."""
        if bias_score < 0.05:
            return 'Low'
        elif bias_score < 0.15:
            return 'Moderate'
        elif bias_score < 0.30:
            return 'High'
        else:
            return 'Severe'
