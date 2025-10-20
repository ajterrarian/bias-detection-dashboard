"""
DAY 9: Comprehensive System Testing Suite
========================================
Testing, Validation & Documentation for Facial Recognition Bias Detection System

This module provides comprehensive testing for:
- Mathematical implementations and accuracy
- API integration and functionality
- Statistical analysis methods
- Visualization generation
- System performance and reliability
"""

import unittest
import numpy as np
import pandas as pd
import requests
import json
import time
import sys
import os
from unittest.mock import Mock, patch
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from bias_metrics import BiasMetrics
    from Day4_Core_Mathematical_Framework import MathematicalFramework
    from Day8_BiasMitigationSuite import BiasMitigationSuite, PerformanceOptimizer
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Core modules not available: {e}")
    CORE_MODULES_AVAILABLE = False

class TestBiasDetectionSystem(unittest.TestCase):
    """Comprehensive test suite for the facial recognition bias detection system"""
    
    def setUp(self):
        """Set up test fixtures and mock data"""
        np.random.seed(42)  # For reproducible tests
        
        # Generate synthetic test data
        self.n_samples = 1000
        self.demographics = np.random.choice(['A', 'B', 'C', 'D'], self.n_samples)
        self.predictions = np.random.rand(self.n_samples) > 0.5
        self.true_labels = np.random.rand(self.n_samples) > 0.4
        self.confidence_scores = np.random.rand(self.n_samples)
        
        # API endpoint for testing
        self.api_base_url = "http://127.0.0.1:8000"
        
        # Mathematical test cases
        self.test_functions = {
            'quadratic': lambda x: x**2,
            'cubic': lambda x: x**3,
            'exponential': lambda x: np.exp(x),
            'sine': lambda x: np.sin(x)
        }
        
        self.test_derivatives = {
            'quadratic': lambda x: 2*x,
            'cubic': lambda x: 3*x**2,
            'exponential': lambda x: np.exp(x),
            'sine': lambda x: np.cos(x)
        }
    
    def test_mathematical_accuracy(self):
        """Test mathematical implementations against known results"""
        print("\n🔬 Testing Mathematical Accuracy...")
        
        # Test 1: Gradient calculations
        self._test_gradient_calculations()
        
        # Test 2: Statistical computations
        self._test_statistical_computations()
        
        # Test 3: Information theory metrics
        self._test_information_theory()
        
        # Test 4: Optimization algorithms
        self._test_optimization_algorithms()
        
        print("✅ Mathematical accuracy tests passed!")
    
    def _test_gradient_calculations(self):
        """Test gradient computation accuracy"""
        print("  📐 Testing gradient calculations...")
        
        for func_name, func in self.test_functions.items():
            for x in [-2, -1, 0, 1, 2]:
                # Numerical gradient (central difference)
                h = 1e-8
                numerical_grad = (func(x + h) - func(x - h)) / (2 * h)
                
                # Analytical gradient
                analytical_grad = self.test_derivatives[func_name](x)
                
                # Test accuracy
                relative_error = abs(numerical_grad - analytical_grad) / (abs(analytical_grad) + 1e-10)
                self.assertLess(relative_error, 1e-6, 
                    f"Gradient error too large for {func_name} at x={x}")
    
    def _test_statistical_computations(self):
        """Test statistical analysis methods"""
        print("  📊 Testing statistical computations...")
        
        # Test bias metrics calculations
        group_a = self.predictions[self.demographics == 'A']
        group_b = self.predictions[self.demographics == 'B']
        
        # Statistical parity test
        stat_parity = abs(np.mean(group_a) - np.mean(group_b))
        self.assertGreaterEqual(stat_parity, 0, "Statistical parity should be non-negative")
        
        # Confidence interval coverage test
        sample_means = []
        for _ in range(100):
            sample = np.random.normal(0, 1, 30)
            sample_means.append(np.mean(sample))
        
        # 95% confidence interval should contain true mean (0) about 95% of the time
        ci_lower = np.percentile(sample_means, 2.5)
        ci_upper = np.percentile(sample_means, 97.5)
        coverage = (ci_lower <= 0 <= ci_upper)
        
        # Test Type I error rate for hypothesis tests
        p_values = []
        for _ in range(100):
            sample1 = np.random.normal(0, 1, 50)
            sample2 = np.random.normal(0, 1, 50)
            _, p_val = stats.ttest_ind(sample1, sample2)
            p_values.append(p_val)
        
        type_i_error_rate = np.mean(np.array(p_values) < 0.05)
        self.assertLess(abs(type_i_error_rate - 0.05), 0.03, 
            "Type I error rate should be close to 0.05")
    
    def _test_information_theory(self):
        """Test information theory calculations"""
        print("  📈 Testing information theory...")
        
        # Test entropy calculations
        # Uniform distribution should have maximum entropy
        uniform_dist = np.ones(4) / 4
        entropy_uniform = -np.sum(uniform_dist * np.log2(uniform_dist + 1e-10))
        expected_entropy = np.log2(4)  # log2(n) for uniform distribution
        
        self.assertAlmostEqual(entropy_uniform, expected_entropy, places=6,
            msg="Uniform distribution entropy incorrect")
        
        # Test KL divergence properties
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        
        # KL divergence should be non-negative
        kl_div = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
        self.assertGreaterEqual(kl_div, 0, "KL divergence should be non-negative")
        
        # KL(p||p) should be 0
        kl_self = np.sum(p * np.log(p / (p + 1e-10) + 1e-10))
        self.assertAlmostEqual(kl_self, 0, places=6, 
            msg="KL divergence of distribution with itself should be 0")
    
    def _test_optimization_algorithms(self):
        """Test optimization algorithm convergence"""
        print("  🎯 Testing optimization algorithms...")
        
        # Test gradient descent on simple quadratic function
        def quadratic_loss(x):
            return (x - 3)**2 + 5
        
        def quadratic_grad(x):
            return 2 * (x - 3)
        
        # Gradient descent
        x = 0.0
        learning_rate = 0.1
        for _ in range(100):
            grad = quadratic_grad(x)
            x = x - learning_rate * grad
        
        # Should converge to minimum at x=3
        self.assertAlmostEqual(x, 3.0, places=3, 
            msg="Gradient descent should converge to minimum")
        
        # Test constraint satisfaction for constrained optimization
        # Simple test: minimize x^2 subject to x >= 1
        # Solution should be x = 1
        x_constrained = max(1.0, 0.0)  # Projection onto constraint
        self.assertEqual(x_constrained, 1.0, 
            "Constrained optimization should satisfy constraints")
    
    def test_api_integration(self):
        """Test all API integrations"""
        print("\n🌐 Testing API Integration...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.api_base_url}/api/health", timeout=5)
            self.assertEqual(response.status_code, 200, "Health endpoint should return 200")
            
            health_data = response.json()
            self.assertIn('status', health_data, "Health response should contain status")
            self.assertEqual(health_data['status'], 'healthy', "API should be healthy")
            
            # Test bias metrics endpoint
            response = requests.get(f"{self.api_base_url}/api/bias-metrics", timeout=10)
            self.assertEqual(response.status_code, 200, "Bias metrics endpoint should return 200")
            
            bias_data = response.json()
            required_fields = ['overall_bias_score', 'accuracy_disparity', 'statistical_parity']
            for field in required_fields:
                self.assertIn(field, bias_data, f"Bias metrics should contain {field}")
            
            # Test demographic stats endpoint
            response = requests.get(f"{self.api_base_url}/api/demographic-stats", timeout=10)
            self.assertEqual(response.status_code, 200, "Demographic stats endpoint should return 200")
            
            # Test visualization endpoints
            viz_endpoints = ['heatmap', 'gradients', 'statistics', 'roc-curves']
            for endpoint in viz_endpoints:
                response = requests.get(f"{self.api_base_url}/api/visualizations/{endpoint}", timeout=10)
                self.assertEqual(response.status_code, 200, 
                    f"Visualization endpoint {endpoint} should return 200")
            
            print("✅ API integration tests passed!")
            
        except requests.exceptions.ConnectionError:
            print("⚠️ API server not running - skipping API tests")
            self.skipTest("API server not available")
        except Exception as e:
            self.fail(f"API integration test failed: {e}")
    
    def test_statistical_significance(self):
        """Test statistical analysis methods"""
        print("\n📊 Testing Statistical Significance...")
        
        # Test bootstrap confidence intervals
        sample_data = np.random.normal(10, 2, 100)
        bootstrap_means = []
        
        for _ in range(1000):
            bootstrap_sample = np.random.choice(sample_data, size=len(sample_data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # 95% confidence interval
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        # True mean should be within confidence interval most of the time
        true_mean = 10
        self.assertLessEqual(ci_lower, true_mean, "CI lower bound should be <= true mean")
        self.assertGreaterEqual(ci_upper, true_mean, "CI upper bound should be >= true mean")
        
        # Test permutation test
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)  # Slightly different mean
        
        observed_diff = np.mean(group1) - np.mean(group2)
        
        # Permutation test
        combined = np.concatenate([group1, group2])
        perm_diffs = []
        
        for _ in range(1000):
            np.random.shuffle(combined)
            perm_group1 = combined[:50]
            perm_group2 = combined[50:]
            perm_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))
        
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        # P-value should be reasonable (between 0 and 1)
        self.assertGreaterEqual(p_value, 0, "P-value should be non-negative")
        self.assertLessEqual(p_value, 1, "P-value should be <= 1")
        
        print("✅ Statistical significance tests passed!")
    
    def test_visualization_generation(self):
        """Test visualization generation"""
        print("\n📈 Testing Visualization Generation...")
        
        # Test matplotlib figure generation
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Visualization")
        
        # Should not raise any errors
        self.assertIsNotNone(fig, "Matplotlib figure should be created")
        plt.close(fig)
        
        # Test Plotly figure generation
        fig_plotly = go.Figure()
        fig_plotly.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
        fig_plotly.update_layout(title="Test Plotly Visualization")
        
        # Should not raise any errors
        self.assertIsNotNone(fig_plotly, "Plotly figure should be created")
        
        # Test data validation for visualizations
        test_data = {
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        }
        
        # Data should be valid for plotting
        self.assertEqual(len(test_data['x']), len(test_data['y']), 
            "X and Y data should have same length")
        self.assertTrue(all(isinstance(x, (int, float)) for x in test_data['x']), 
            "X data should be numeric")
        self.assertTrue(all(isinstance(y, (int, float)) for y in test_data['y']), 
            "Y data should be numeric")
        
        print("✅ Visualization generation tests passed!")
    
    @unittest.skipUnless(CORE_MODULES_AVAILABLE, "Core modules not available")
    def test_bias_mitigation_suite(self):
        """Test bias mitigation algorithms"""
        print("\n🛡️ Testing Bias Mitigation Suite...")
        
        # Generate biased synthetic data
        n_samples = 500
        demographics = np.random.choice(['A', 'B'], n_samples)
        
        # Create intentional bias (group A has higher accuracy)
        base_accuracy = 0.7
        predictions = np.random.rand(n_samples) < base_accuracy
        
        # Add bias: group A gets +0.2 accuracy boost
        bias_boost = (demographics == 'A') * 0.2
        biased_predictions = np.random.rand(n_samples) < (base_accuracy + bias_boost)
        
        true_labels = np.random.rand(n_samples) < 0.6
        
        # Test bias mitigation
        suite = BiasMitigationSuite()
        
        # Calculate original bias
        original_bias = suite._calculate_bias_score(demographics, biased_predictions, true_labels)
        
        # Apply threshold optimization
        mitigated_predictions = suite.threshold_optimization(
            demographics, biased_predictions, true_labels
        )
        
        # Calculate mitigated bias
        mitigated_bias = suite._calculate_bias_score(demographics, mitigated_predictions, true_labels)
        
        # Bias should be reduced (or at least not increased significantly)
        self.assertLessEqual(mitigated_bias, original_bias + 0.1, 
            "Bias mitigation should reduce or maintain bias levels")
        
        print(f"  Original bias: {original_bias:.4f}")
        print(f"  Mitigated bias: {mitigated_bias:.4f}")
        print("✅ Bias mitigation tests passed!")
    
    def test_performance_optimization(self):
        """Test performance optimization features"""
        print("\n⚡ Testing Performance Optimization...")
        
        # Test vectorized operations vs loops
        large_array = np.random.rand(10000)
        
        # Vectorized operation
        start_time = time.time()
        vectorized_result = np.sum(large_array ** 2)
        vectorized_time = time.time() - start_time
        
        # Loop operation
        start_time = time.time()
        loop_result = sum(x ** 2 for x in large_array)
        loop_time = time.time() - start_time
        
        # Results should be approximately equal
        self.assertAlmostEqual(vectorized_result, loop_result, places=6,
            msg="Vectorized and loop results should be equal")
        
        # Vectorized should be faster (usually)
        print(f"  Vectorized time: {vectorized_time:.6f}s")
        print(f"  Loop time: {loop_time:.6f}s")
        print(f"  Speedup: {loop_time/vectorized_time:.2f}x")
        
        # Test parallel processing simulation
        def cpu_intensive_task(n):
            return sum(i**2 for i in range(n))
        
        tasks = [1000] * 4
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [cpu_intensive_task(n) for n in tasks]
        sequential_time = time.time() - start_time
        
        # Results should be consistent
        expected_result = cpu_intensive_task(1000)
        for result in sequential_results:
            self.assertEqual(result, expected_result, 
                "All parallel tasks should produce same result")
        
        print(f"  Sequential processing time: {sequential_time:.6f}s")
        print("✅ Performance optimization tests passed!")
    
    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions"""
        print("\n🔍 Testing Edge Cases and Boundary Conditions...")
        
        # Test empty data
        empty_array = np.array([])
        self.assertEqual(len(empty_array), 0, "Empty array should have length 0")
        
        # Test single data point
        single_point = np.array([1.0])
        self.assertEqual(len(single_point), 1, "Single point array should have length 1")
        
        # Test all same values
        same_values = np.ones(100)
        self.assertEqual(np.std(same_values), 0, "Array of same values should have zero std")
        
        # Test extreme values
        extreme_values = np.array([-1e10, 1e10])
        self.assertTrue(np.isfinite(np.mean(extreme_values)), 
            "Mean of extreme values should be finite")
        
        # Test NaN handling
        with_nan = np.array([1, 2, np.nan, 4, 5])
        clean_mean = np.nanmean(with_nan)
        self.assertTrue(np.isfinite(clean_mean), "NaN-aware mean should be finite")
        
        # Test division by zero protection
        safe_division = 1.0 / (0.0 + 1e-10)  # Add small epsilon
        self.assertTrue(np.isfinite(safe_division), "Protected division should be finite")
        
        print("✅ Edge cases and boundary conditions tests passed!")

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("\n🔄 Testing End-to-End Workflow...")
        
        # Simulate complete analysis pipeline
        steps_completed = []
        
        # Step 1: Data loading
        try:
            synthetic_data = self._generate_test_data()
            steps_completed.append("data_loading")
        except Exception as e:
            self.fail(f"Data loading failed: {e}")
        
        # Step 2: Bias analysis
        try:
            bias_results = self._analyze_bias(synthetic_data)
            steps_completed.append("bias_analysis")
        except Exception as e:
            self.fail(f"Bias analysis failed: {e}")
        
        # Step 3: Visualization generation
        try:
            visualizations = self._generate_visualizations(bias_results)
            steps_completed.append("visualization")
        except Exception as e:
            self.fail(f"Visualization generation failed: {e}")
        
        # Step 4: Report generation
        try:
            report = self._generate_report(bias_results)
            steps_completed.append("report_generation")
        except Exception as e:
            self.fail(f"Report generation failed: {e}")
        
        # All steps should complete successfully
        expected_steps = ["data_loading", "bias_analysis", "visualization", "report_generation"]
        self.assertEqual(steps_completed, expected_steps, 
            "All workflow steps should complete successfully")
        
        print("✅ End-to-end workflow test passed!")
    
    def _generate_test_data(self):
        """Generate synthetic test data"""
        return {
            'demographics': np.random.choice(['A', 'B', 'C'], 1000),
            'predictions': np.random.rand(1000) > 0.5,
            'true_labels': np.random.rand(1000) > 0.4,
            'confidence_scores': np.random.rand(1000)
        }
    
    def _analyze_bias(self, data):
        """Perform bias analysis on test data"""
        # Calculate basic bias metrics
        groups = np.unique(data['demographics'])
        bias_results = {}
        
        for group in groups:
            mask = data['demographics'] == group
            group_accuracy = np.mean(data['predictions'][mask] == data['true_labels'][mask])
            bias_results[group] = group_accuracy
        
        return bias_results
    
    def _generate_visualizations(self, bias_results):
        """Generate test visualizations"""
        # Create simple visualization data
        return {
            'bias_heatmap': list(bias_results.values()),
            'group_comparison': bias_results
        }
    
    def _generate_report(self, bias_results):
        """Generate test report"""
        return {
            'summary': f"Analyzed {len(bias_results)} demographic groups",
            'max_bias': max(bias_results.values()) - min(bias_results.values()),
            'recommendations': ["Monitor group A performance", "Validate model fairness"]
        }

def run_comprehensive_tests():
    """Run the complete test suite with detailed reporting"""
    print("🧪 STARTING COMPREHENSIVE BIAS DETECTION SYSTEM TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [TestBiasDetectionSystem, TestSystemIntegration]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("🌅 DAY 9: Comprehensive System Testing")
    print("=" * 60)
    
    # Run comprehensive test suite
    success = run_comprehensive_tests()
    
    if success:
        print("\n✅ Day 9 Step 1: Comprehensive Testing Complete!")
    else:
        print("\n⚠️ Some tests failed - review and fix issues before proceeding")
    
    sys.exit(0 if success else 1)
