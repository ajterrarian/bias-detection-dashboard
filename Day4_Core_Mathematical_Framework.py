# ============================================================================
# DAY 4: CORE MATHEMATICAL FRAMEWORK - ADVANCED IMPLEMENTATION
# Complete mathematical framework with gradient analysis, differential geometry,
# optimization theory, and information theory
# Copy each section into separate Google Colab cells
# ============================================================================

# ============================================================================
# CELL 1: Install Advanced Mathematical Libraries (Colab only)
# ============================================================================
# !pip install numpy scipy scikit-learn pandas matplotlib plotly seaborn
# !pip install sympy autograd jax jaxlib cvxpy
# !pip install statsmodels pymc3 arviz theano-pymc
# !pip install numdifftools torch torchvision
# !pip install networkx igraph-python python-igraph
print("✅ Advanced mathematical libraries available!")

# ============================================================================
# CELL 2: Import Libraries and Mathematical Setup
# ============================================================================
import numpy as np
import pandas as pd
from scipy import optimize, stats, linalg
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata, RBFInterpolator
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

# Advanced mathematical libraries
import sympy as sp
from autograd import grad, hessian, jacobian
import autograd.numpy as anp
try:
    import jax.numpy as jnp
    from jax import grad as jax_grad, hessian as jax_hessian
    JAX_AVAILABLE = True
except:
    JAX_AVAILABLE = False
    print("⚠️ JAX not available, using autograd instead")

import cvxpy as cp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("✅ All libraries imported successfully!")

# ============================================================================
# CELL 3: Advanced BiasAnalyzer Class - Core Mathematical Framework
# ============================================================================
class AdvancedBiasAnalyzer:
    """
    Advanced mathematical bias analyzer implementing:
    1. Gradient Analysis with numerical differentiation
    2. Differential Geometry on demographic manifolds  
    3. Optimization Theory for fairness constraints
    4. Information Theory with bias correction
    
    Mathematical Framework:
    - Bias Function: f(x) where x ∈ demographic space
    - Gradient: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    - Hessian: H = [∂²f/∂xᵢ∂xⱼ] (curvature matrix)
    - Riemannian Metric: g(u,v) = uᵀGv (distance measure)
    """
    
    def __init__(self, epsilon: float = 1e-8, max_iterations: int = 1000):
        """
        Initialize the advanced mathematical bias analyzer.
        
        Args:
            epsilon: Small value to prevent numerical issues (like dividing by zero)
            max_iterations: Maximum iterations for optimization algorithms
        """
        print("🔬 Initializing Advanced Mathematical Bias Analyzer...")
        
        self.epsilon = epsilon  # Prevents division by zero
        self.max_iterations = max_iterations
        self.convergence_tolerance = 1e-6
        
        # Storage for computed results
        self.bias_function = None
        self.gradient_cache = {}
        self.hessian_cache = {}
        self.metric_tensor = None
        self.demographic_manifold = None
        
        # Mathematical constants
        self.golden_ratio = (1 + np.sqrt(5)) / 2  # For optimization
        
        print("✅ Advanced BiasAnalyzer initialized!")
    
    # ========================================================================
    # SECTION 1: GRADIENT ANALYSIS
    # ========================================================================
    
    def compute_numerical_gradient(self, f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Compute numerical gradient using central differences.
        
        EXPLANATION FOR BEGINNERS:
        - A gradient tells us the "slope" or "direction of steepest increase"
        - Like standing on a hill and finding which way is steepest uphill
        - We use "central differences": look a tiny step left and right, see how much the function changes
        
        Mathematical Formula: ∇f(x) ≈ [f(x+h) - f(x-h)] / (2h)
        
        Args:
            f: Function to differentiate (our bias function)
            x: Point where we want the gradient (demographic coordinates)
            h: Step size (how big steps we take to estimate slope)
            
        Returns:
            Gradient vector (direction of steepest bias increase)
        """
        print(f"🔢 Computing numerical gradient at point {x}")
        
        gradient = np.zeros_like(x)
        
        for i in range(len(x)):
            # Create points slightly to the left and right
            x_forward = x.copy()
            x_backward = x.copy()
            
            x_forward[i] += h   # Step forward in dimension i
            x_backward[i] -= h  # Step backward in dimension i
            
            # Central difference formula
            try:
                f_forward = f(x_forward)
                f_backward = f(x_backward)
                gradient[i] = (f_forward - f_backward) / (2 * h)
            except Exception as e:
                logger.warning(f"Gradient computation failed at dimension {i}: {e}")
                gradient[i] = 0
        
        # Cache the result
        cache_key = tuple(x)
        self.gradient_cache[cache_key] = gradient
        
        print(f"  ✅ Gradient computed: magnitude = {np.linalg.norm(gradient):.6f}")
        return gradient
    
    def compute_hessian_matrix(self, f: Callable, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """
        Compute Hessian matrix (second derivatives) for curvature analysis.
        
        EXPLANATION FOR BEGINNERS:
        - Hessian tells us about "curvature" - is the surface curved like a bowl or saddle?
        - Like gradient tells us slope, Hessian tells us how the slope is changing
        - Positive eigenvalues = bowl-shaped (minimum), negative = mountain peak (maximum)
        
        Mathematical Formula: H[i,j] = ∂²f/∂xᵢ∂xⱼ
        
        Args:
            f: Function to analyze
            x: Point for Hessian computation
            h: Step size for numerical differentiation
            
        Returns:
            Hessian matrix (curvature information)
        """
        print(f"🌊 Computing Hessian matrix at point {x}")
        
        n = len(x)
        hessian = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Second derivative: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    
                    try:
                        f_plus = f(x_plus)
                        f_center = f(x)
                        f_minus = f(x_minus)
                        hessian[i, j] = (f_plus - 2*f_center + f_minus) / (h**2)
                    except:
                        hessian[i, j] = 0
                else:
                    # Mixed partial derivative: ∂²f/∂xᵢ∂xⱼ
                    x_pp = x.copy()  # +i, +j
                    x_pm = x.copy()  # +i, -j  
                    x_mp = x.copy()  # -i, +j
                    x_mm = x.copy()  # -i, -j
                    
                    x_pp[i] += h; x_pp[j] += h
                    x_pm[i] += h; x_pm[j] -= h
                    x_mp[i] -= h; x_mp[j] += h
                    x_mm[i] -= h; x_mm[j] -= h
                    
                    try:
                        f_pp = f(x_pp)
                        f_pm = f(x_pm)
                        f_mp = f(x_mp)
                        f_mm = f(x_mm)
                        hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)
                    except:
                        hessian[i, j] = 0
        
        # Make symmetric (should be symmetric for real functions)
        hessian = (hessian + hessian.T) / 2
        
        # Cache result
        cache_key = tuple(x)
        self.hessian_cache[cache_key] = hessian
        
        # Analyze curvature
        eigenvals = np.linalg.eigvals(hessian)
        curvature_type = "saddle"
        if np.all(eigenvals > 0):
            curvature_type = "convex (bowl-shaped)"
        elif np.all(eigenvals < 0):
            curvature_type = "concave (mountain-shaped)"
        
        print(f"  ✅ Hessian computed: curvature type = {curvature_type}")
        print(f"  📊 Eigenvalues: {eigenvals}")
        
        return hessian
    
    def gradient_descent_optimization(self, f: Callable, x0: np.ndarray, 
                                    learning_rate: float = 0.01, 
                                    tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Implement gradient descent to find bias minimum.
        
        EXPLANATION FOR BEGINNERS:
        - Gradient descent is like rolling a ball down a hill to find the bottom
        - We start somewhere, look at the slope, take a step downhill, repeat
        - Eventually we reach the bottom (minimum bias point)
        
        Algorithm:
        1. Start at point x₀
        2. Compute gradient ∇f(x)
        3. Take step: x_new = x - learning_rate * ∇f(x)
        4. Repeat until we stop moving (converged)
        
        Args:
            f: Bias function to minimize
            x0: Starting point
            learning_rate: How big steps to take (too big = overshoot, too small = slow)
            tolerance: When to stop (how close to minimum is "good enough")
            
        Returns:
            Optimization results with path and final point
        """
        print(f"🎯 Running gradient descent optimization from {x0}")
        
        x = x0.copy()
        path = [x.copy()]
        gradients = []
        function_values = []
        
        for iteration in range(self.max_iterations):
            # Compute current function value and gradient
            current_f = f(x)
            current_grad = self.compute_numerical_gradient(f, x)
            
            function_values.append(current_f)
            gradients.append(current_grad.copy())
            
            # Check convergence
            grad_norm = np.linalg.norm(current_grad)
            if grad_norm < tolerance:
                print(f"  ✅ Converged after {iteration} iterations (gradient norm: {grad_norm:.8f})")
                break
            
            # Gradient descent step
            x = x - learning_rate * current_grad
            path.append(x.copy())
            
            # Adaptive learning rate (if we're not making progress)
            if iteration > 10 and len(function_values) > 10:
                if function_values[-1] > function_values[-10]:  # Not improving
                    learning_rate *= 0.9  # Reduce learning rate
                    print(f"  📉 Reducing learning rate to {learning_rate:.6f}")
        
        results = {
            'final_point': x,
            'final_value': f(x),
            'final_gradient': self.compute_numerical_gradient(f, x),
            'path': np.array(path),
            'function_values': function_values,
            'gradients': gradients,
            'iterations': len(path) - 1,
            'converged': grad_norm < tolerance
        }
        
        print(f"  🎯 Final point: {x}")
        print(f"  📊 Final bias value: {results['final_value']:.6f}")
        
        return results

print("✅ Gradient Analysis methods implemented!")

# ============================================================================
# CELL 4: Differential Geometry Implementation
# ============================================================================
class DifferentialGeometry:
    """
    Differential geometry methods for analyzing bias on demographic manifolds.
    
    EXPLANATION FOR BEGINNERS:
    - Think of demographic groups as points on a curved surface (manifold)
    - Different groups are at different locations on this surface
    - We measure distances and curvature on this surface to understand bias
    """
    
    def __init__(self, analyzer: AdvancedBiasAnalyzer):
        self.analyzer = analyzer
        self.metric_tensor = None
        
    def compute_riemannian_metric_tensor(self, bias_function: Callable, 
                                       demographic_points: np.ndarray) -> np.ndarray:
        """
        Compute Riemannian metric tensor for measuring distances on bias manifold.
        
        EXPLANATION FOR BEGINNERS:
        - A metric tensor tells us how to measure distances on curved surfaces
        - Like how GPS measures distances on Earth's curved surface vs flat map
        - It accounts for the "curvature" of the bias landscape
        
        Mathematical: G[i,j] = ∂²f/∂xᵢ∂xⱼ (Hessian of bias function)
        
        Args:
            bias_function: Function that gives bias at any demographic point
            demographic_points: Array of demographic coordinates
            
        Returns:
            Metric tensor matrix
        """
        print("🌐 Computing Riemannian metric tensor...")
        
        # Use average point as reference
        center_point = np.mean(demographic_points, axis=0)
        
        # Compute Hessian at center point (this becomes our metric tensor)
        hessian = self.analyzer.compute_hessian_matrix(bias_function, center_point)
        
        # Ensure positive definiteness (required for valid metric)
        eigenvals, eigenvecs = np.linalg.eigh(hessian)
        eigenvals = np.maximum(eigenvals, self.analyzer.epsilon)  # Make positive
        self.metric_tensor = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        print(f"  ✅ Metric tensor computed: condition number = {np.linalg.cond(self.metric_tensor):.2f}")
        return self.metric_tensor
    
    def compute_geodesic_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute geodesic distance between two demographic groups.
        
        EXPLANATION FOR BEGINNERS:
        - Geodesic = shortest path on a curved surface (like great circle on Earth)
        - Regular distance = straight line through space
        - Geodesic distance = path following the surface curvature
        - This gives us the "true" bias distance between demographic groups
        
        Mathematical: d = √[(p₂-p₁)ᵀ G (p₂-p₁)]
        
        Args:
            point1, point2: Demographic coordinates
            
        Returns:
            Geodesic distance
        """
        if self.metric_tensor is None:
            # Fallback to Euclidean distance
            return np.linalg.norm(point2 - point1)
        
        diff = point2 - point1
        distance = np.sqrt(diff.T @ self.metric_tensor @ diff)
        
        return float(distance)
    
    def compute_curvature_analysis(self, bias_function: Callable, 
                                 demographic_points: np.ndarray) -> Dict[str, float]:
        """
        Analyze curvature of the bias manifold.
        
        EXPLANATION FOR BEGINNERS:
        - Curvature tells us how "bent" the bias surface is
        - Flat surface = no bias variation, curved = complex bias patterns
        - Positive curvature = bowl-shaped (bias has clear minimum)
        - Negative curvature = saddle-shaped (bias has complex structure)
        
        Args:
            bias_function: Bias function
            demographic_points: Points to analyze
            
        Returns:
            Curvature measures
        """
        print("🌊 Computing curvature analysis...")
        
        curvatures = []
        
        for point in demographic_points:
            hessian = self.analyzer.compute_hessian_matrix(bias_function, point)
            eigenvals = np.linalg.eigvals(hessian)
            
            # Gaussian curvature (product of eigenvalues)
            gaussian_curvature = np.prod(eigenvals) if len(eigenvals) >= 2 else 0
            
            # Mean curvature (average of eigenvalues)
            mean_curvature = np.mean(eigenvals)
            
            curvatures.append({
                'gaussian': gaussian_curvature,
                'mean': mean_curvature,
                'eigenvalues': eigenvals
            })
        
        # Aggregate statistics
        gaussian_curvatures = [c['gaussian'] for c in curvatures]
        mean_curvatures = [c['mean'] for c in curvatures]
        
        analysis = {
            'mean_gaussian_curvature': np.mean(gaussian_curvatures),
            'std_gaussian_curvature': np.std(gaussian_curvatures),
            'mean_mean_curvature': np.mean(mean_curvatures),
            'std_mean_curvature': np.std(mean_curvatures),
            'max_curvature': np.max([np.max(np.abs(c['eigenvalues'])) for c in curvatures]),
            'curvature_complexity': np.std(gaussian_curvatures) + np.std(mean_curvatures)
        }
        
        print(f"  ✅ Curvature analysis complete:")
        print(f"    Mean Gaussian curvature: {analysis['mean_gaussian_curvature']:.6f}")
        print(f"    Curvature complexity: {analysis['curvature_complexity']:.6f}")
        
        return analysis
    
    def parallel_transport_bias_vector(self, vector: np.ndarray, 
                                     start_point: np.ndarray, 
                                     end_point: np.ndarray) -> np.ndarray:
        """
        Parallel transport a bias vector along the manifold.
        
        EXPLANATION FOR BEGINNERS:
        - Parallel transport = moving a vector along a curved surface while keeping it "parallel"
        - Like carrying a compass along Earth's surface - the needle direction changes
        - This tells us how bias "directions" change as we move between demographic groups
        
        Args:
            vector: Bias vector to transport
            start_point: Starting demographic coordinates
            end_point: Ending demographic coordinates
            
        Returns:
            Transported vector
        """
        if self.metric_tensor is None:
            return vector  # No transport without metric
        
        # Simplified parallel transport (first-order approximation)
        path_vector = end_point - start_point
        
        # Christoffel symbols approximation (connection coefficients)
        # This is a simplified version - full implementation would need tensor calculus
        
        # For small displacements, parallel transport ≈ identity + correction
        correction = -0.5 * self.metric_tensor @ path_vector
        transported_vector = vector + correction
        
        return transported_vector

print("✅ Differential Geometry methods implemented!")

# ============================================================================
# CELL 5: Optimization Theory Implementation  
# ============================================================================
class OptimizationTheory:
    """
    Optimization theory methods for fairness-constrained bias minimization.
    
    EXPLANATION FOR BEGINNERS:
    - We want to minimize bias while keeping accuracy high
    - This is like trying to make a cake that's both delicious AND healthy
    - We use mathematical optimization to find the best balance
    """
    
    def __init__(self, analyzer: AdvancedBiasAnalyzer):
        self.analyzer = analyzer
        
    def lagrange_multiplier_optimization(self, objective_func: Callable,
                                       constraint_funcs: List[Callable],
                                       x0: np.ndarray) -> Dict[str, Any]:
        """
        Solve constrained optimization using Lagrange multipliers.
        
        EXPLANATION FOR BEGINNERS:
        - Lagrange multipliers solve problems like "minimize cost subject to quality constraints"
        - We want to minimize bias (objective) while keeping accuracy above threshold (constraint)
        - The method finds the point where objective and constraint gradients are parallel
        
        Mathematical: ∇f = λ∇g (gradient of objective = λ × gradient of constraint)
        
        Args:
            objective_func: Function to minimize (bias)
            constraint_funcs: List of constraint functions (accuracy requirements)
            x0: Starting point
            
        Returns:
            Optimization results
        """
        print("🎯 Solving constrained optimization with Lagrange multipliers...")
        
        def lagrangian(x_and_lambdas):
            """
            Lagrangian function: L(x,λ) = f(x) + Σλᵢgᵢ(x)
            """
            n_vars = len(x0)
            x = x_and_lambdas[:n_vars]
            lambdas = x_and_lambdas[n_vars:]
            
            # Objective function
            L = objective_func(x)
            
            # Add constraint terms
            for i, constraint_func in enumerate(constraint_funcs):
                if i < len(lambdas):
                    L += lambdas[i] * constraint_func(x)
            
            return L
        
        # Initial guess: x0 + small lambda values
        initial_guess = np.concatenate([x0, np.ones(len(constraint_funcs)) * 0.1])
        
        try:
            # Minimize the Lagrangian
            result = optimize.minimize(lagrangian, initial_guess, method='BFGS')
            
            n_vars = len(x0)
            optimal_x = result.x[:n_vars]
            optimal_lambdas = result.x[n_vars:]
            
            # Evaluate final values
            final_objective = objective_func(optimal_x)
            final_constraints = [f(optimal_x) for f in constraint_funcs]
            
            results = {
                'optimal_point': optimal_x,
                'optimal_lambdas': optimal_lambdas,
                'final_objective_value': final_objective,
                'constraint_values': final_constraints,
                'optimization_success': result.success,
                'optimization_message': result.message
            }
            
            print(f"  ✅ Optimization {'succeeded' if result.success else 'failed'}")
            print(f"  📊 Final objective value: {final_objective:.6f}")
            print(f"  🎯 Constraint violations: {[abs(c) for c in final_constraints]}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Optimization failed: {e}")
            return {'error': str(e)}
    
    def multi_objective_optimization(self, objective_funcs: List[Callable],
                                   x0: np.ndarray, 
                                   weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Multi-objective optimization (accuracy vs fairness).
        
        EXPLANATION FOR BEGINNERS:
        - Sometimes we have multiple goals that conflict (accuracy vs fairness)
        - Multi-objective optimization finds the best compromises
        - Like choosing a car: you want fast, cheap, and reliable, but can't maximize all three
        
        Args:
            objective_funcs: List of functions to optimize [bias_func, accuracy_func]
            x0: Starting point
            weights: Importance weights for each objective
            
        Returns:
            Multi-objective optimization results
        """
        print("⚖️ Running multi-objective optimization (accuracy vs fairness)...")
        
        if weights is None:
            weights = [1.0] * len(objective_funcs)  # Equal weights
        
        def weighted_objective(x):
            """
            Weighted sum of objectives: f(x) = Σwᵢfᵢ(x)
            """
            total = 0
            for i, (func, weight) in enumerate(zip(objective_funcs, weights)):
                total += weight * func(x)
            return total
        
        # Optimize weighted sum
        result = optimize.minimize(weighted_objective, x0, method='BFGS')
        
        # Evaluate individual objectives at optimal point
        individual_values = [f(result.x) for f in objective_funcs]
        
        results = {
            'optimal_point': result.x,
            'weighted_objective_value': result.fun,
            'individual_objective_values': individual_values,
            'weights_used': weights,
            'optimization_success': result.success
        }
        
        print(f"  ✅ Multi-objective optimization complete")
        print(f"  📊 Individual objective values: {individual_values}")
        
        return results
    
    def compute_pareto_frontier(self, objective_funcs: List[Callable],
                              x0: np.ndarray, 
                              n_points: int = 20) -> Dict[str, Any]:
        """
        Compute Pareto frontier for accuracy-fairness trade-offs.
        
        EXPLANATION FOR BEGINNERS:
        - Pareto frontier = all the "best possible" trade-offs
        - Like efficient cars: for each level of speed, what's the best fuel economy possible?
        - Shows the fundamental trade-off between accuracy and fairness
        
        Args:
            objective_funcs: [bias_function, accuracy_function]
            x0: Starting point
            n_points: Number of points on frontier
            
        Returns:
            Pareto frontier points and analysis
        """
        print(f"📈 Computing Pareto frontier with {n_points} points...")
        
        pareto_points = []
        pareto_objectives = []
        
        # Generate different weight combinations
        for i in range(n_points):
            # Weight varies from [1,0] to [0,1]
            w1 = i / (n_points - 1)
            w2 = 1 - w1
            weights = [w1, w2]
            
            # Solve multi-objective problem with these weights
            result = self.multi_objective_optimization(objective_funcs, x0, weights)
            
            if result.get('optimization_success', False):
                pareto_points.append(result['optimal_point'])
                pareto_objectives.append(result['individual_objective_values'])
        
        pareto_points = np.array(pareto_points)
        pareto_objectives = np.array(pareto_objectives)
        
        # Analyze the frontier
        if len(pareto_objectives) > 0:
            # Find extreme points
            min_bias_idx = np.argmin(pareto_objectives[:, 0])  # Minimum bias
            max_accuracy_idx = np.argmin(pareto_objectives[:, 1])  # Maximum accuracy (min negative accuracy)
            
            analysis = {
                'pareto_points': pareto_points,
                'pareto_objectives': pareto_objectives,
                'min_bias_point': pareto_points[min_bias_idx],
                'min_bias_values': pareto_objectives[min_bias_idx],
                'max_accuracy_point': pareto_points[max_accuracy_idx],
                'max_accuracy_values': pareto_objectives[max_accuracy_idx],
                'frontier_length': len(pareto_points),
                'trade_off_slope': self._compute_trade_off_slope(pareto_objectives)
            }
            
            print(f"  ✅ Pareto frontier computed with {len(pareto_points)} points")
            print(f"  📊 Best bias: {analysis['min_bias_values'][0]:.4f}")
            print(f"  📊 Best accuracy: {-analysis['max_accuracy_values'][1]:.4f}")
            
            return analysis
        else:
            print("  ❌ Failed to compute Pareto frontier")
            return {'error': 'No valid Pareto points found'}
    
    def _compute_trade_off_slope(self, pareto_objectives: np.ndarray) -> float:
        """
        Compute the trade-off slope between objectives.
        
        Returns the rate at which we lose accuracy for each unit of bias reduction.
        """
        if len(pareto_objectives) < 2:
            return 0
        
        # Sort by first objective (bias)
        sorted_indices = np.argsort(pareto_objectives[:, 0])
        sorted_objectives = pareto_objectives[sorted_indices]
        
        # Compute average slope
        slopes = []
        for i in range(len(sorted_objectives) - 1):
            dx = sorted_objectives[i+1, 0] - sorted_objectives[i, 0]
            dy = sorted_objectives[i+1, 1] - sorted_objectives[i, 1]
            if abs(dx) > 1e-8:
                slopes.append(dy / dx)
        
        return np.mean(slopes) if slopes else 0

print("✅ Optimization Theory methods implemented!")

# ============================================================================
# CELL 6: Advanced Information Theory Implementation
# ============================================================================
class InformationTheory:
    """
    Advanced information theory methods with bias correction.
    
    EXPLANATION FOR BEGINNERS:
    - Information theory measures how much "information" leaks between variables
    - Like measuring how much knowing someone's demographic tells you about AI predictions
    - Lower mutual information = less bias (demographics don't predict outcomes)
    """
    
    def __init__(self, analyzer: AdvancedBiasAnalyzer):
        self.analyzer = analyzer
        self.epsilon = analyzer.epsilon
    
    def compute_mutual_information_with_bias_correction(self, X: np.ndarray, Y: np.ndarray, 
                                                      correction_method: str = 'miller_madow') -> float:
        """
        Compute bias-corrected mutual information I(X;Y).
        
        EXPLANATION FOR BEGINNERS:
        - Mutual information = how much knowing X tells us about Y
        - Raw calculation has bias (overestimates), so we apply corrections
        - Like adjusting survey results for sampling bias
        
        Mathematical: I(X;Y) = Σ p(x,y) log[p(x,y)/(p(x)p(y))]
        """
        print("📊 Computing bias-corrected mutual information...")
        
        # Discretize continuous variables if needed
        if X.dtype in [np.float32, np.float64]:
            X = self._discretize_variable(X)
        if Y.dtype in [np.float32, np.float64]:
            Y = self._discretize_variable(Y)
        
        # Compute joint and marginal probability distributions
        joint_counts = self._compute_joint_counts(X, Y)
        px = np.sum(joint_counts, axis=1) / np.sum(joint_counts)
        py = np.sum(joint_counts, axis=0) / np.sum(joint_counts)
        pxy = joint_counts / np.sum(joint_counts)
        
        # Raw mutual information
        mi_raw = 0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i,j] > self.epsilon and px[i] > self.epsilon and py[j] > self.epsilon:
                    mi_raw += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[j]))
        
        # Apply bias correction
        if correction_method == 'miller_madow':
            # Miller-Madow correction for finite sample bias
            n_samples = np.sum(joint_counts)
            n_bins_x = np.sum(px > 0)
            n_bins_y = np.sum(py > 0)
            correction = (n_bins_x - 1) * (n_bins_y - 1) / (2 * n_samples * np.log(2))
            mi_corrected = mi_raw - correction
        else:
            mi_corrected = mi_raw
        
        print(f"  ✅ MI computed: raw={mi_raw:.6f}, corrected={mi_corrected:.6f}")
        return max(0, mi_corrected)  # MI cannot be negative
    
    def compute_conditional_entropy_with_regularization(self, Y: np.ndarray, X: np.ndarray, 
                                                      regularization: float = 0.01) -> float:
        """
        Compute regularized conditional entropy H(Y|X).
        
        EXPLANATION FOR BEGINNERS:
        - Conditional entropy = uncertainty in Y after knowing X
        - High H(Y|X) = X doesn't help predict Y (good for fairness)
        - Low H(Y|X) = X strongly predicts Y (potential bias)
        - Regularization prevents overfitting to small samples
        """
        print("📊 Computing regularized conditional entropy...")
        
        # Discretize if needed
        if X.dtype in [np.float32, np.float64]:
            X = self._discretize_variable(X)
        if Y.dtype in [np.float32, np.float64]:
            Y = self._discretize_variable(Y)
        
        joint_counts = self._compute_joint_counts(X, Y)
        
        # Add regularization (Laplace smoothing)
        joint_counts_reg = joint_counts + regularization
        
        # Compute conditional probabilities
        px = np.sum(joint_counts_reg, axis=1)
        conditional_entropy = 0
        
        for i in range(joint_counts_reg.shape[0]):
            if px[i] > self.epsilon:
                p_x = px[i] / np.sum(joint_counts_reg)
                for j in range(joint_counts_reg.shape[1]):
                    p_y_given_x = joint_counts_reg[i,j] / px[i]
                    if p_y_given_x > self.epsilon:
                        conditional_entropy -= p_x * p_y_given_x * np.log2(p_y_given_x)
        
        print(f"  ✅ Conditional entropy H(Y|X): {conditional_entropy:.6f}")
        return conditional_entropy
    
    def compute_kl_divergence_with_regularization(self, P: np.ndarray, Q: np.ndarray, 
                                                regularization: float = 1e-8) -> float:
        """
        Compute regularized KL divergence D_KL(P||Q).
        
        EXPLANATION FOR BEGINNERS:
        - KL divergence measures how different two probability distributions are
        - Like comparing two different demographic prediction patterns
        - Higher KL = more different = more bias
        - Regularization prevents infinite values when probabilities are zero
        """
        # Add regularization to prevent log(0)
        P_reg = P + regularization
        Q_reg = Q + regularization
        
        # Normalize to ensure they're probability distributions
        P_reg = P_reg / np.sum(P_reg)
        Q_reg = Q_reg / np.sum(Q_reg)
        
        # Compute KL divergence
        kl_div = 0
        for i in range(len(P_reg)):
            if P_reg[i] > self.epsilon and Q_reg[i] > self.epsilon:
                kl_div += P_reg[i] * np.log2(P_reg[i] / Q_reg[i])
        
        return kl_div
    
    def compute_jensen_shannon_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence (symmetric version of KL divergence).
        
        EXPLANATION FOR BEGINNERS:
        - JS divergence is like KL divergence but symmetric (P vs Q same as Q vs P)
        - Bounded between 0 and 1, easier to interpret
        - Measures how different demographic group predictions are
        """
        # Compute average distribution
        M = (P + Q) / 2
        
        # JS divergence = 0.5 * [KL(P||M) + KL(Q||M)]
        js_div = 0.5 * (self.compute_kl_divergence_with_regularization(P, M) + 
                        self.compute_kl_divergence_with_regularization(Q, M))
        
        return js_div
    
    def _discretize_variable(self, X: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Discretize continuous variable into bins."""
        if len(np.unique(X)) <= n_bins:
            return X.astype(int)
        
        # Use quantile-based binning
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(X, quantiles)
        bin_edges[-1] += 1e-8  # Ensure last value is included
        
        return np.digitize(X, bin_edges) - 1
    
    def _compute_joint_counts(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute joint count matrix."""
        x_unique = np.unique(X)
        y_unique = np.unique(Y)
        
        joint_counts = np.zeros((len(x_unique), len(y_unique)))
        
        for i, x_val in enumerate(x_unique):
            for j, y_val in enumerate(y_unique):
                joint_counts[i,j] = np.sum((X == x_val) & (Y == y_val))
        
        return joint_counts

print("✅ Information Theory methods implemented!")
