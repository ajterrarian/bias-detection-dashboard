# ============================================================================
# DAY 3 AFTERNOON: MATHEMATICAL BIAS ANALYSIS FRAMEWORK
# Advanced mathematical implementation with differential geometry
# Copy each section into separate Colab cells after running the morning session
# ============================================================================

# ============================================================================
# CELL 8: Install Additional Mathematical Libraries
# ============================================================================
!pip install sympy autograd jax jaxlib
!pip install statsmodels scipy scikit-learn
!pip install pymc3 arviz theano-pymc
print("✅ Advanced mathematical libraries installed!")

# ============================================================================
# CELL 9: Mathematical BiasAnalyzer Class - Core Implementation
# ============================================================================
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sympy as sp
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

class BiasAnalyzer:
    """
    Comprehensive mathematical bias analyzer implementing differential geometry,
    information theory, and statistical significance testing for facial recognition bias.
    
    Mathematical Framework:
    - Bias Gradient: ∇f = ∂accuracy/∂demographics  
    - Riemannian Metrics: Distance measures on demographic manifold
    - Information Theory: H(Y|D) entropy calculations
    - Statistical Testing: Multiple comparisons with Bonferroni correction
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the mathematical bias analyzer.
        
        Args:
            significance_level: Alpha level for statistical tests (default: 0.05)
        """
        print("🔬 Initializing Mathematical BiasAnalyzer...")
        
        self.alpha = significance_level
        self.results_cache = {}
        self.demographic_manifold = None
        self.bias_gradients = {}
        
        # Mathematical constants and parameters
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.max_iterations = 1000  # For optimization algorithms
        self.convergence_tolerance = 1e-6
        
        print("✅ BiasAnalyzer initialized with mathematical framework")
    
    def load_face_recognition_results(self, results_file: str) -> pd.DataFrame:
        """
        Load and preprocess face recognition results for mathematical analysis.
        
        Args:
            results_file: Path to JSON results file from FaceRecognitionTester
            
        Returns:
            Processed DataFrame with demographic and accuracy data
        """
        print(f"📊 Loading face recognition results from {results_file}...")
        
        import json
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract results data
        results = data.get('results', [])
        
        # Convert to structured DataFrame
        processed_data = []
        
        for result in results:
            if not result.get('success', False):
                continue
                
            demographic_info = result.get('demographic_info', {})
            api_results = result.get('api_results', {})
            
            # Extract accuracy metrics from each API
            row_data = {
                'image_id': result.get('image_id'),
                'demographic_group': demographic_info.get('group', 'unknown'),
                'age_category': demographic_info.get('age_category', 'unknown'),
                'gender': demographic_info.get('gender', 'unknown'),
            }
            
            # AWS metrics
            if 'aws' in api_results and api_results['aws'].get('success'):
                aws_data = api_results['aws']
                faces = aws_data.get('faces', [])
                if faces:
                    row_data['aws_confidence'] = np.mean([face['confidence'] for face in faces])
                    row_data['aws_face_count'] = len(faces)
                    row_data['aws_detected'] = 1
                else:
                    row_data['aws_confidence'] = 0
                    row_data['aws_face_count'] = 0
                    row_data['aws_detected'] = 0
            else:
                row_data['aws_confidence'] = 0
                row_data['aws_face_count'] = 0
                row_data['aws_detected'] = 0
            
            # Google metrics
            if 'google' in api_results and api_results['google'].get('success'):
                google_data = api_results['google']
                faces = google_data.get('faces', [])
                if faces:
                    row_data['google_confidence'] = np.mean([face['confidence'] for face in faces])
                    row_data['google_face_count'] = len(faces)
                    row_data['google_detected'] = 1
                else:
                    row_data['google_confidence'] = 0
                    row_data['google_face_count'] = 0
                    row_data['google_detected'] = 0
            else:
                row_data['google_confidence'] = 0
                row_data['google_face_count'] = 0
                row_data['google_detected'] = 0
            
            processed_data.append(row_data)
        
        df = pd.DataFrame(processed_data)
        
        # Create composite accuracy measures
        df['combined_confidence'] = (df['aws_confidence'] + df['google_confidence']) / 2
        df['combined_detection'] = ((df['aws_detected'] + df['google_detected']) > 0).astype(int)
        
        print(f"✅ Loaded {len(df)} processed results")
        print(f"📊 Demographic groups: {df['demographic_group'].unique()}")
        
        return df
    
    def compute_bias_gradients(self, df: pd.DataFrame, 
                             accuracy_column: str = 'combined_confidence') -> Dict[str, np.ndarray]:
        """
        Compute bias gradients using numerical differentiation.
        
        Mathematical Formula: ∇f = ∂accuracy/∂demographics
        
        Args:
            df: DataFrame with results
            accuracy_column: Column name for accuracy measure
            
        Returns:
            Dictionary of gradient vectors for each demographic dimension
        """
        print("🔬 Computing bias gradients using numerical differentiation...")
        
        # Create demographic encoding
        demographic_features = ['age_category', 'gender']
        encoded_demographics = pd.get_dummies(df[demographic_features])
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(encoded_demographics)
        y = df[accuracy_column].values
        
        # Compute gradients using finite differences
        gradients = {}
        h = 1e-5  # Step size for numerical differentiation
        
        for i, feature in enumerate(encoded_demographics.columns):
            # Forward difference approximation
            X_forward = X.copy()
            X_forward[:, i] += h
            
            X_backward = X.copy()
            X_backward[:, i] -= h
            
            # Fit polynomial surfaces to approximate the function
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Predict at forward and backward points
            y_forward = model.predict(poly.transform(X_forward))
            y_backward = model.predict(poly.transform(X_backward))
            
            # Compute gradient
            gradient = (y_forward - y_backward) / (2 * h)
            gradients[feature] = gradient
            
            print(f"  ∇{feature}: mean={np.mean(gradient):.6f}, std={np.std(gradient):.6f}")
        
        self.bias_gradients = gradients
        
        # Compute gradient magnitude (overall bias intensity)
        gradient_magnitude = np.sqrt(sum(grad**2 for grad in gradients.values()))
        
        print(f"✅ Bias gradients computed")
        print(f"📊 Overall gradient magnitude: {np.mean(gradient_magnitude):.6f}")
        
        return gradients
    
    def compute_riemannian_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Riemannian metrics for measuring bias "distance" on demographic manifold.
        
        Mathematical Framework:
        - Treats demographic space as a Riemannian manifold
        - Computes geodesic distances between demographic groups
        - Uses metric tensor to measure bias curvature
        
        Args:
            df: DataFrame with results
            
        Returns:
            Dictionary of Riemannian distance measures
        """
        print("🌐 Computing Riemannian metrics on demographic manifold...")
        
        # Create demographic coordinate system
        demographic_coords = []
        group_labels = []
        
        for group in df['demographic_group'].unique():
            group_data = df[df['demographic_group'] == group]
            
            # Map to coordinate system
            age_coord = {'young': 0, 'middle_aged': 1, 'elderly': 2}.get(
                group_data['age_category'].iloc[0], 1)
            gender_coord = {'male': 0, 'female': 1}.get(
                group_data['gender'].iloc[0], 0.5)
            
            demographic_coords.append([age_coord, gender_coord])
            group_labels.append(group)
        
        demographic_coords = np.array(demographic_coords)
        
        # Compute accuracy surface over demographic manifold
        accuracy_values = []
        for group in group_labels:
            group_accuracy = df[df['demographic_group'] == group]['combined_confidence'].mean()
            accuracy_values.append(group_accuracy)
        
        accuracy_values = np.array(accuracy_values)
        
        # Compute metric tensor components
        # G_ij = ∂²f/∂x^i∂x^j (Hessian of accuracy function)
        
        def accuracy_function(coords):
            """Interpolate accuracy at given coordinates."""
            from scipy.interpolate import griddata
            return griddata(demographic_coords, accuracy_values, coords, method='linear', fill_value=0)
        
        # Compute Hessian matrix (metric tensor)
        hessian = np.zeros((2, 2))
        h = 1e-4
        
        for i in range(2):
            for j in range(2):
                # Compute second partial derivatives
                coords_pp = demographic_coords.mean(axis=0).copy()
                coords_pm = demographic_coords.mean(axis=0).copy()
                coords_mp = demographic_coords.mean(axis=0).copy()
                coords_mm = demographic_coords.mean(axis=0).copy()
                
                coords_pp[i] += h
                coords_pp[j] += h
                
                coords_pm[i] += h
                coords_pm[j] -= h
                
                coords_mp[i] -= h
                coords_mp[j] += h
                
                coords_mm[i] -= h
                coords_mm[j] -= h
                
                # Mixed partial derivative
                f_pp = accuracy_function(coords_pp.reshape(1, -1))[0] if not np.isnan(accuracy_function(coords_pp.reshape(1, -1))[0]) else 0
                f_pm = accuracy_function(coords_pm.reshape(1, -1))[0] if not np.isnan(accuracy_function(coords_pm.reshape(1, -1))[0]) else 0
                f_mp = accuracy_function(coords_mp.reshape(1, -1))[0] if not np.isnan(accuracy_function(coords_mp.reshape(1, -1))[0]) else 0
                f_mm = accuracy_function(coords_mm.reshape(1, -1))[0] if not np.isnan(accuracy_function(coords_mm.reshape(1, -1))[0]) else 0
                
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
        
        # Compute Riemannian distances between all pairs of demographic groups
        riemannian_distances = {}
        
        for i, group1 in enumerate(group_labels):
            for j, group2 in enumerate(group_labels[i+1:], i+1):
                # Geodesic distance approximation
                coord_diff = demographic_coords[j] - demographic_coords[i]
                
                # Distance using metric tensor: d = sqrt(dx^T G dx)
                distance = np.sqrt(coord_diff.T @ hessian @ coord_diff) if np.all(np.linalg.eigvals(hessian) > 0) else np.linalg.norm(coord_diff)
                
                riemannian_distances[f"{group1}_to_{group2}"] = abs(distance)
        
        # Compute curvature measures
        try:
            eigenvals = np.linalg.eigvals(hessian)
            gaussian_curvature = np.prod(eigenvals) if len(eigenvals) == 2 else 0
            mean_curvature = np.mean(eigenvals)
        except:
            gaussian_curvature = 0
            mean_curvature = 0
        
        metrics = {
            'max_riemannian_distance': max(riemannian_distances.values()) if riemannian_distances else 0,
            'mean_riemannian_distance': np.mean(list(riemannian_distances.values())) if riemannian_distances else 0,
            'gaussian_curvature': gaussian_curvature,
            'mean_curvature': mean_curvature,
            'pairwise_distances': riemannian_distances
        }
        
        print(f"✅ Riemannian metrics computed")
        print(f"📊 Max geodesic distance: {metrics['max_riemannian_distance']:.6f}")
        print(f"🌐 Gaussian curvature: {metrics['gaussian_curvature']:.6f}")
        
        return metrics
    
    def compute_information_theoretic_measures(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute information-theoretic bias measures.
        
        Mathematical Framework:
        - Conditional Entropy: H(Y|D) = -Σ p(y,d) log p(y|d)
        - Mutual Information: I(Y;D) = H(Y) - H(Y|D)
        - Differential Privacy measures
        
        Args:
            df: DataFrame with results
            
        Returns:
            Dictionary of information-theoretic measures
        """
        print("📊 Computing information-theoretic bias measures...")
        
        # Discretize accuracy for entropy calculations
        accuracy_bins = pd.cut(df['combined_confidence'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Compute joint and conditional probabilities
        joint_probs = pd.crosstab(accuracy_bins, df['demographic_group'], normalize=True)
        marginal_accuracy = joint_probs.sum(axis=1)
        marginal_demographic = joint_probs.sum(axis=0)
        
        # Conditional entropy H(Y|D)
        conditional_entropy = 0
        for demo_group in joint_probs.columns:
            for acc_level in joint_probs.index:
                p_joint = joint_probs.loc[acc_level, demo_group]
                p_demo = marginal_demographic[demo_group]
                
                if p_joint > self.epsilon and p_demo > self.epsilon:
                    p_conditional = p_joint / p_demo
                    if p_conditional > self.epsilon:
                        conditional_entropy -= p_joint * np.log2(p_conditional)
        
        # Marginal entropy H(Y)
        marginal_entropy = 0
        for acc_level in marginal_accuracy.index:
            p_acc = marginal_accuracy[acc_level]
            if p_acc > self.epsilon:
                marginal_entropy -= p_acc * np.log2(p_acc)
        
        # Mutual information I(Y;D) = H(Y) - H(Y|D)
        mutual_information = marginal_entropy - conditional_entropy
        
        # Normalized mutual information
        demographic_entropy = 0
        for demo_group in marginal_demographic.index:
            p_demo = marginal_demographic[demo_group]
            if p_demo > self.epsilon:
                demographic_entropy -= p_demo * np.log2(p_demo)
        
        normalized_mutual_info = mutual_information / max(marginal_entropy, demographic_entropy, self.epsilon)
        
        # Information gain ratio
        info_gain_ratio = mutual_information / demographic_entropy if demographic_entropy > self.epsilon else 0
        
        measures = {
            'conditional_entropy': conditional_entropy,
            'marginal_entropy': marginal_entropy,
            'mutual_information': mutual_information,
            'normalized_mutual_information': normalized_mutual_info,
            'information_gain_ratio': info_gain_ratio,
            'demographic_entropy': demographic_entropy
        }
        
        print(f"✅ Information-theoretic measures computed")
        print(f"📊 Mutual Information I(Y;D): {mutual_information:.6f}")
        print(f"📊 Conditional Entropy H(Y|D): {conditional_entropy:.6f}")
        
        return measures
    
    def perform_statistical_significance_testing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive statistical significance testing with multiple comparisons correction.
        
        Tests:
        - ANOVA for overall group differences
        - Pairwise t-tests with Bonferroni correction
        - Kruskal-Wallis for non-parametric testing
        - Bootstrap confidence intervals
        
        Args:
            df: DataFrame with results
            
        Returns:
            Dictionary of statistical test results
        """
        print("📈 Performing statistical significance testing...")
        
        # Group data by demographics
        groups = {}
        for demo_group in df['demographic_group'].unique():
            groups[demo_group] = df[df['demographic_group'] == demo_group]['combined_confidence'].values
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        
        if len(groups) < 2:
            print("⚠️ Insufficient groups for statistical testing")
            return {'error': 'Insufficient groups'}
        
        # 1. One-way ANOVA
        group_values = list(groups.values())
        f_stat, p_value_anova = stats.f_oneway(*group_values)
        
        # 2. Kruskal-Wallis test (non-parametric)
        h_stat, p_value_kruskal = stats.kruskal(*group_values)
        
        # 3. Pairwise t-tests with Bonferroni correction
        group_names = list(groups.keys())
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        bonferroni_alpha = self.alpha / n_comparisons
        
        pairwise_results = {}
        significant_pairs = []
        
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i+1:], i+1):
                t_stat, p_val = stats.ttest_ind(groups[group1], groups[group2])
                
                pairwise_results[f"{group1}_vs_{group2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'bonferroni_significant': p_val < bonferroni_alpha,
                    'effect_size': abs(np.mean(groups[group1]) - np.mean(groups[group2])) / 
                                 np.sqrt((np.var(groups[group1]) + np.var(groups[group2])) / 2)
                }
                
                if p_val < bonferroni_alpha:
                    significant_pairs.append(f"{group1}_vs_{group2}")
        
        # 4. Bootstrap confidence intervals
        bootstrap_results = {}
        n_bootstrap = 1000
        
        for group_name, group_data in groups.items():
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(group_data, size=len(group_data), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            
            bootstrap_results[group_name] = {
                'mean': np.mean(group_data),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower
            }
        
        # 5. Effect size measures
        # Eta-squared for ANOVA
        ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(group_values)))**2 
                        for group in group_values)
        ss_total = sum((x - np.mean(np.concatenate(group_values)))**2 
                      for group in group_values for x in group)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        results = {
            'anova': {
                'f_statistic': f_stat,
                'p_value': p_value_anova,
                'significant': p_value_anova < self.alpha,
                'eta_squared': eta_squared
            },
            'kruskal_wallis': {
                'h_statistic': h_stat,
                'p_value': p_value_kruskal,
                'significant': p_value_kruskal < self.alpha
            },
            'pairwise_tests': pairwise_results,
            'bonferroni_correction': {
                'original_alpha': self.alpha,
                'corrected_alpha': bonferroni_alpha,
                'n_comparisons': n_comparisons,
                'significant_pairs': significant_pairs
            },
            'bootstrap_confidence_intervals': bootstrap_results
        }
        
        print(f"✅ Statistical testing completed")
        print(f"📊 ANOVA p-value: {p_value_anova:.6f} ({'significant' if p_value_anova < self.alpha else 'not significant'})")
        print(f"📊 Significant pairwise comparisons: {len(significant_pairs)}/{n_comparisons}")
        
        return results

print("✅ Mathematical BiasAnalyzer class created successfully!")

# ============================================================================
# CELL 10: Comprehensive Mathematical Analysis Runner
# ============================================================================
async def run_comprehensive_mathematical_analysis(results_file: str):
    """
    Run the complete mathematical analysis pipeline.
    
    Args:
        results_file: Path to the face recognition results JSON file
    """
    print("🔬 Starting Comprehensive Mathematical Bias Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = BiasAnalyzer(significance_level=0.05)
    
    # Load results
    print("\n📊 Step 1: Loading and preprocessing data...")
    df = analyzer.load_face_recognition_results(results_file)
    
    # Compute bias gradients
    print("\n🔬 Step 2: Computing bias gradients...")
    gradients = analyzer.compute_bias_gradients(df)
    
    # Compute Riemannian metrics
    print("\n🌐 Step 3: Computing Riemannian metrics...")
    riemannian_metrics = analyzer.compute_riemannian_metrics(df)
    
    # Compute information-theoretic measures
    print("\n📊 Step 4: Computing information-theoretic measures...")
    info_measures = analyzer.compute_information_theoretic_measures(df)
    
    # Perform statistical testing
    print("\n📈 Step 5: Performing statistical significance testing...")
    statistical_results = analyzer.perform_statistical_significance_testing(df)
    
    # Compile comprehensive results
    comprehensive_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_images': len(df),
            'n_demographic_groups': df['demographic_group'].nunique(),
            'significance_level': analyzer.alpha
        },
        'bias_gradients': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in gradients.items()},
        'riemannian_metrics': riemannian_metrics,
        'information_theory': info_measures,
        'statistical_tests': statistical_results,
        'summary_scores': {}
    }
    
    # Calculate overall bias severity score
    gradient_magnitude = np.mean([np.mean(np.abs(grad)) for grad in gradients.values()])
    riemannian_distance = riemannian_metrics['max_riemannian_distance']
    mutual_info = info_measures['mutual_information']
    
    # Composite bias score (normalized 0-1)
    bias_score = min(1.0, (gradient_magnitude * 10 + riemannian_distance * 5 + mutual_info * 2) / 3)
    
    # Bias severity classification
    if bias_score < 0.1:
        severity = "Minimal"
    elif bias_score < 0.3:
        severity = "Low"
    elif bias_score < 0.6:
        severity = "Moderate"
    elif bias_score < 0.8:
        severity = "High"
    else:
        severity = "Severe"
    
    comprehensive_results['summary_scores'] = {
        'overall_bias_score': bias_score,
        'bias_severity': severity,
        'gradient_magnitude': gradient_magnitude,
        'max_riemannian_distance': riemannian_distance,
        'mutual_information': mutual_info
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/mathematical_bias_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE MATHEMATICAL BIAS ANALYSIS RESULTS")
    print("=" * 60)
    print(f"📊 Overall Bias Score: {bias_score:.4f}")
    print(f"🎯 Bias Severity: {severity}")
    print(f"📈 Gradient Magnitude: {gradient_magnitude:.6f}")
    print(f"🌐 Max Riemannian Distance: {riemannian_distance:.6f}")
    print(f"📊 Mutual Information: {mutual_info:.6f}")
    
    if 'anova' in statistical_results:
        anova_sig = "Yes" if statistical_results['anova']['significant'] else "No"
        print(f"📈 Statistically Significant Bias: {anova_sig}")
        print(f"📊 Effect Size (η²): {statistical_results['anova']['eta_squared']:.4f}")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return comprehensive_results, output_file

print("✅ Mathematical analysis framework ready!")
print("🚀 Ready to run comprehensive analysis on your face recognition results!")

# ============================================================================
# CELL 11: Execute Complete Mathematical Analysis
# ============================================================================
# Run the comprehensive mathematical analysis
print("⏳ Running comprehensive mathematical bias analysis...")

# Use the results file from the morning session
import glob
results_files = glob.glob('./results/api_test_results_*.json')

if results_files:
    latest_results_file = max(results_files)  # Get the most recent file
    print(f"📊 Using results file: {latest_results_file}")
    
    # Run the analysis
    mathematical_results, output_file = await run_comprehensive_mathematical_analysis(latest_results_file)
    
    print("\n🎉 DAY 3 COMPLETE!")
    print("✅ Advanced API integration implemented")
    print("✅ Mathematical framework with differential geometry")
    print("✅ Information theory measures computed")
    print("✅ Statistical significance testing performed")
    print("✅ Riemannian metrics on demographic manifold")
    
else:
    print("❌ No results files found. Please run the morning session first.")
    print("💡 Make sure to run cells 1-7 before running the mathematical analysis.")
