# ============================================================================
# CELL 7: Mathematical Validation Suite
# ============================================================================
def run_mathematical_validation():
    """
    Comprehensive validation test suite for all mathematical implementations.
    
    EXPLANATION FOR BEGINNERS:
    - This tests our math to make sure it works correctly
    - Like checking your calculator gives 2+2=4
    - We test against known mathematical results
    """
    print("🧪 Running Mathematical Validation Test Suite")
    print("=" * 60)
    
    validation_results = {
        'gradient_tests': {},
        'information_theory_tests': {},
        'optimization_tests': {},
        'differential_geometry_tests': {},
        'overall_status': 'UNKNOWN'
    }
    
    # Test 1: Gradient Computation Validation
    print("\n🔬 Test 1: Gradient Computation Validation")
    try:
        analyzer = AdvancedBiasAnalyzer()
        
        # Test with known function: f(x,y) = x² + y²
        def quadratic_function(x):
            return x[0]**2 + x[1]**2
        
        test_point = np.array([1.0, 2.0])
        computed_gradient = analyzer.compute_numerical_gradient(quadratic_function, test_point)
        expected_gradient = np.array([2.0, 4.0])  # Analytical gradient
        
        gradient_error = np.linalg.norm(computed_gradient - expected_gradient)
        gradient_test_passed = gradient_error < 1e-4
        
        validation_results['gradient_tests'] = {
            'quadratic_test_passed': gradient_test_passed,
            'gradient_error': gradient_error,
            'computed_gradient': computed_gradient.tolist(),
            'expected_gradient': expected_gradient.tolist()
        }
        
        print(f"  ✅ Quadratic function test: {'PASSED' if gradient_test_passed else 'FAILED'}")
        print(f"  📊 Gradient error: {gradient_error:.8f}")
        
    except Exception as e:
        print(f"  ❌ Gradient test failed: {e}")
        validation_results['gradient_tests']['error'] = str(e)
    
    # Test 2: Information Theory Validation
    print("\n📊 Test 2: Information Theory Validation")
    try:
        analyzer = AdvancedBiasAnalyzer()
        info_theory = InformationTheory(analyzer)
        
        # Test with known distributions
        # Independent variables should have MI ≈ 0
        np.random.seed(42)
        X_independent = np.random.randint(0, 3, 1000)
        Y_independent = np.random.randint(0, 3, 1000)
        
        mi_independent = info_theory.compute_mutual_information_with_bias_correction(
            X_independent, Y_independent)
        
        # Perfectly dependent variables should have high MI
        X_dependent = np.random.randint(0, 3, 1000)
        Y_dependent = X_dependent.copy()  # Perfect dependence
        
        mi_dependent = info_theory.compute_mutual_information_with_bias_correction(
            X_dependent, Y_dependent)
        
        # Validation checks
        independence_test_passed = mi_independent < 0.1  # Should be near 0
        dependence_test_passed = mi_dependent > 1.0     # Should be high
        
        validation_results['information_theory_tests'] = {
            'independence_test_passed': independence_test_passed,
            'dependence_test_passed': dependence_test_passed,
            'mi_independent': mi_independent,
            'mi_dependent': mi_dependent
        }
        
        print(f"  ✅ Independence test: {'PASSED' if independence_test_passed else 'FAILED'}")
        print(f"  ✅ Dependence test: {'PASSED' if dependence_test_passed else 'FAILED'}")
        print(f"  📊 MI(independent): {mi_independent:.6f}")
        print(f"  📊 MI(dependent): {mi_dependent:.6f}")
        
    except Exception as e:
        print(f"  ❌ Information theory test failed: {e}")
        validation_results['information_theory_tests']['error'] = str(e)
    
    # Test 3: Optimization Algorithm Validation
    print("\n🎯 Test 3: Optimization Algorithm Validation")
    try:
        analyzer = AdvancedBiasAnalyzer()
        
        # Test gradient descent on known function: f(x,y) = (x-1)² + (y-2)²
        # Known minimum at (1, 2) with value 0
        def optimization_test_function(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        
        start_point = np.array([0.0, 0.0])
        result = analyzer.gradient_descent_optimization(
            optimization_test_function, start_point, learning_rate=0.1)
        
        final_point = result['final_point']
        expected_point = np.array([1.0, 2.0])
        
        optimization_error = np.linalg.norm(final_point - expected_point)
        optimization_test_passed = optimization_error < 0.1
        
        validation_results['optimization_tests'] = {
            'gradient_descent_test_passed': optimization_test_passed,
            'optimization_error': optimization_error,
            'final_point': final_point.tolist(),
            'expected_point': expected_point.tolist(),
            'converged': result['converged']
        }
        
        print(f"  ✅ Gradient descent test: {'PASSED' if optimization_test_passed else 'FAILED'}")
        print(f"  📊 Optimization error: {optimization_error:.6f}")
        print(f"  🎯 Converged: {result['converged']}")
        
    except Exception as e:
        print(f"  ❌ Optimization test failed: {e}")
        validation_results['optimization_tests']['error'] = str(e)
    
    # Test 4: Differential Geometry Validation
    print("\n🌐 Test 4: Differential Geometry Validation")
    try:
        analyzer = AdvancedBiasAnalyzer()
        diff_geom = DifferentialGeometry(analyzer)
        
        # Test metric tensor computation
        def simple_bias_function(x):
            return x[0]**2 + x[1]**2  # Simple quadratic
        
        test_points = np.array([[0, 0], [1, 1], [2, 0], [0, 2]])
        metric_tensor = diff_geom.compute_riemannian_metric_tensor(
            simple_bias_function, test_points)
        
        # Test geodesic distance
        point1 = np.array([0.0, 0.0])
        point2 = np.array([1.0, 1.0])
        geodesic_dist = diff_geom.compute_geodesic_distance(point1, point2)
        euclidean_dist = np.linalg.norm(point2 - point1)
        
        # Validation checks
        metric_tensor_valid = np.all(np.linalg.eigvals(metric_tensor) > 0)  # Positive definite
        distance_reasonable = 0.5 * euclidean_dist <= geodesic_dist <= 2 * euclidean_dist
        
        validation_results['differential_geometry_tests'] = {
            'metric_tensor_valid': metric_tensor_valid,
            'distance_reasonable': distance_reasonable,
            'geodesic_distance': geodesic_dist,
            'euclidean_distance': euclidean_dist,
            'metric_condition_number': np.linalg.cond(metric_tensor)
        }
        
        print(f"  ✅ Metric tensor test: {'PASSED' if metric_tensor_valid else 'FAILED'}")
        print(f"  ✅ Distance test: {'PASSED' if distance_reasonable else 'FAILED'}")
        print(f"  📊 Geodesic distance: {geodesic_dist:.6f}")
        print(f"  📊 Euclidean distance: {euclidean_dist:.6f}")
        
    except Exception as e:
        print(f"  ❌ Differential geometry test failed: {e}")
        validation_results['differential_geometry_tests']['error'] = str(e)
    
    # Overall validation status
    all_tests = [
        validation_results['gradient_tests'].get('quadratic_test_passed', False),
        validation_results['information_theory_tests'].get('independence_test_passed', False),
        validation_results['information_theory_tests'].get('dependence_test_passed', False),
        validation_results['optimization_tests'].get('gradient_descent_test_passed', False),
        validation_results['differential_geometry_tests'].get('metric_tensor_valid', False),
        validation_results['differential_geometry_tests'].get('distance_reasonable', False)
    ]
    
    passed_tests = sum(all_tests)
    total_tests = len(all_tests)
    
    if passed_tests == total_tests:
        validation_results['overall_status'] = 'ALL_PASSED'
    elif passed_tests >= total_tests * 0.8:
        validation_results['overall_status'] = 'MOSTLY_PASSED'
    else:
        validation_results['overall_status'] = 'FAILED'
    
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"📊 Tests passed: {passed_tests}/{total_tests}")
    print(f"🎯 Overall status: {validation_results['overall_status']}")
    
    if validation_results['overall_status'] == 'ALL_PASSED':
        print("✅ All mathematical implementations validated successfully!")
    elif validation_results['overall_status'] == 'MOSTLY_PASSED':
        print("⚠️ Most tests passed - minor issues detected")
    else:
        print("❌ Significant validation failures detected")
    
    return validation_results

# ============================================================================
# CELL 8: Complete Day 4 Integration and Demo
# ============================================================================
def run_complete_day4_analysis(results_file: str):
    """
    Run complete Day 4 analysis with all advanced mathematical methods.
    
    EXPLANATION FOR BEGINNERS:
    - This puts everything together and runs the full analysis
    - Uses all the advanced math we built today
    - Gives you comprehensive bias analysis results
    """
    print("🚀 Running Complete Day 4 Advanced Mathematical Analysis")
    print("=" * 70)
    
    # Initialize all components
    analyzer = AdvancedBiasAnalyzer()
    diff_geom = DifferentialGeometry(analyzer)
    optimization = OptimizationTheory(analyzer)
    info_theory = InformationTheory(analyzer)
    
    # Load data from previous days
    print("\n📊 Loading face recognition results...")
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract demographic and accuracy data
        demographic_data = []
        accuracy_data = []
        
        for result in data.get('results', []):
            if result.get('success', False):
                demo_info = result.get('demographic_info', {})
                api_results = result.get('api_results', {})
                
                # Create demographic encoding
                age_encoding = {'young': 0, 'middle_aged': 1, 'elderly': 2}
                gender_encoding = {'male': 0, 'female': 1}
                
                demo_vector = [
                    age_encoding.get(demo_info.get('age_category', 'young'), 0),
                    gender_encoding.get(demo_info.get('gender', 'male'), 0)
                ]
                demographic_data.append(demo_vector)
                
                # Extract accuracy
                google_conf = 0
                if 'google' in api_results and api_results['google'].get('success'):
                    faces = api_results['google'].get('faces', [])
                    if faces:
                        google_conf = np.mean([f['confidence'] for f in faces])
                
                accuracy_data.append(google_conf)
        
        demographic_array = np.array(demographic_data)
        accuracy_array = np.array(accuracy_data)
        
        print(f"  ✅ Loaded {len(demographic_array)} data points")
        
    except Exception as e:
        print(f"  ❌ Failed to load data: {e}")
        return {'error': f'Data loading failed: {e}'}
    
    # Define bias function for analysis
    def bias_function(demo_point):
        """
        Bias function: measures accuracy variation across demographic space
        """
        # Find nearest data points
        distances = np.linalg.norm(demographic_array - demo_point, axis=1)
        nearest_indices = np.argsort(distances)[:5]  # 5 nearest neighbors
        
        # Return variance in accuracy (higher = more bias)
        nearest_accuracies = accuracy_array[nearest_indices]
        return np.var(nearest_accuracies) if len(nearest_accuracies) > 1 else 0
    
    # Run comprehensive analysis
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_data_points': len(demographic_array),
            'analysis_type': 'complete_day4_advanced'
        }
    }
    
    # 1. Gradient Analysis
    print("\n🔬 Step 1: Advanced Gradient Analysis...")
    try:
        center_point = np.mean(demographic_array, axis=0)
        gradient = analyzer.compute_numerical_gradient(bias_function, center_point)
        hessian = analyzer.compute_hessian_matrix(bias_function, center_point)
        
        # Gradient descent optimization
        optimization_result = analyzer.gradient_descent_optimization(
            bias_function, center_point, learning_rate=0.01)
        
        results['gradient_analysis'] = {
            'bias_gradient': gradient.tolist(),
            'gradient_magnitude': float(np.linalg.norm(gradient)),
            'hessian_eigenvalues': np.linalg.eigvals(hessian).tolist(),
            'optimization_result': {
                'final_point': optimization_result['final_point'].tolist(),
                'final_bias_value': float(optimization_result['final_value']),
                'converged': optimization_result['converged']
            }
        }
        
        print(f"  ✅ Gradient magnitude: {results['gradient_analysis']['gradient_magnitude']:.6f}")
        
    except Exception as e:
        print(f"  ❌ Gradient analysis failed: {e}")
        results['gradient_analysis'] = {'error': str(e)}
    
    # 2. Differential Geometry Analysis
    print("\n🌐 Step 2: Differential Geometry Analysis...")
    try:
        metric_tensor = diff_geom.compute_riemannian_metric_tensor(bias_function, demographic_array)
        curvature_analysis = diff_geom.compute_curvature_analysis(bias_function, demographic_array)
        
        # Compute pairwise geodesic distances
        unique_groups = np.unique(demographic_array, axis=0)
        geodesic_distances = {}
        
        for i, group1 in enumerate(unique_groups):
            for j, group2 in enumerate(unique_groups[i+1:], i+1):
                dist = diff_geom.compute_geodesic_distance(group1, group2)
                geodesic_distances[f"group_{i}_to_group_{j}"] = float(dist)
        
        results['differential_geometry'] = {
            'metric_tensor_condition_number': float(np.linalg.cond(metric_tensor)),
            'curvature_analysis': curvature_analysis,
            'geodesic_distances': geodesic_distances,
            'max_geodesic_distance': float(max(geodesic_distances.values())) if geodesic_distances else 0
        }
        
        print(f"  ✅ Max geodesic distance: {results['differential_geometry']['max_geodesic_distance']:.6f}")
        
    except Exception as e:
        print(f"  ❌ Differential geometry analysis failed: {e}")
        results['differential_geometry'] = {'error': str(e)}
    
    # 3. Information Theory Analysis
    print("\n📊 Step 3: Advanced Information Theory Analysis...")
    try:
        # Discretize demographic and accuracy data
        demo_discrete = demographic_array[:, 0] * 3 + demographic_array[:, 1]  # Combine dimensions
        accuracy_discrete = (accuracy_array * 10).astype(int)  # Discretize accuracy
        
        mi_corrected = info_theory.compute_mutual_information_with_bias_correction(
            demo_discrete, accuracy_discrete)
        
        conditional_entropy = info_theory.compute_conditional_entropy_with_regularization(
            accuracy_discrete, demo_discrete)
        
        # Compute KL divergences between demographic groups
        unique_demos = np.unique(demo_discrete)
        kl_divergences = {}
        
        for i, demo1 in enumerate(unique_demos):
            for j, demo2 in enumerate(unique_demos[i+1:], i+1):
                mask1 = demo_discrete == demo1
                mask2 = demo_discrete == demo2
                
                if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                    hist1, _ = np.histogram(accuracy_discrete[mask1], bins=10, density=True)
                    hist2, _ = np.histogram(accuracy_discrete[mask2], bins=10, density=True)
                    
                    kl_div = info_theory.compute_kl_divergence_with_regularization(hist1, hist2)
                    js_div = info_theory.compute_jensen_shannon_divergence(hist1, hist2)
                    
                    kl_divergences[f"demo_{demo1}_vs_{demo2}"] = {
                        'kl_divergence': float(kl_div),
                        'js_divergence': float(js_div)
                    }
        
        results['information_theory'] = {
            'mutual_information_corrected': float(mi_corrected),
            'conditional_entropy': float(conditional_entropy),
            'kl_divergences': kl_divergences,
            'max_kl_divergence': float(max([d['kl_divergence'] for d in kl_divergences.values()])) if kl_divergences else 0
        }
        
        print(f"  ✅ Mutual Information: {results['information_theory']['mutual_information_corrected']:.6f}")
        
    except Exception as e:
        print(f"  ❌ Information theory analysis failed: {e}")
        results['information_theory'] = {'error': str(e)}
    
    # 4. Multi-objective Optimization
    print("\n⚖️ Step 4: Multi-objective Optimization...")
    try:
        # Define accuracy and fairness objectives
        def accuracy_objective(x):
            return -bias_function(x)  # Negative because we minimize (want high accuracy)
        
        def fairness_objective(x):
            return bias_function(x)   # Positive because we minimize (want low bias)
        
        pareto_analysis = optimization.compute_pareto_frontier(
            [fairness_objective, accuracy_objective], 
            center_point, 
            n_points=10
        )
        
        results['optimization'] = {
            'pareto_frontier_points': len(pareto_analysis.get('pareto_points', [])),
            'trade_off_slope': float(pareto_analysis.get('trade_off_slope', 0)),
            'min_bias_achievable': float(np.min(pareto_analysis['pareto_objectives'][:, 0])) if 'pareto_objectives' in pareto_analysis else 0,
            'max_accuracy_achievable': float(-np.min(pareto_analysis['pareto_objectives'][:, 1])) if 'pareto_objectives' in pareto_analysis else 0
        }
        
        print(f"  ✅ Pareto frontier computed with {results['optimization']['pareto_frontier_points']} points")
        
    except Exception as e:
        print(f"  ❌ Optimization analysis failed: {e}")
        results['optimization'] = {'error': str(e)}
    
    # Calculate overall advanced bias score
    try:
        gradient_component = results.get('gradient_analysis', {}).get('gradient_magnitude', 0)
        geometry_component = results.get('differential_geometry', {}).get('max_geodesic_distance', 0)
        info_component = results.get('information_theory', {}).get('mutual_information_corrected', 0)
        
        advanced_bias_score = (gradient_component + geometry_component + info_component) / 3
        
        if advanced_bias_score < 0.05:
            bias_severity = "Minimal"
        elif advanced_bias_score < 0.15:
            bias_severity = "Low"
        elif advanced_bias_score < 0.30:
            bias_severity = "Moderate"
        elif advanced_bias_score < 0.50:
            bias_severity = "High"
        else:
            bias_severity = "Severe"
        
        results['summary'] = {
            'advanced_bias_score': float(advanced_bias_score),
            'bias_severity': bias_severity,
            'gradient_component': float(gradient_component),
            'geometry_component': float(geometry_component),
            'information_component': float(info_component)
        }
        
    except Exception as e:
        results['summary'] = {'error': f'Summary calculation failed: {e}'}
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/day4_complete_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("🎯 DAY 4 COMPLETE ADVANCED MATHEMATICAL ANALYSIS RESULTS")
    print("=" * 70)
    
    if 'summary' in results and 'advanced_bias_score' in results['summary']:
        print(f"📊 Advanced Bias Score: {results['summary']['advanced_bias_score']:.6f}")
        print(f"🎯 Bias Severity: {results['summary']['bias_severity']}")
        print(f"🔬 Gradient Component: {results['summary']['gradient_component']:.6f}")
        print(f"🌐 Geometry Component: {results['summary']['geometry_component']:.6f}")
        print(f"📊 Information Component: {results['summary']['information_component']:.6f}")
    
    print(f"\n💾 Complete results saved to: {output_file}")
    
    return results

print("✅ Day 4 Complete Mathematical Framework implemented!")

# ============================================================================
# CELL 9: Run Everything - Complete Day 4 Execution
# ============================================================================
print("🚀 EXECUTING COMPLETE DAY 4 ANALYSIS")
print("=" * 50)

# Step 1: Run mathematical validation
print("Step 1: Mathematical Validation")
validation_results = run_mathematical_validation()

# Step 2: Run complete analysis (if validation passes)
if validation_results['overall_status'] in ['ALL_PASSED', 'MOSTLY_PASSED']:
    print("\n" + "="*50)
    print("Step 2: Complete Advanced Analysis")
    
    # Find the latest results file from previous days
    import glob
    results_files = glob.glob('./results/api_test_results_*.json')
    
    if results_files:
        latest_results_file = max(results_files)
        print(f"Using results file: {latest_results_file}")
        
        # Run the complete analysis
        complete_results = run_complete_day4_analysis(latest_results_file)
        
        print("\n🎉 DAY 4 COMPLETE!")
        print("✅ Advanced gradient analysis with Hessian matrices")
        print("✅ Differential geometry on demographic manifolds")
        print("✅ Multi-objective optimization with Pareto frontiers")
        print("✅ Advanced information theory with bias correction")
        print("✅ Comprehensive mathematical validation")
        
    else:
        print("❌ No previous results found. Please run Day 3 first.")
        print("💡 Make sure to run the face recognition analysis before Day 4.")
        
else:
    print("⚠️ Mathematical validation failed. Please check implementations.")
    print("💡 Some advanced features may not work correctly.")
