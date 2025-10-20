#!/usr/bin/env python3
"""
SIMPLE USAGE EXAMPLES: Facial Recognition Bias Detection
========================================================
Copy and paste these examples to get started immediately
"""

import numpy as np
import requests
import json
from pathlib import Path

# ============================================================================
# EXAMPLE 1: Basic Bias Detection (Copy and Run This)
# ============================================================================

def example_1_basic_bias_detection():
    """Detect bias in your facial recognition system"""
    print("🔍 EXAMPLE 1: Basic Bias Detection")
    print("-" * 50)
    
    # Step 1: Import the bias detection tools
    try:
        from Day8_BiasMitigationSuite import BiasMitigationSuite
    except ImportError:
        print("❌ Error: Make sure you're in the project directory")
        return
    
    # Step 2: Create sample data (replace with your real data)
    # Format: demographics = list of group names, predictions/labels = 0 or 1
    demographics = (['Asian'] * 100 + ['Black'] * 100 + 
                   ['Hispanic'] * 100 + ['White'] * 100)
    
    # Simulated biased predictions (your AI's outputs)
    predictions = ([1] * 90 + [0] * 10 +      # Asian: 90% positive
                  [1] * 60 + [0] * 40 +       # Black: 60% positive (biased!)
                  [1] * 75 + [0] * 25 +       # Hispanic: 75% positive
                  [1] * 85 + [0] * 15)        # White: 85% positive
    
    # Ground truth labels (what should be correct)
    true_labels = ([1] * 80 + [0] * 20) * 4   # 80% should be positive for all groups
    
    # Step 3: Detect bias
    suite = BiasMitigationSuite()
    bias_score = suite._calculate_bias_score(demographics, predictions, true_labels)
    
    # Step 4: Show results
    print(f"🎯 Bias Score: {bias_score:.1%}")
    if bias_score > 0.1:
        print("🚨 HIGH BIAS DETECTED - Your system needs fixing!")
    else:
        print("✅ Low bias - Your system is relatively fair")
    
    return demographics, predictions, true_labels, suite

# ============================================================================
# EXAMPLE 2: Fix the Bias (Run After Example 1)
# ============================================================================

def example_2_fix_bias(demographics, predictions, true_labels, suite):
    """Apply bias mitigation to improve fairness"""
    print("\n🔧 EXAMPLE 2: Fix the Bias")
    print("-" * 50)
    
    # Calculate original bias
    original_bias = suite._calculate_bias_score(demographics, predictions, true_labels)
    print(f"Original bias: {original_bias:.1%}")
    
    # Try different fixing methods
    methods = {
        'threshold_optimization': 'Threshold Optimization',
        'lagrange_multiplier_fairness': 'Lagrange Multiplier Method',
        'calibration_adjustment': 'Calibration Adjustment'
    }
    
    best_method = None
    best_improvement = 0
    
    for method_name, display_name in methods.items():
        try:
            # Apply the bias fix
            fixed_predictions = getattr(suite, method_name)(demographics, predictions, true_labels)
            new_bias = suite._calculate_bias_score(demographics, fixed_predictions, true_labels)
            improvement = ((original_bias - new_bias) / original_bias * 100)
            
            print(f"  {display_name}: {new_bias:.1%} bias ({improvement:.1f}% improvement)")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_method = display_name
                
        except Exception as e:
            print(f"  {display_name}: Error - {e}")
    
    print(f"\n🏆 Best method: {best_method} ({best_improvement:.1f}% improvement)")

# ============================================================================
# EXAMPLE 3: Use the API (Server Must Be Running)
# ============================================================================

def example_3_api_usage():
    """Use the REST API for bias analysis"""
    print("\n🌐 EXAMPLE 3: API Usage")
    print("-" * 50)
    
    api_url = "http://127.0.0.1:8000"
    
    # Check if server is running
    try:
        response = requests.get(f"{api_url}/api/health", timeout=5)
        print(f"✅ Server status: {response.json()['status']}")
    except requests.exceptions.RequestException:
        print("❌ Server not running. Start it with: python Day7_FastAPI_Backend.py")
        return
    
    # Get bias metrics
    try:
        response = requests.get(f"{api_url}/api/bias-metrics")
        metrics = response.json()
        
        print("📊 Current Bias Metrics:")
        print(f"  Statistical Parity: {metrics['statistical_parity']:.3f}")
        print(f"  Accuracy Disparity: {metrics['accuracy_disparity']:.3f}")
        print(f"  Overall Bias Score: {metrics['overall_bias_score']:.1%}")
        
    except Exception as e:
        print(f"❌ API Error: {e}")
    
    # Generate a report
    try:
        response = requests.post(f"{api_url}/api/export", 
                               json={"format": "csv", "data_type": "bias_metrics"})
        
        if response.status_code == 200:
            with open("bias_report.csv", "w") as f:
                f.write(response.text)
            print("✅ Report saved as bias_report.csv")
        
    except Exception as e:
        print(f"❌ Export Error: {e}")

# ============================================================================
# EXAMPLE 4: Complete Workflow (Everything Together)
# ============================================================================

def example_4_complete_workflow():
    """Complete bias detection and mitigation workflow"""
    print("\n🎯 EXAMPLE 4: Complete Workflow")
    print("-" * 50)
    
    # Step 1: Load your data (replace this with your actual data loading)
    print("1️⃣ Loading data...")
    
    # Example: Load from CSV (uncomment and modify for your data)
    # import pandas as pd
    # df = pd.read_csv('your_facial_recognition_results.csv')
    # demographics = df['demographic_group'].tolist()
    # predictions = df['ai_prediction'].tolist()
    # true_labels = df['ground_truth'].tolist()
    
    # For demo, use synthetic data
    demographics = np.random.choice(['Asian', 'Black', 'Hispanic', 'White'], 1000)
    predictions = np.random.choice([0, 1], 1000)
    true_labels = np.random.choice([0, 1], 1000)
    
    print(f"   Loaded {len(demographics)} samples across {len(set(demographics))} groups")
    
    # Step 2: Analyze bias
    print("2️⃣ Analyzing bias...")
    from Day8_BiasMitigationSuite import BiasMitigationSuite
    suite = BiasMitigationSuite()
    
    original_bias = suite._calculate_bias_score(demographics, predictions, true_labels)
    print(f"   Original bias score: {original_bias:.1%}")
    
    # Step 3: Apply best mitigation
    print("3️⃣ Applying bias mitigation...")
    fixed_predictions = suite.threshold_optimization(demographics, predictions, true_labels)
    new_bias = suite._calculate_bias_score(demographics, fixed_predictions, true_labels)
    improvement = ((original_bias - new_bias) / original_bias * 100)
    
    print(f"   After mitigation: {new_bias:.1%} bias")
    print(f"   Improvement: {improvement:.1f}%")
    
    # Step 4: Save results
    print("4️⃣ Saving results...")
    results = {
        'original_bias': float(original_bias),
        'mitigated_bias': float(new_bias),
        'improvement_percentage': float(improvement),
        'sample_size': len(demographics),
        'demographic_groups': list(set(demographics))
    }
    
    with open('complete_bias_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Complete analysis saved to complete_bias_analysis.json")

# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("🚀 FACIAL RECOGNITION BIAS DETECTION - USAGE EXAMPLES")
    print("=" * 60)
    
    # Run basic bias detection
    demographics, predictions, true_labels, suite = example_1_basic_bias_detection()
    
    # Fix the bias
    example_2_fix_bias(demographics, predictions, true_labels, suite)
    
    # Try API usage
    example_3_api_usage()
    
    # Complete workflow
    example_4_complete_workflow()
    
    print("\n" + "=" * 60)
    print("🎉 ALL EXAMPLES COMPLETE!")
    print("\n📚 Next Steps:")
    print("  1. Replace sample data with your real facial recognition data")
    print("  2. Start the dashboard: cd dashboard && npm start")
    print("  3. Explore the API docs: http://127.0.0.1:8000/docs")
    print("  4. Read the academic report: Day9_Academic_Report.md")
    print("\n💡 Need help? Check QUICK_START_GUIDE.md")
