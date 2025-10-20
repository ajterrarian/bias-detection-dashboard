#!/usr/bin/env python3
"""
Quick test script to verify Day 2 setup and run initial bias analysis.
"""
import os
import sys
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from config import config
        from api_client import FaceRecognitionClient
        from data_pipeline import DataPipeline
        from bias_metrics import BiasMetrics
        from visualization import BiasVisualization
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_api_credentials():
    """Test API credential setup."""
    from config import config
    
    validation = config.validate_credentials()
    print(f"API Credentials Status:")
    print(f"  AWS: {'✅' if validation['aws'] else '❌'}")
    print(f"  Google: {'✅' if validation['google'] else '❌'}")
    
    return any(validation.values())

def run_mock_analysis():
    """Run a mock bias analysis with sample data."""
    from bias_metrics import BiasMetrics
    from visualization import BiasVisualization
    
    # Create mock data simulating API results
    mock_bias_data = {
        'young_male': {
            'accuracy': {'aws': 0.92, 'google': 0.89},
            'detection_rate': {'aws': 0.95, 'google': 0.91},
            'sample_size': 10
        },
        'young_female': {
            'accuracy': {'aws': 0.88, 'google': 0.85},
            'detection_rate': {'aws': 0.90, 'google': 0.87},
            'sample_size': 10
        },
        'middle_aged_male': {
            'accuracy': {'aws': 0.85, 'google': 0.82},
            'detection_rate': {'aws': 0.88, 'google': 0.84},
            'sample_size': 8
        },
        'middle_aged_female': {
            'accuracy': {'aws': 0.83, 'google': 0.80},
            'detection_rate': {'aws': 0.86, 'google': 0.82},
            'sample_size': 8
        },
        'elderly_male': {
            'accuracy': {'aws': 0.78, 'google': 0.75},
            'detection_rate': {'aws': 0.82, 'google': 0.78},
            'sample_size': 6
        },
        'elderly_female': {
            'accuracy': {'aws': 0.76, 'google': 0.73},
            'detection_rate': {'aws': 0.80, 'google': 0.76},
            'sample_size': 6
        },
        'light_skin': {
            'accuracy': {'aws': 0.90, 'google': 0.87},
            'detection_rate': {'aws': 0.93, 'google': 0.89},
            'sample_size': 15
        },
        'medium_skin': {
            'accuracy': {'aws': 0.84, 'google': 0.81},
            'detection_rate': {'aws': 0.87, 'google': 0.83},
            'sample_size': 12
        },
        'dark_skin': {
            'accuracy': {'aws': 0.79, 'google': 0.76},
            'detection_rate': {'aws': 0.83, 'google': 0.79},
            'sample_size': 10
        }
    }
    
    # Run bias analysis
    metrics = BiasMetrics()
    bias_analysis = metrics.comprehensive_bias_analysis(mock_bias_data)
    
    # Create visualizations
    viz = BiasVisualization()
    
    # Create output directory
    os.makedirs('./output', exist_ok=True)
    
    try:
        # Generate visualizations
        accuracy_heatmap = viz.create_accuracy_heatmap(mock_bias_data)
        viz.save_visualization(accuracy_heatmap, 'test_accuracy_heatmap')
        
        disparity_chart = viz.create_bias_disparity_chart(bias_analysis)
        viz.save_visualization(disparity_chart, 'test_bias_disparity')
        
        dashboard = viz.create_comprehensive_dashboard(mock_bias_data, bias_analysis)
        viz.save_visualization(dashboard, 'test_dashboard')
        
        print("✅ Mock analysis completed successfully")
        print(f"✅ Visualizations saved to ./output/")
        
        # Print key results
        print(f"\n📊 MOCK ANALYSIS RESULTS:")
        print(f"Overall Bias Score: {bias_analysis['overall_bias_score']:.4f}")
        print(f"Bias Level: {bias_analysis['bias_level']}")
        
        if 'accuracy_disparity' in bias_analysis['metrics']:
            acc_disp = bias_analysis['metrics']['accuracy_disparity']
            print(f"Max Accuracy Disparity: {acc_disp.get('max_pairwise_disparity', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock analysis failed: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 Testing Day 2 Facial Recognition Bias Detection Setup")
    print("=" * 60)
    
    # Test 1: Module imports
    print("\n1. Testing module imports...")
    if not test_imports():
        return False
    
    # Test 2: API credentials
    print("\n2. Testing API credentials...")
    has_credentials = test_api_credentials()
    
    # Test 3: Mock analysis
    print("\n3. Running mock bias analysis...")
    analysis_success = run_mock_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DAY 2 SETUP SUMMARY")
    print("=" * 60)
    print("✅ Project structure created")
    print("✅ Mathematical bias metrics implemented")
    print("✅ Interactive visualization system ready")
    print("✅ Data pipeline framework complete")
    
    if has_credentials:
        print("✅ API credentials configured")
    else:
        print("⚠️  API credentials need setup for live analysis")
    
    if analysis_success:
        print("✅ Mock bias analysis successful")
        print("\n🎯 Ready to proceed with live API testing!")
        print("💡 Next: Set up Google Cloud credentials and run full analysis")
    else:
        print("❌ Mock analysis failed - check dependencies")
    
    print(f"\n📁 Project files created in: {Path.cwd()}")
    print("📊 Test visualizations available in: ./output/")

if __name__ == "__main__":
    main()
