"""
Main execution script for facial recognition bias detection analysis.
Orchestrates the complete pipeline from data collection to visualization.
"""
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bias_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from config import config
from api_client import FaceRecognitionClient
from data_pipeline import DataPipeline
from bias_metrics import BiasMetrics
from visualization import BiasVisualization

def main():
    """Main execution function for Day 2 bias analysis."""
    logger.info("Starting Day 2: Facial Recognition Bias Detection Analysis")
    
    # Initialize components
    logger.info("Initializing components...")
    client = FaceRecognitionClient()
    pipeline = DataPipeline()
    metrics = BiasMetrics()
    viz = BiasVisualization()
    
    # Test API connections
    logger.info("Testing API connections...")
    connection_status = client.test_connections()
    logger.info(f"API Connection Status: {connection_status}")
    
    if not any(connection_status.values()):
        logger.error("No API connections available. Please check your credentials.")
        return
    
    # Load and process sample dataset
    logger.info("Loading sample dataset...")
    sample_data = pipeline.load_sample_dataset()
    
    logger.info("Processing dataset (this may take several minutes)...")
    results = pipeline.process_full_dataset('sample')
    
    if not results:
        logger.error("Failed to process dataset")
        return
    
    # Prepare data for bias analysis
    logger.info("Preparing bias analysis data...")
    bias_data = pipeline.get_bias_analysis_data('sample')
    
    # Perform comprehensive bias analysis
    logger.info("Performing bias analysis...")
    bias_analysis = metrics.comprehensive_bias_analysis(bias_data)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Accuracy heatmap
    accuracy_heatmap = viz.create_accuracy_heatmap(bias_data)
    viz.save_visualization(accuracy_heatmap, 'accuracy_heatmap')
    
    # 2. Bias disparity chart
    disparity_chart = viz.create_bias_disparity_chart(bias_analysis)
    viz.save_visualization(disparity_chart, 'bias_disparity_chart')
    
    # 3. Demographic comparison
    demo_comparison = viz.create_demographic_comparison(bias_data, 'accuracy')
    viz.save_visualization(demo_comparison, 'demographic_comparison')
    
    # 4. Comprehensive dashboard
    dashboard = viz.create_comprehensive_dashboard(bias_data, bias_analysis)
    viz.save_visualization(dashboard, 'comprehensive_dashboard')
    
    # Print summary results
    print("\n" + "="*60)
    print("FACIAL RECOGNITION BIAS ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nDataset: {results['dataset_name']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Groups Analyzed: {results['summary']['total_groups']}")
    print(f"Total Images Processed: {results['summary']['total_images_processed']}")
    print(f"Total Images Failed: {results['summary']['total_images_failed']}")
    
    print(f"\nOverall Bias Score: {bias_analysis['overall_bias_score']:.4f}")
    print(f"Bias Level: {bias_analysis['bias_level']}")
    
    print("\nDetection Rates by Service:")
    for service in ['aws', 'google']:
        overall_rate = results['summary'].get(f'overall_detection_rate_{service}', 0)
        print(f"  {service.upper()}: {overall_rate:.3f}")
    
    print("\nAccuracy Scores by Service:")
    for service in ['aws', 'google']:
        overall_acc = results['summary'].get(f'overall_accuracy_{service}', 0)
        print(f"  {service.upper()}: {overall_acc:.3f}")
    
    if 'accuracy_disparity' in bias_analysis['metrics']:
        acc_disp = bias_analysis['metrics']['accuracy_disparity']
        print(f"\nAccuracy Disparity:")
        print(f"  Range: {acc_disp.get('accuracy_range', 0):.4f}")
        print(f"  Max Pairwise Disparity: {acc_disp.get('max_pairwise_disparity', 0):.4f}")
    
    print(f"\nVisualization files saved to: {config.output_dir}/")
    print("- accuracy_heatmap.html")
    print("- bias_disparity_chart.html") 
    print("- demographic_comparison.html")
    print("- comprehensive_dashboard.html")
    
    logger.info("Day 2 analysis completed successfully!")
    return results, bias_analysis

if __name__ == "__main__":
    main()
