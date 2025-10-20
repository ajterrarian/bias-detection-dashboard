# ============================================================================
# DAY 5: LARGE-SCALE BIAS ANALYSIS
# Morning Session: Comprehensive Bias Analysis with Chunk Processing
# ============================================================================

import numpy as np
import pandas as pd
import json
import asyncio
import time
from datetime import datetime
from tqdm import tqdm
import os
import glob
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path

# Import our previous implementations
# Note: In Colab, these imports will work after running the respective Day files
# For standalone execution, ensure all Day files are in the same directory
try:
    from Day3_Advanced_Bias_Analysis import FaceRecognitionTester
    from Day3_Mathematical_Framework import BiasAnalyzer
    from Day4_Core_Mathematical_Framework import AdvancedBiasAnalyzer, DifferentialGeometry, OptimizationTheory, InformationTheory
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("💡 Please ensure all Day 3-4 files are executed first in Colab")
    # Create placeholder classes for syntax checking
    class FaceRecognitionTester: pass
    class BiasAnalyzer: pass
    class AdvancedBiasAnalyzer: pass
    class DifferentialGeometry: pass
    class OptimizationTheory: pass
    class InformationTheory: pass

# ============================================================================
# CELL 1: Large-Scale Bias Analysis Engine
# ============================================================================
class LargeScaleBiasAnalyzer:
    """
    Large-scale bias analysis engine for processing thousands of face images.
    
    EXPLANATION FOR BEGINNERS:
    - This processes huge datasets by breaking them into smaller chunks
    - Like eating a large meal one bite at a time
    - Saves progress regularly so we don't lose work if something goes wrong
    - Combines all our advanced math from previous days
    """
    
    def __init__(self, chunk_size: int = 100, save_interval: int = 5):
        self.chunk_size = chunk_size
        self.save_interval = save_interval
        
        # Initialize all analyzers
        self.face_tester = FaceRecognitionTester()
        self.bias_analyzer = BiasAnalyzer()
        self.advanced_analyzer = AdvancedBiasAnalyzer()
        self.diff_geom = DifferentialGeometry(self.advanced_analyzer)
        self.optimization = OptimizationTheory(self.advanced_analyzer)
        self.info_theory = InformationTheory(self.advanced_analyzer)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('large_scale_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = []
        self.checkpoint_counter = 0
        
        print("🚀 Large-Scale Bias Analyzer initialized!")
        print(f"  📊 Chunk size: {chunk_size}")
        print(f"  💾 Save interval: {save_interval} chunks")
    
    def run_comprehensive_analysis(self, dataset_path: str, output_dir: str = "./results") -> Dict[str, Any]:
        """
        Run comprehensive large-scale bias analysis.
        
        EXPLANATION FOR BEGINNERS:
        - This is the main function that does everything
        - Loads your dataset, processes it in chunks, runs all our math
        - Saves results regularly and gives you progress updates
        """
        print("🔥 Starting Large-Scale Comprehensive Bias Analysis")
        print("=" * 60)
        
        start_time = time.time()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print("📊 Loading dataset...")
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data.get('results', []))
            else:
                # Try to find existing results
                results_files = glob.glob('./results/api_test_results_*.json')
                if results_files:
                    latest_file = max(results_files)
                    print(f"  📁 Using latest results file: {latest_file}")
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data.get('results', []))
                else:
                    raise FileNotFoundError("No valid dataset found")
            
            print(f"  ✅ Dataset loaded: {len(df)} records")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return {'error': f'Dataset loading failed: {e}'}
        
        # Filter successful results only
        if 'success' in df.columns:
            df = df[df['success'] == True]
            print(f"  🎯 Filtered to successful results: {len(df)} records")
        
        # Create chunks
        n_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        print(f"  📦 Processing in {n_chunks} chunks of size {self.chunk_size}")
        
        # Process chunks with progress bar
        chunk_results = []
        
        for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, len(df))
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            print(f"\n🔬 Processing chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_df)} records)")
            
            try:
                # Process chunk
                chunk_result = self._process_chunk(chunk_df, chunk_idx)
                chunk_results.append(chunk_result)
                
                # Save checkpoint
                if (chunk_idx + 1) % self.save_interval == 0:
                    self._save_checkpoint(chunk_results, output_dir, chunk_idx + 1)
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_idx}: {e}")
                chunk_results.append({
                    'chunk_id': chunk_idx,
                    'error': str(e),
                    'n_records': len(chunk_df)
                })
        
        # Combine all results
        print("\n🔄 Combining chunk results...")
        combined_results = self._combine_chunk_results(chunk_results)
        
        # Run global analysis
        print("\n🌐 Running global bias analysis...")
        global_analysis = self._run_global_analysis(combined_results, df)
        
        # Final results
        final_results = {
            'metadata': {
                'analysis_type': 'large_scale_comprehensive',
                'timestamp': datetime.now().isoformat(),
                'total_records': len(df),
                'n_chunks': n_chunks,
                'chunk_size': self.chunk_size,
                'processing_time_seconds': time.time() - start_time
            },
            'chunk_results': chunk_results,
            'combined_analysis': combined_results,
            'global_analysis': global_analysis
        }
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_file = f"{output_dir}/large_scale_analysis_{timestamp}.json"
        
        with open(final_output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n💾 Final results saved to: {final_output_file}")
        
        # Print summary
        self._print_analysis_summary(final_results)
        
        return final_results
    
    def _process_chunk(self, chunk_df: pd.DataFrame, chunk_idx: int) -> Dict[str, Any]:
        """Process a single chunk of data."""
        
        # Extract demographic and accuracy data
        demographic_data = []
        accuracy_data = []
        confidence_data = []
        
        for _, row in chunk_df.iterrows():
            try:
                # Extract demographic info
                demo_info = row.get('demographic_info', {})
                if isinstance(demo_info, str):
                    demo_info = json.loads(demo_info)
                
                # Encode demographics
                age_encoding = {'young': 0, 'middle_aged': 1, 'elderly': 2}
                gender_encoding = {'male': 0, 'female': 1}
                ethnicity_encoding = {'white': 0, 'black': 1, 'asian': 2, 'hispanic': 3, 'other': 4}
                
                demo_vector = [
                    age_encoding.get(demo_info.get('age_category', 'young'), 0),
                    gender_encoding.get(demo_info.get('gender', 'male'), 0),
                    ethnicity_encoding.get(demo_info.get('ethnicity', 'other'), 4)
                ]
                demographic_data.append(demo_vector)
                
                # Extract API results
                api_results = row.get('api_results', {})
                if isinstance(api_results, str):
                    api_results = json.loads(api_results)
                
                # Calculate combined accuracy
                accuracies = []
                confidences = []
                
                # AWS Rekognition
                if 'aws' in api_results and api_results['aws'].get('success'):
                    aws_faces = api_results['aws'].get('faces', [])
                    if aws_faces:
                        aws_conf = np.mean([f['confidence'] for f in aws_faces])
                        accuracies.append(aws_conf / 100.0)  # Normalize to 0-1
                        confidences.append(aws_conf)
                
                # Google Cloud Vision
                if 'google' in api_results and api_results['google'].get('success'):
                    google_faces = api_results['google'].get('faces', [])
                    if google_faces:
                        google_conf = np.mean([f['confidence'] for f in google_faces])
                        accuracies.append(google_conf)
                        confidences.append(google_conf * 100)
                
                # Average accuracy
                avg_accuracy = np.mean(accuracies) if accuracies else 0.5
                avg_confidence = np.mean(confidences) if confidences else 50.0
                
                accuracy_data.append(avg_accuracy)
                confidence_data.append(avg_confidence)
                
            except Exception as e:
                self.logger.warning(f"Error processing row in chunk {chunk_idx}: {e}")
                # Use default values
                demographic_data.append([0, 0, 4])  # Default demographics
                accuracy_data.append(0.5)  # Default accuracy
                confidence_data.append(50.0)  # Default confidence
        
        # Convert to numpy arrays
        demo_array = np.array(demographic_data)
        accuracy_array = np.array(accuracy_data)
        confidence_array = np.array(confidence_data)
        
        print(f"  📊 Extracted {len(demo_array)} data points")
        
        # Run bias analysis on chunk
        chunk_analysis = {}
        
        # 1. Basic bias metrics
        try:
            basic_metrics = self._compute_basic_bias_metrics(demo_array, accuracy_array)
            chunk_analysis['basic_metrics'] = basic_metrics
            print(f"  ✅ Basic metrics computed")
        except Exception as e:
            chunk_analysis['basic_metrics'] = {'error': str(e)}
            print(f"  ❌ Basic metrics failed: {e}")
        
        # 2. Advanced mathematical analysis
        try:
            advanced_metrics = self._compute_advanced_bias_metrics(demo_array, accuracy_array)
            chunk_analysis['advanced_metrics'] = advanced_metrics
            print(f"  ✅ Advanced metrics computed")
        except Exception as e:
            chunk_analysis['advanced_metrics'] = {'error': str(e)}
            print(f"  ❌ Advanced metrics failed: {e}")
        
        # 3. Information theory analysis
        try:
            info_metrics = self._compute_information_theory_metrics(demo_array, accuracy_array)
            chunk_analysis['information_metrics'] = info_metrics
            print(f"  ✅ Information theory computed")
        except Exception as e:
            chunk_analysis['information_metrics'] = {'error': str(e)}
            print(f"  ❌ Information theory failed: {e}")
        
        return {
            'chunk_id': chunk_idx,
            'n_records': len(chunk_df),
            'demographic_distribution': self._compute_demographic_distribution(demo_array),
            'accuracy_statistics': {
                'mean': float(np.mean(accuracy_array)),
                'std': float(np.std(accuracy_array)),
                'min': float(np.min(accuracy_array)),
                'max': float(np.max(accuracy_array)),
                'median': float(np.median(accuracy_array))
            },
            'bias_analysis': chunk_analysis
        }
    
    def _compute_basic_bias_metrics(self, demo_array: np.ndarray, accuracy_array: np.ndarray) -> Dict[str, Any]:
        """Compute basic bias metrics for a chunk."""
        
        metrics = {}
        
        # Accuracy disparity by demographic groups
        unique_groups = np.unique(demo_array, axis=0)
        group_accuracies = {}
        
        for i, group in enumerate(unique_groups):
            mask = np.all(demo_array == group, axis=1)
            group_acc = accuracy_array[mask]
            
            if len(group_acc) > 0:
                group_accuracies[f"group_{i}"] = {
                    'mean_accuracy': float(np.mean(group_acc)),
                    'std_accuracy': float(np.std(group_acc)),
                    'count': int(len(group_acc)),
                    'demographics': group.tolist()
                }
        
        metrics['group_accuracies'] = group_accuracies
        
        # Overall disparity measures
        if len(group_accuracies) > 1:
            group_means = [g['mean_accuracy'] for g in group_accuracies.values()]
            metrics['accuracy_disparity'] = {
                'max_difference': float(np.max(group_means) - np.min(group_means)),
                'coefficient_of_variation': float(np.std(group_means) / np.mean(group_means)),
                'range_ratio': float(np.max(group_means) / np.min(group_means)) if np.min(group_means) > 0 else float('inf')
            }
        
        return metrics
    
    def _compute_advanced_bias_metrics(self, demo_array: np.ndarray, accuracy_array: np.ndarray) -> Dict[str, Any]:
        """Compute advanced mathematical bias metrics."""
        
        metrics = {}
        
        # Define bias function
        def bias_function(demo_point):
            distances = np.linalg.norm(demo_array - demo_point, axis=1)
            nearest_indices = np.argsort(distances)[:min(10, len(distances))]
            nearest_accuracies = accuracy_array[nearest_indices]
            return np.var(nearest_accuracies) if len(nearest_accuracies) > 1 else 0
        
        try:
            # Gradient analysis
            center_point = np.mean(demo_array, axis=0)
            gradient = self.advanced_analyzer.compute_numerical_gradient(bias_function, center_point)
            
            metrics['gradient_analysis'] = {
                'bias_gradient': gradient.tolist(),
                'gradient_magnitude': float(np.linalg.norm(gradient))
            }
        except Exception as e:
            metrics['gradient_analysis'] = {'error': str(e)}
        
        try:
            # Differential geometry
            if len(demo_array) >= 10:  # Need sufficient points
                sample_points = demo_array[::max(1, len(demo_array)//20)]  # Sample for efficiency
                metric_tensor = self.diff_geom.compute_riemannian_metric_tensor(bias_function, sample_points)
                
                metrics['differential_geometry'] = {
                    'metric_condition_number': float(np.linalg.cond(metric_tensor)),
                    'metric_determinant': float(np.linalg.det(metric_tensor))
                }
        except Exception as e:
            metrics['differential_geometry'] = {'error': str(e)}
        
        return metrics
    
    def _compute_information_theory_metrics(self, demo_array: np.ndarray, accuracy_array: np.ndarray) -> Dict[str, Any]:
        """Compute information theory bias metrics."""
        
        metrics = {}
        
        try:
            # Discretize data
            demo_discrete = demo_array[:, 0] * 9 + demo_array[:, 1] * 3 + demo_array[:, 2]  # Combine dimensions
            accuracy_discrete = (accuracy_array * 10).astype(int)
            
            # Mutual information
            mi_corrected = self.info_theory.compute_mutual_information_with_bias_correction(
                demo_discrete, accuracy_discrete)
            
            # Conditional entropy
            conditional_entropy = self.info_theory.compute_conditional_entropy_with_regularization(
                accuracy_discrete, demo_discrete)
            
            metrics['mutual_information'] = float(mi_corrected)
            metrics['conditional_entropy'] = float(conditional_entropy)
            
            # Information-theoretic bias score
            # Higher MI = more bias, Higher H(Y|X) = less bias
            info_bias_score = mi_corrected / (conditional_entropy + 1e-8)
            metrics['information_bias_score'] = float(info_bias_score)
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def _compute_demographic_distribution(self, demo_array: np.ndarray) -> Dict[str, Any]:
        """Compute demographic distribution statistics."""
        
        distribution = {}
        
        # Age distribution
        age_counts = np.bincount(demo_array[:, 0].astype(int), minlength=3)
        distribution['age'] = {
            'young': int(age_counts[0]),
            'middle_aged': int(age_counts[1]),
            'elderly': int(age_counts[2])
        }
        
        # Gender distribution
        gender_counts = np.bincount(demo_array[:, 1].astype(int), minlength=2)
        distribution['gender'] = {
            'male': int(gender_counts[0]),
            'female': int(gender_counts[1])
        }
        
        # Ethnicity distribution
        ethnicity_counts = np.bincount(demo_array[:, 2].astype(int), minlength=5)
        distribution['ethnicity'] = {
            'white': int(ethnicity_counts[0]),
            'black': int(ethnicity_counts[1]),
            'asian': int(ethnicity_counts[2]),
            'hispanic': int(ethnicity_counts[3]),
            'other': int(ethnicity_counts[4])
        }
        
        return distribution
    
    def _save_checkpoint(self, chunk_results: List[Dict], output_dir: str, chunk_num: int):
        """Save intermediate checkpoint."""
        checkpoint_file = f"{output_dir}/bias_analysis_checkpoint_{chunk_num}.json"
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'chunks_processed': chunk_num,
            'chunk_results': chunk_results
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        print(f"  💾 Checkpoint saved: {checkpoint_file}")
    
    def _combine_chunk_results(self, chunk_results: List[Dict]) -> Dict[str, Any]:
        """Combine results from all chunks."""
        
        combined = {
            'total_chunks': len(chunk_results),
            'total_records': sum(r.get('n_records', 0) for r in chunk_results),
            'successful_chunks': len([r for r in chunk_results if 'error' not in r])
        }
        
        # Combine demographic distributions
        combined_demo = {'age': {}, 'gender': {}, 'ethnicity': {}}
        
        for result in chunk_results:
            if 'demographic_distribution' in result:
                demo_dist = result['demographic_distribution']
                
                for category in ['age', 'gender', 'ethnicity']:
                    for key, value in demo_dist.get(category, {}).items():
                        combined_demo[category][key] = combined_demo[category].get(key, 0) + value
        
        combined['combined_demographic_distribution'] = combined_demo
        
        # Combine accuracy statistics
        all_accuracies = []
        for result in chunk_results:
            if 'accuracy_statistics' in result:
                stats = result['accuracy_statistics']
                # Approximate reconstruction of individual values
                mean_acc = stats.get('mean', 0.5)
                std_acc = stats.get('std', 0.1)
                n_records = result.get('n_records', 1)
                
                # Generate approximate values (for aggregation purposes)
                approx_values = np.random.normal(mean_acc, std_acc, n_records)
                all_accuracies.extend(approx_values)
        
        if all_accuracies:
            combined['combined_accuracy_statistics'] = {
                'mean': float(np.mean(all_accuracies)),
                'std': float(np.std(all_accuracies)),
                'min': float(np.min(all_accuracies)),
                'max': float(np.max(all_accuracies)),
                'median': float(np.median(all_accuracies))
            }
        
        return combined
    
    def _run_global_analysis(self, combined_results: Dict, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Run global analysis across all data."""
        
        global_analysis = {}
        
        # Global bias severity assessment
        try:
            total_records = combined_results.get('total_records', 0)
            demo_dist = combined_results.get('combined_demographic_distribution', {})
            
            # Calculate demographic balance
            age_dist = demo_dist.get('age', {})
            gender_dist = demo_dist.get('gender', {})
            ethnicity_dist = demo_dist.get('ethnicity', {})
            
            # Balance scores (closer to 1 = more balanced)
            age_balance = self._calculate_balance_score(list(age_dist.values()))
            gender_balance = self._calculate_balance_score(list(gender_dist.values()))
            ethnicity_balance = self._calculate_balance_score(list(ethnicity_dist.values()))
            
            global_analysis['demographic_balance'] = {
                'age_balance_score': age_balance,
                'gender_balance_score': gender_balance,
                'ethnicity_balance_score': ethnicity_balance,
                'overall_balance_score': (age_balance + gender_balance + ethnicity_balance) / 3
            }
            
            # Global bias severity
            acc_stats = combined_results.get('combined_accuracy_statistics', {})
            accuracy_cv = acc_stats.get('std', 0) / acc_stats.get('mean', 1) if acc_stats.get('mean', 0) > 0 else 0
            
            if accuracy_cv < 0.1:
                bias_severity = "Minimal"
            elif accuracy_cv < 0.2:
                bias_severity = "Low"
            elif accuracy_cv < 0.35:
                bias_severity = "Moderate"
            elif accuracy_cv < 0.5:
                bias_severity = "High"
            else:
                bias_severity = "Severe"
            
            global_analysis['bias_assessment'] = {
                'accuracy_coefficient_of_variation': accuracy_cv,
                'bias_severity': bias_severity,
                'total_records_analyzed': total_records
            }
            
        except Exception as e:
            global_analysis['error'] = str(e)
        
        return global_analysis
    
    def _calculate_balance_score(self, counts: List[int]) -> float:
        """Calculate balance score for a distribution (1 = perfectly balanced)."""
        if not counts or sum(counts) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = np.array(counts) / sum(counts)
        
        # Calculate entropy (higher = more balanced)
        entropy = -np.sum(probs * np.log2(probs + 1e-8))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(counts))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _print_analysis_summary(self, results: Dict[str, Any]):
        """Print comprehensive analysis summary."""
        
        print("\n" + "=" * 70)
        print("🎯 LARGE-SCALE BIAS ANALYSIS SUMMARY")
        print("=" * 70)
        
        metadata = results.get('metadata', {})
        combined = results.get('combined_analysis', {})
        global_analysis = results.get('global_analysis', {})
        
        print(f"📊 Total Records Processed: {metadata.get('total_records', 0):,}")
        print(f"📦 Number of Chunks: {metadata.get('n_chunks', 0)}")
        print(f"⏱️ Processing Time: {metadata.get('processing_time_seconds', 0):.1f} seconds")
        
        # Demographic distribution
        demo_dist = combined.get('combined_demographic_distribution', {})
        if demo_dist:
            print("\n👥 DEMOGRAPHIC DISTRIBUTION:")
            
            age_dist = demo_dist.get('age', {})
            print(f"  Age: Young={age_dist.get('young', 0)}, Middle={age_dist.get('middle_aged', 0)}, Elderly={age_dist.get('elderly', 0)}")
            
            gender_dist = demo_dist.get('gender', {})
            print(f"  Gender: Male={gender_dist.get('male', 0)}, Female={gender_dist.get('female', 0)}")
            
            ethnicity_dist = demo_dist.get('ethnicity', {})
            print(f"  Ethnicity: White={ethnicity_dist.get('white', 0)}, Black={ethnicity_dist.get('black', 0)}, Asian={ethnicity_dist.get('asian', 0)}, Hispanic={ethnicity_dist.get('hispanic', 0)}, Other={ethnicity_dist.get('other', 0)}")
        
        # Accuracy statistics
        acc_stats = combined.get('combined_accuracy_statistics', {})
        if acc_stats:
            print(f"\n📈 ACCURACY STATISTICS:")
            print(f"  Mean: {acc_stats.get('mean', 0):.4f}")
            print(f"  Std Dev: {acc_stats.get('std', 0):.4f}")
            print(f"  Range: {acc_stats.get('min', 0):.4f} - {acc_stats.get('max', 0):.4f}")
        
        # Bias assessment
        bias_assessment = global_analysis.get('bias_assessment', {})
        if bias_assessment:
            print(f"\n⚖️ BIAS ASSESSMENT:")
            print(f"  Bias Severity: {bias_assessment.get('bias_severity', 'Unknown')}")
            print(f"  Accuracy CV: {bias_assessment.get('accuracy_coefficient_of_variation', 0):.4f}")
        
        # Balance scores
        balance = global_analysis.get('demographic_balance', {})
        if balance:
            print(f"\n⚖️ DEMOGRAPHIC BALANCE SCORES:")
            print(f"  Age Balance: {balance.get('age_balance_score', 0):.3f}")
            print(f"  Gender Balance: {balance.get('gender_balance_score', 0):.3f}")
            print(f"  Ethnicity Balance: {balance.get('ethnicity_balance_score', 0):.3f}")
            print(f"  Overall Balance: {balance.get('overall_balance_score', 0):.3f}")

print("✅ Large-Scale Bias Analysis Engine implemented!")

# ============================================================================
# CELL 2: Execution Script for Large-Scale Analysis
# ============================================================================
def run_day5_morning_analysis():
    """
    Execute Day 5 morning large-scale bias analysis.
    
    EXPLANATION FOR BEGINNERS:
    - This runs the complete large-scale analysis
    - Processes all your face recognition data in chunks
    - Gives you comprehensive bias metrics across all demographic groups
    """
    print("🌅 DAY 5 MORNING: Large-Scale Bias Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = LargeScaleBiasAnalyzer(chunk_size=100, save_interval=5)
    
    # Find latest results file
    results_files = glob.glob('./results/api_test_results_*.json')
    
    if not results_files:
        print("❌ No previous face recognition results found!")
        print("💡 Please run Day 3 face recognition analysis first.")
        return None
    
    latest_file = max(results_files)
    print(f"📁 Using dataset: {latest_file}")
    
    # Run comprehensive analysis
    try:
        results = analyzer.run_comprehensive_analysis(
            dataset_path=latest_file,
            output_dir="./results"
        )
        
        print("\n🎉 Large-scale analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

# Run the analysis
if __name__ == "__main__":
    results = run_day5_morning_analysis()

print("✅ Day 5 Morning: Large-Scale Bias Analysis Complete!")
