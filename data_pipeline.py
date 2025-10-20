"""
Data pipeline for collecting and processing diverse face datasets for bias analysis.
Handles dataset loading, demographic labeling, and batch processing.
"""
import os
import pandas as pd
import numpy as np
import requests
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from io import BytesIO

from config import config
from api_client import FaceRecognitionClient

logger = logging.getLogger(__name__)

class DataPipeline:
    """Pipeline for managing facial recognition datasets and processing."""
    
    def __init__(self):
        self.client = FaceRecognitionClient()
        self.datasets = {}
        self.processed_results = {}
        
        # Create data directories
        config.create_directories()
    
    def load_sample_dataset(self) -> Dict[str, List]:
        """Load a sample dataset with diverse demographic representation."""
        # Sample dataset with publicly available images and demographic labels
        sample_data = {
            'young_male': [
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400',
                'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400'
            ],
            'young_female': [
                'https://images.unsplash.com/photo-1494790108755-2616b612b786?w=400',
                'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400'
            ],
            'middle_aged_male': [
                'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400',
                'https://images.unsplash.com/photo-1560250097-0b93528c311a?w=400'
            ],
            'middle_aged_female': [
                'https://images.unsplash.com/photo-1580489944761-15a19d654956?w=400',
                'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=400'
            ],
            'elderly_male': [
                'https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=400'
            ],
            'elderly_female': [
                'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400'
            ]
        }
        
        # Add skin tone categories (overlapping with age/gender)
        sample_data.update({
            'light_skin': sample_data['young_male'][:1] + sample_data['young_female'][:1],
            'medium_skin': sample_data['middle_aged_male'][:1] + sample_data['middle_aged_female'][:1],
            'dark_skin': sample_data['young_male'][1:] + sample_data['young_female'][1:]
        })
        
        self.datasets['sample'] = sample_data
        logger.info(f"Loaded sample dataset with {sum(len(urls) for urls in sample_data.values())} images")
        return sample_data
    
    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download image from URL with error handling."""
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Verify it's a valid image
                img = Image.open(BytesIO(response.content))
                img.verify()
                
                return response.content
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download image after {max_retries} attempts: {url}")
        
        return None
    
    def process_demographic_group(self, group_name: str, image_urls: List[str]) -> Dict:
        """Process all images in a demographic group."""
        group_results = {
            'group': group_name,
            'total_images': len(image_urls),
            'processed_images': 0,
            'failed_images': 0,
            'aws_results': [],
            'google_results': [],
            'accuracy_scores': {'aws': [], 'google': []},
            'detection_rates': {'aws': 0.0, 'google': 0.0}
        }
        
        for i, url in enumerate(image_urls):
            logger.info(f"Processing {group_name} image {i+1}/{len(image_urls)}")
            
            # Download image
            image_bytes = self.download_image(url)
            if not image_bytes:
                group_results['failed_images'] += 1
                continue
            
            # Analyze with APIs
            try:
                api_results = self.client.analyze_image(image_bytes=image_bytes)
                
                # Process AWS results
                if 'aws' in api_results and 'faces' in api_results['aws']:
                    aws_faces = api_results['aws']['faces']
                    group_results['aws_results'].append({
                        'image_index': i,
                        'face_count': len(aws_faces),
                        'faces': aws_faces
                    })
                    
                    # Calculate accuracy score (confidence-based)
                    if aws_faces:
                        avg_confidence = np.mean([face['confidence'] for face in aws_faces])
                        group_results['accuracy_scores']['aws'].append(avg_confidence / 100.0)
                
                # Process Google results
                if 'google' in api_results and 'faces' in api_results['google']:
                    google_faces = api_results['google']['faces']
                    group_results['google_results'].append({
                        'image_index': i,
                        'face_count': len(google_faces),
                        'faces': google_faces
                    })
                    
                    # Calculate accuracy score (confidence-based)
                    if google_faces:
                        avg_confidence = np.mean([face['confidence'] for face in google_faces])
                        group_results['accuracy_scores']['google'].append(avg_confidence)
                
                group_results['processed_images'] += 1
                
            except Exception as e:
                logger.error(f"Failed to analyze image {i} for group {group_name}: {e}")
                group_results['failed_images'] += 1
        
        # Calculate detection rates
        if group_results['processed_images'] > 0:
            group_results['detection_rates']['aws'] = len(group_results['aws_results']) / group_results['processed_images']
            group_results['detection_rates']['google'] = len(group_results['google_results']) / group_results['processed_images']
        
        # Calculate average accuracy scores
        for service in ['aws', 'google']:
            if group_results['accuracy_scores'][service]:
                group_results[f'avg_accuracy_{service}'] = np.mean(group_results['accuracy_scores'][service])
                group_results[f'std_accuracy_{service}'] = np.std(group_results['accuracy_scores'][service])
            else:
                group_results[f'avg_accuracy_{service}'] = 0.0
                group_results[f'std_accuracy_{service}'] = 0.0
        
        return group_results
    
    def process_full_dataset(self, dataset_name: str = 'sample') -> Dict:
        """Process entire dataset and return comprehensive results."""
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not found. Available: {list(self.datasets.keys())}")
            return {}
        
        dataset = self.datasets[dataset_name]
        results = {
            'dataset_name': dataset_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'demographic_groups': {},
            'summary': {}
        }
        
        # Process each demographic group
        for group_name, image_urls in dataset.items():
            logger.info(f"Processing demographic group: {group_name}")
            group_results = self.process_demographic_group(group_name, image_urls)
            results['demographic_groups'][group_name] = group_results
        
        # Generate summary statistics
        results['summary'] = self._generate_summary_stats(results['demographic_groups'])
        
        # Save results
        self.processed_results[dataset_name] = results
        self._save_results(results, dataset_name)
        
        return results
    
    def _generate_summary_stats(self, group_results: Dict) -> Dict:
        """Generate summary statistics across all demographic groups."""
        summary = {
            'total_groups': len(group_results),
            'total_images_processed': sum(group['processed_images'] for group in group_results.values()),
            'total_images_failed': sum(group['failed_images'] for group in group_results.values()),
            'avg_detection_rates': {'aws': [], 'google': []},
            'avg_accuracy_scores': {'aws': [], 'google': []}
        }
        
        # Collect metrics across groups
        for group_data in group_results.values():
            for service in ['aws', 'google']:
                if group_data['detection_rates'][service] > 0:
                    summary['avg_detection_rates'][service].append(group_data['detection_rates'][service])
                
                if f'avg_accuracy_{service}' in group_data and group_data[f'avg_accuracy_{service}'] > 0:
                    summary['avg_accuracy_scores'][service].append(group_data[f'avg_accuracy_{service}'])
        
        # Calculate overall averages
        for service in ['aws', 'google']:
            if summary['avg_detection_rates'][service]:
                summary[f'overall_detection_rate_{service}'] = np.mean(summary['avg_detection_rates'][service])
                summary[f'detection_rate_std_{service}'] = np.std(summary['avg_detection_rates'][service])
            else:
                summary[f'overall_detection_rate_{service}'] = 0.0
                summary[f'detection_rate_std_{service}'] = 0.0
            
            if summary['avg_accuracy_scores'][service]:
                summary[f'overall_accuracy_{service}'] = np.mean(summary['avg_accuracy_scores'][service])
                summary[f'accuracy_std_{service}'] = np.std(summary['avg_accuracy_scores'][service])
            else:
                summary[f'overall_accuracy_{service}'] = 0.0
                summary[f'accuracy_std_{service}'] = 0.0
        
        return summary
    
    def _save_results(self, results: Dict, dataset_name: str):
        """Save results to JSON file."""
        output_file = Path(config.results_dir) / f"{dataset_name}_results.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_bias_analysis_data(self, dataset_name: str = 'sample') -> Dict:
        """Prepare data for bias analysis."""
        if dataset_name not in self.processed_results:
            logger.error(f"No processed results found for dataset {dataset_name}")
            return {}
        
        results = self.processed_results[dataset_name]
        bias_data = {}
        
        for group_name, group_data in results['demographic_groups'].items():
            bias_data[group_name] = {
                'accuracy': {
                    'aws': group_data.get('avg_accuracy_aws', 0.0),
                    'google': group_data.get('avg_accuracy_google', 0.0)
                },
                'detection_rate': {
                    'aws': group_data['detection_rates']['aws'],
                    'google': group_data['detection_rates']['google']
                },
                'sample_size': group_data['processed_images']
            }
        
        return bias_data
