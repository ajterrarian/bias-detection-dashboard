"""
Interactive visualization tool for facial recognition bias analysis.
Creates heat maps, charts, and dashboards showing performance disparities.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import logging

from config import config

logger = logging.getLogger(__name__)

class BiasVisualization:
    """Class for creating interactive bias visualizations."""
    
    def __init__(self):
        self.color_schemes = {
            'bias_levels': {
                'Low': '#2E8B57',      # Sea Green
                'Moderate': '#FFD700',  # Gold
                'High': '#FF6347',      # Tomato
                'Severe': '#DC143C'     # Crimson
            },
            'demographics': px.colors.qualitative.Set3
        }
    
    def create_accuracy_heatmap(self, bias_data: Dict, services: List[str] = ['aws', 'google']) -> go.Figure:
        """Create heat map showing accuracy across demographic groups."""
        # Prepare data for heatmap
        groups = list(bias_data.keys())
        accuracy_matrix = []
        
        for service in services:
            service_accuracies = []
            for group in groups:
                if group in bias_data and 'accuracy' in bias_data[group]:
                    accuracy = bias_data[group]['accuracy'].get(service, 0.0)
                    service_accuracies.append(accuracy)
                else:
                    service_accuracies.append(0.0)
            accuracy_matrix.append(service_accuracies)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=accuracy_matrix,
            x=groups,
            y=[service.upper() for service in services],
            colorscale='RdYlGn',
            text=[[f'{val:.3f}' for val in row] for row in accuracy_matrix],
            texttemplate='%{text}',
            textfont={'size': 12},
            colorbar=dict(title='Accuracy Score')
        ))
        
        fig.update_layout(
            title='Facial Recognition Accuracy by Demographic Group',
            xaxis_title='Demographic Groups',
            yaxis_title='API Services',
            height=400,
            font=dict(size=12)
        )
        
        return fig
    
    def create_bias_disparity_chart(self, bias_analysis: Dict) -> go.Figure:
        """Create chart showing bias disparity metrics."""
        if 'metrics' not in bias_analysis:
            return go.Figure()
        
        metrics = bias_analysis['metrics']
        
        # Create subplot with multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Accuracy Disparity',
                'Statistical Parity',
                'Detection Rate Variance',
                'Overall Bias Score'
            ],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'indicator'}]]
        )
        
        # Accuracy disparity
        if 'accuracy_disparity' in metrics:
            acc_disp = metrics['accuracy_disparity']
            fig.add_trace(
                go.Bar(
                    x=['Max', 'Min', 'Range', 'Std Dev'],
                    y=[
                        acc_disp.get('max_accuracy', 0),
                        acc_disp.get('min_accuracy', 0),
                        acc_disp.get('accuracy_range', 0),
                        acc_disp.get('std_accuracy', 0)
                    ],
                    name='Accuracy Metrics'
                ),
                row=1, col=1
            )
        
        # Statistical parity
        if 'statistical_parity' in metrics:
            stat_parity = metrics['statistical_parity']
            if 'group_rates' in stat_parity:
                groups = list(stat_parity['group_rates'].keys())
                rates = list(stat_parity['group_rates'].values())
                fig.add_trace(
                    go.Bar(x=groups, y=rates, name='Positive Rates'),
                    row=1, col=2
                )
        
        # Overall bias indicator
        bias_score = bias_analysis.get('overall_bias_score', 0)
        bias_level = bias_analysis.get('bias_level', 'Unknown')
        
        fig.add_trace(
            go.Indicator(
                mode='gauge+number+delta',
                value=bias_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f'Bias Level: {bias_level}'},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': self.color_schemes['bias_levels'].get(bias_level, 'gray')},
                    'steps': [
                        {'range': [0, 0.05], 'color': 'lightgreen'},
                        {'range': [0.05, 0.15], 'color': 'yellow'},
                        {'range': [0.15, 0.30], 'color': 'orange'},
                        {'range': [0.30, 1], 'color': 'red'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 0.30
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Comprehensive Bias Analysis Dashboard',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_demographic_comparison(self, bias_data: Dict, metric: str = 'accuracy') -> go.Figure:
        """Create comparison chart across demographic groups."""
        groups = []
        aws_values = []
        google_values = []
        
        for group, data in bias_data.items():
            if metric in data:
                groups.append(group.replace('_', ' ').title())
                aws_values.append(data[metric].get('aws', 0))
                google_values.append(data[metric].get('google', 0))
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='AWS Rekognition',
            x=groups,
            y=aws_values,
            marker_color='#FF9500'
        ))
        
        fig.add_trace(go.Bar(
            name='Google Cloud Vision',
            x=groups,
            y=google_values,
            marker_color='#4285F4'
        ))
        
        fig.update_layout(
            title=f'{metric.title()} Comparison Across Demographic Groups',
            xaxis_title='Demographic Groups',
            yaxis_title=f'{metric.title()} Score',
            barmode='group',
            height=500
        )
        
        return fig
    
    def create_pairwise_disparity_matrix(self, bias_analysis: Dict) -> go.Figure:
        """Create matrix showing pairwise disparities between groups."""
        if 'metrics' not in bias_analysis or 'accuracy_disparity' not in bias_analysis['metrics']:
            return go.Figure()
        
        pairwise = bias_analysis['metrics']['accuracy_disparity'].get('pairwise_disparities', {})
        
        if not pairwise:
            return go.Figure()
        
        # Extract unique groups
        groups = set()
        for pair in pairwise.keys():
            group1, group2 = pair.split('_vs_')
            groups.add(group1)
            groups.add(group2)
        
        groups = sorted(list(groups))
        n_groups = len(groups)
        
        # Create disparity matrix
        disparity_matrix = np.zeros((n_groups, n_groups))
        
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i != j:
                    pair_key1 = f"{group1}_vs_{group2}"
                    pair_key2 = f"{group2}_vs_{group1}"
                    
                    if pair_key1 in pairwise:
                        disparity_matrix[i][j] = pairwise[pair_key1]
                    elif pair_key2 in pairwise:
                        disparity_matrix[i][j] = pairwise[pair_key2]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=disparity_matrix,
            x=[g.replace('_', ' ').title() for g in groups],
            y=[g.replace('_', ' ').title() for g in groups],
            colorscale='Reds',
            text=[[f'{val:.3f}' if val > 0 else '' for val in row] for row in disparity_matrix],
            texttemplate='%{text}',
            colorbar=dict(title='Accuracy Disparity')
        ))
        
        fig.update_layout(
            title='Pairwise Accuracy Disparities Between Demographic Groups',
            xaxis_title='Demographic Group',
            yaxis_title='Demographic Group',
            height=500
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, bias_data: Dict, bias_analysis: Dict) -> go.Figure:
        """Create comprehensive dashboard with multiple visualizations."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Accuracy Heatmap',
                'Detection Rate Comparison',
                'Bias Metrics Overview',
                'Sample Size Distribution',
                'Service Performance',
                'Bias Level Indicator'
            ],
            specs=[
                [{'type': 'heatmap'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'scatter'}, {'type': 'indicator'}]
            ]
        )
        
        # 1. Accuracy heatmap data
        groups = list(bias_data.keys())
        services = ['aws', 'google']
        accuracy_matrix = []
        
        for service in services:
            service_accuracies = []
            for group in groups:
                accuracy = bias_data[group]['accuracy'].get(service, 0.0)
                service_accuracies.append(accuracy)
            accuracy_matrix.append(service_accuracies)
        
        fig.add_trace(
            go.Heatmap(
                z=accuracy_matrix,
                x=[g.replace('_', ' ').title() for g in groups],
                y=[s.upper() for s in services],
                colorscale='RdYlGn',
                showscale=False
            ),
            row=1, col=1
        )
        
        # 2. Detection rate comparison
        aws_detection = [bias_data[g]['detection_rate']['aws'] for g in groups]
        google_detection = [bias_data[g]['detection_rate']['google'] for g in groups]
        
        fig.add_trace(
            go.Bar(name='AWS', x=[g.replace('_', ' ').title() for g in groups], y=aws_detection),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Google', x=[g.replace('_', ' ').title() for g in groups], y=google_detection),
            row=1, col=2
        )
        
        # 3. Sample size distribution
        sample_sizes = [bias_data[g]['sample_size'] for g in groups]
        fig.add_trace(
            go.Pie(
                labels=[g.replace('_', ' ').title() for g in groups],
                values=sample_sizes,
                name='Sample Sizes'
            ),
            row=2, col=2
        )
        
        # 4. Bias level indicator
        bias_score = bias_analysis.get('overall_bias_score', 0)
        bias_level = bias_analysis.get('bias_level', 'Unknown')
        
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=bias_score,
                title={'text': f'Bias Level: {bias_level}'},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': self.color_schemes['bias_levels'].get(bias_level, 'gray')},
                    'steps': [
                        {'range': [0, 0.05], 'color': 'lightgreen'},
                        {'range': [0.05, 0.15], 'color': 'yellow'},
                        {'range': [0.15, 0.30], 'color': 'orange'},
                        {'range': [0.30, 1], 'color': 'red'}
                    ]
                }
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title='Facial Recognition Bias Analysis Dashboard',
            height=900,
            showlegend=True
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: str, format: str = 'html'):
        """Save visualization to file."""
        output_path = f"{config.output_dir}/{filename}.{format}"
        
        try:
            if format == 'html':
                fig.write_html(output_path)
            elif format == 'png':
                fig.write_image(output_path)
            elif format == 'pdf':
                fig.write_image(output_path)
            
            logger.info(f"Visualization saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
            return None
