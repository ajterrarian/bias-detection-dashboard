#!/usr/bin/env python3
"""
Day 7: Backend API Integration for Facial Recognition Bias Detection
==================================================================

EXPLANATION FOR BEGINNERS:
This creates a REST API server that connects your React dashboard to the Python
bias analysis engine. It provides endpoints for running analysis and getting results.
"""

import os
import json
import time
import uuid
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename

# Import our bias analysis modules (with fallbacks)
try:
    from bias_metrics import BiasAnalyzer
    from api_client import APIClient
    print("✅ Core modules imported")
except ImportError as e:
    print(f"⚠️ Core import warning: {e}")
    # Create mock classes for testing
    class BiasAnalyzer:
        def calculate_bias_metrics(self, demo_array, accuracy_array):
            return {'mock': True, 'bias_score': 0.3}
    
    class APIClient:
        def __init__(self):
            self.aws_client = None
            self.google_client = None

try:
    from Day5_Large_Scale_Bias_Analysis import LargeScaleBiasAnalyzer
    print("✅ Large scale analyzer imported")
except ImportError:
    print("⚠️ Large scale analyzer not available - using mock")
    class LargeScaleBiasAnalyzer:
        def run_comprehensive_analysis(self, dataset_path, output_dir):
            return {'mock': True, 'analysis': 'simulated'}

class BiasAnalysisAPI:
    """Flask API server for bias analysis."""
    
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'bias-detection-secret-key'
        self.app.config['UPLOAD_FOLDER'] = './uploads'
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
        
        # Enable CORS for React frontend
        CORS(self.app, origins=["http://localhost:3000"])
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize analysis components
        try:
            self.api_client = APIClient()
            self.bias_analyzer = BiasAnalyzer()
            self.large_scale_analyzer = LargeScaleBiasAnalyzer()
        except:
            print("⚠️ Some analysis components not available")
        
        # Active analysis sessions
        self.active_sessions = {}
        
        # Create upload directory
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Setup routes
        self._setup_routes()
        self._setup_websocket_handlers()
        
        print("🚀 Bias Analysis API Server initialized!")
    
    def _setup_routes(self):
        """Setup REST API endpoints."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'aws_rekognition': hasattr(self, 'api_client') and self.api_client.aws_client is not None,
                    'google_vision': hasattr(self, 'api_client') and self.api_client.google_client is not None
                }
            })
        
        @self.app.route('/api/upload', methods=['POST'])
        def upload_image():
            """Upload and analyze a single image."""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Analyze image
                session_id = str(uuid.uuid4())
                analysis_result = self._analyze_single_image(file_path, session_id)
                
                return jsonify({
                    'session_id': session_id,
                    'filename': filename,
                    'analysis': analysis_result
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/live', methods=['GET'])
        def get_live_metrics():
            """Get live bias metrics from latest analysis."""
            try:
                # Find latest results file
                results_dir = './results'
                if not os.path.exists(results_dir):
                    return jsonify({'error': 'No analysis results found'}), 404
                
                json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
                if not json_files:
                    return jsonify({'error': 'No analysis results found'}), 404
                
                latest_file = max(json_files, key=lambda f: os.path.getctime(
                    os.path.join(results_dir, f)
                ))
                
                with open(os.path.join(results_dir, latest_file), 'r') as f:
                    data = json.load(f)
                
                # Extract key metrics for dashboard
                metrics = self._extract_dashboard_metrics(data)
                
                return jsonify({
                    'metrics': metrics,
                    'timestamp': data.get('metadata', {}).get('timestamp'),
                    'source_file': latest_file
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analysis/start', methods=['POST'])
        def start_analysis():
            """Start new bias analysis."""
            try:
                data = request.get_json()
                analysis_type = data.get('type', 'quick')
                
                session_id = str(uuid.uuid4())
                
                # Start analysis in background
                thread = threading.Thread(
                    target=self._run_background_analysis,
                    args=(analysis_type, session_id)
                )
                thread.start()
                
                return jsonify({
                    'session_id': session_id,
                    'status': 'started',
                    'type': analysis_type
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"🔌 Client connected: {request.sid}")
            emit('status', {'message': 'Connected to bias analysis server'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"🔌 Client disconnected: {request.sid}")
    
    def _analyze_single_image(self, image_path: str, session_id: str) -> Dict[str, Any]:
        """Analyze a single uploaded image."""
        
        try:
            # Mock analysis for demo (replace with actual analysis)
            demographic_info = {
                'age': np.random.randint(18, 80),
                'gender': np.random.choice(['Male', 'Female']),
                'ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Hispanic']),
                'confidence': np.random.uniform(0.7, 0.95)
            }
            
            # Calculate simple bias score
            bias_score = self._calculate_bias_score(demographic_info)
            
            return {
                'demographic_info': demographic_info,
                'bias_score': bias_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_bias_score(self, demographic_info: Dict) -> float:
        """Calculate bias score for demographic info."""
        
        base_confidence = demographic_info.get('confidence', 0.5)
        
        # Apply bias factors
        age_factor = 0.9 if demographic_info.get('age', 25) > 65 else 1.0
        gender_factor = 0.95 if demographic_info.get('gender') == 'Female' else 1.0
        ethnicity_factor = 0.92 if demographic_info.get('ethnicity') in ['Black', 'Hispanic'] else 1.0
        
        expected_confidence = base_confidence * age_factor * gender_factor * ethnicity_factor
        bias_score = abs(base_confidence - expected_confidence)
        
        return float(bias_score)
    
    def _run_background_analysis(self, analysis_type: str, session_id: str):
        """Run analysis in background thread."""
        
        try:
            self.active_sessions[session_id] = {
                'status': 'running',
                'start_time': datetime.now().isoformat(),
                'type': analysis_type
            }
            
            if analysis_type == 'large_scale':
                # Run large-scale analysis
                results = self.large_scale_analyzer.run_comprehensive_analysis(
                    dataset_path="./results/sample_results.json",
                    output_dir="./results"
                )
            else:
                # Quick analysis
                results = self._generate_sample_metrics()
            
            # Update session
            self.active_sessions[session_id].update({
                'status': 'completed',
                'results': results,
                'end_time': datetime.now().isoformat()
            })
            
            # Emit completion
            self.socketio.emit('analysis_complete', {
                'session_id': session_id,
                'results': self._extract_dashboard_metrics(results)
            })
            
        except Exception as e:
            self.active_sessions[session_id] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.socketio.emit('analysis_error', {
                'session_id': session_id,
                'error': str(e)
            })
    
    def _generate_sample_metrics(self) -> Dict[str, Any]:
        """Generate sample metrics for testing."""
        
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'sample',
                'total_records': 100
            },
            'global_analysis': {
                'bias_severity': {
                    'overall_score': np.random.uniform(0.2, 0.8),
                    'severity_level': np.random.choice(['Low', 'Moderate', 'High'])
                },
                'demographic_balance': {
                    'age_balance_score': np.random.uniform(0.6, 0.9),
                    'gender_balance_score': np.random.uniform(0.7, 0.95),
                    'ethnicity_balance_score': np.random.uniform(0.5, 0.8)
                }
            },
            'combined_results': {
                'basic_metrics': {
                    'accuracy_disparity': {
                        'max_difference': np.random.uniform(0.1, 0.3),
                        'coefficient_of_variation': np.random.uniform(0.05, 0.15)
                    }
                },
                'advanced_metrics': {
                    'gradient_analysis': {
                        'magnitude': np.random.uniform(0.1, 0.5)
                    }
                }
            }
        }
    
    def _extract_dashboard_metrics(self, analysis_data: Dict) -> Dict[str, Any]:
        """Extract key metrics for dashboard display."""
        
        metrics = {
            'overview': {},
            'mathematical': {},
            'statistical': {}
        }
        
        try:
            # Overview metrics
            if 'global_analysis' in analysis_data:
                global_data = analysis_data['global_analysis']
                metrics['overview'] = {
                    'overall_bias_score': global_data.get('bias_severity', {}).get('overall_score', 0),
                    'bias_level': global_data.get('bias_severity', {}).get('severity_level', 'Unknown'),
                    'total_records': analysis_data.get('metadata', {}).get('total_records', 0),
                    'demographic_balance': global_data.get('demographic_balance', {})
                }
            
            # Mathematical metrics
            if 'combined_results' in analysis_data:
                combined = analysis_data['combined_results']
                if 'advanced_metrics' in combined:
                    adv_metrics = combined['advanced_metrics']
                    metrics['mathematical'] = {
                        'gradient_magnitude': adv_metrics.get('gradient_analysis', {}).get('magnitude', 0),
                        'curvature_analysis': adv_metrics.get('differential_geometry', {}),
                        'optimization_metrics': adv_metrics.get('optimization', {})
                    }
            
            # Statistical metrics
            if 'combined_results' in analysis_data:
                combined = analysis_data['combined_results']
                if 'basic_metrics' in combined:
                    basic_metrics = combined['basic_metrics']
                    metrics['statistical'] = {
                        'accuracy_disparity': basic_metrics.get('accuracy_disparity', {}),
                        'group_statistics': basic_metrics.get('group_accuracies', {}),
                        'statistical_tests': basic_metrics.get('statistical_tests', {})
                    }
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Start the API server."""
        print(f"🌐 Starting Bias Analysis API Server on {host}:{port}")
        print(f"📊 Dashboard will connect to: http://{host}:{port}")
        
        self.socketio.run(self.app, host=host, port=port, debug=debug)

# ============================================================================
# CELL 2: API Integration Test
# ============================================================================

def test_api_integration():
    """Test the API integration."""
    print("🧪 Testing API Integration")
    print("=" * 40)
    
    # Initialize API server
    api_server = BiasAnalysisAPI()
    
    print("✅ API server initialized successfully!")
    print("🌐 Ready to serve React dashboard requests")
    
    return api_server

# ============================================================================
# CELL 3: Run API Server
# ============================================================================

def run_day7_api_server():
    """
    Run Day 7 API server for dashboard integration.
    
    EXPLANATION FOR BEGINNERS:
    - This starts a web server that your React dashboard can connect to
    - Provides REST API endpoints for bias analysis
    - Enables real-time updates via WebSockets
    """
    print("🌅 DAY 7: Backend API Integration")
    print("=" * 50)
    
    # Initialize and run API server
    api_server = BiasAnalysisAPI()
    
    print("\n🚀 Starting API server...")
    print("📱 React dashboard can now connect!")
    print("🔗 API endpoints available at: http://127.0.0.1:5000/api/")
    
    # Start server
    api_server.run(host='127.0.0.1', port=5000, debug=True)

if __name__ == "__main__":
    run_day7_api_server()

print("✅ Day 7: Backend API Integration Complete!")
