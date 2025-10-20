# Facial Recognition Bias Detection Tool

An interactive visualization tool that maps facial recognition accuracy across different demographic groups using mathematical bias metrics.

## Project Overview

This tool analyzes bias in facial recognition systems by:
- Collecting accuracy data from AWS Rekognition and Google Cloud Vision APIs
- Applying statistical analysis and differential geometry to quantify bias gradients
- Creating interactive heat maps and dashboards showing performance disparities

## 10-Day Sprint Progress

### ✅ Day 1: Environment Setup & Account Creation
- [x] API account setup (AWS Rekognition, Google Cloud Vision)
- [x] Dependency installation
- [x] Basic API connection testing

### ✅ Day 2: Data Pipeline & Initial Analysis
- [x] Local development environment setup
- [x] Core pipeline implementation
- [x] Mathematical bias metrics framework
- [x] Interactive visualization system

### ✅ Day 3-4: Mathematical Framework Implementation
- [x] Advanced gradient analysis with numerical computation
- [x] Differential geometry for bias surface analysis
- [x] Optimization theory with Pareto frontier computation
- [x] Information theory metrics (mutual information, KL divergence)
- [x] Comprehensive validation suite

### ✅ Day 5-6: Large-Scale Analysis & Visualization Dashboard
- [x] Large-scale bias analysis engine with chunked processing
- [x] React.js dashboard with Material-UI design
- [x] Interactive Plotly.js visualizations
- [x] Mathematical notation rendering with KaTeX
- [x] Three analysis tabs: Overview, Mathematical, Statistical

### ✅ Day 7: Dashboard Integration & Real-time Features
- [x] FastAPI backend with comprehensive REST endpoints
- [x] WebSocket integration for real-time analysis updates
- [x] React API service with socket.io-client integration
- [x] Live analysis controls and server status monitoring
- [x] Visualization data endpoints (heatmap, gradients, ROC curves)
- [x] Export features (PDF, CSV, LaTeX, PNG)
- [x] Full-stack integration testing

### ✅ Day 8: Advanced Features & Optimization
- [x] BiasMitigationSuite with mathematical bias reduction methods
- [x] Post-processing approaches (threshold optimization, calibration)
- [x] Optimization-based methods (Lagrange multipliers, multi-objective)
- [x] Mathematical transformations (feature space, probability distribution)
- [x] Validation methods (bootstrap, permutation tests, cross-validation)
- [x] Performance optimization (vectorization, parallel processing, caching)

### 📅 Upcoming Days 9-10:
- Day 9: Production Deployment & Integration Testing
- Day 10: Final Documentation & Project Completion

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API credentials:
```bash
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

## Usage

### FastAPI Backend Server
Start the FastAPI server:
```bash
python Day7_FastAPI_Backend.py
```

This starts the backend server on `http://127.0.0.1:8000` with:
- Comprehensive REST API endpoints for bias analysis
- WebSocket support for real-time updates
- Interactive API documentation at `/docs`
- Export capabilities (PDF, CSV, LaTeX, PNG)
- Performance optimization and caching

### React Dashboard
Start the React dashboard (requires Node.js):
```bash
cd dashboard
npm install  # First time only
npm start
```

The dashboard will be available at `http://localhost:3000` and automatically connects to the Python backend.

### Complete Analysis Pipeline
Run the complete analysis pipeline:
```bash
python main.py
```

This will:
1. Test API connections
2. Load sample diverse face dataset
3. Process images through facial recognition APIs
4. Calculate bias metrics using statistical analysis
5. Generate interactive visualizations

## Output

The tool generates several HTML visualizations in the `output/` directory:
- `accuracy_heatmap.html` - Heat map of accuracy across demographic groups
- `bias_disparity_chart.html` - Comprehensive bias metrics dashboard
- `demographic_comparison.html` - Side-by-side service comparisons
- `comprehensive_dashboard.html` - Complete analysis dashboard

## Bias Metrics Implemented

### Statistical Measures
- **Accuracy Disparity**: Range and variance of accuracy across groups
- **Statistical Parity**: Difference in positive prediction rates
- **Equalized Odds**: True/false positive rate differences
- **Demographic Parity**: Equal treatment across groups

### Mathematical Framework
- **Differential Geometry**: Bias gradient calculations using surface interpolation
- **Individual Fairness**: Similarity-based outcome consistency
- **Pairwise Disparities**: All group-to-group comparisons

## Architecture

```
├── config.py              # Configuration and API credentials
├── api_client.py           # AWS Rekognition & Google Cloud Vision clients
├── data_pipeline.py        # Dataset loading and processing pipeline
├── bias_metrics.py         # Mathematical bias calculations
├── visualization.py        # Interactive Plotly visualizations
├── main.py                # Main execution orchestrator
└── requirements.txt        # Python dependencies
```

## API Services

- **AWS Rekognition**: Face detection with demographic attributes
- **Google Cloud Vision**: Face detection with emotion/landmark analysis
- **Azure Face API**: Excluded due to connection issues

## Sample Dataset

Uses diverse, publicly available portrait images from Unsplash covering:
- Age groups: Young, middle-aged, elderly
- Gender: Male, female
- Skin tones: Light, medium, dark

## Bias Analysis Results

The tool provides:
- Overall bias score (0-1 scale)
- Bias level classification (Low/Moderate/High/Severe)
- Service-specific performance metrics
- Demographic group comparisons
- Statistical significance testing

## Next Steps

Continue with Days 3-4 to implement advanced mathematical frameworks and expand the dataset for more comprehensive analysis.
