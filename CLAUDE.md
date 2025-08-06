## Project Overview

The SAR Ship Detection Project (BlueGuard) is an AI-powered ship detection system using Synthetic Aperture Radar (SAR) data from Sentinel-1. The system integrates machine learning models for land-sea segmentation, ship detection (YOLOv11m), tracking algorithms, and AIS data integration. It offers multiple user interfaces through a Streamlit-based web application: raw SAR data processing, Google Earth Engine integration for map-based analysis, and comprehensive ship detection with metadata export capabilities.

## Core System Components and Workflow:

### 1. Data Acquisition and Management
* **Purpose**: Handle the ingestion of SAR imagery from various sources.
* **Sources**:
    * Copernicus API (based on user-selected coordinates and time range from a map).
    * User-uploaded SAR images (.tiff, .png, .jpeg).
    * Preloaded raw SAR image files (for Jetson Nano).
* **Key Operations**: Data download, file type handling, temporary storage management (local server, isolated sessions, cleanup).

### 2. Preprocessing Subsystem
* **Purpose**: Transform raw SAR images to enhance features, reduce noise, and prepare for machine learning inference.
* **Key Steps (applied sequentially)**:
    * **Polarimetric Band Selection**: Prioritize VV polarization; fall back to others if VV is not available.
    * **Amplitude Scaling & Grayscale Conversion**: Amplify VV returns (factor of 2), convert to grayscale (duplicate across RGB channels).
    * **Noise Reduction & Edge Enhancement**: Apply Gamma Correction and an Alpha-based enhancement filter.
    * **Land Masking**: Remove land areas from consideration while keeping them visible in the output image. This involves multiple techniques like Lee Filter, CLAHE, Dual Thresholding, Morphological Processing, Flood-Fill, and Contour Filtering, with iterative refinement and land coverage assessment (discard if >85% land, refine if >15% land).
    * **Format Conversion and Standardization**: Convert to normalized NumPy array (0-1).

### 3. Ship Detection Inference (Machine Learning Model)
* **Purpose**: Identify ship instances within the preprocessed SAR imagery.
* **Model**: YOLOv11m (preliminary model mentioned, trained with PyTorch, weights converted to ONNX/TensorRT).
* **Output**: Annotated image and JSON file with ship geographical coordinates, confidence scores, and bounding box information. The model only detects presence of ships, not specific types.

### 4. Tracking Algorithm
* **Purpose**: Maintain persistent ship identities across sequential SAR images.
* **Methodology**: DeepSort (or similar multi-object tracking).
* **Integration**: Combine detection outputs with temporal association and appearance features. Potential fusion with AIS data for enhanced accuracy.

### 4.1 AIS Integration Module
* **Purpose**: Integrate Automatic Identification System (AIS) data with SAR detections to identify AIS-equipped vs non-AIS vessels (dark vessels).
* **Data Source**: Real-time AIS data from aisstream.io API via WebSocket connection.
* **Key Functions**:
    * Spatial-temporal matching between SAR detections and AIS transmissions
    * Classification of ships as AIS-equipped or potential dark vessels
    * Vessel information enrichment (MMSI, vessel name, type, dimensions)
* **Implementation**: `ais_detector.py` - AISDetector class with real-time streaming capabilities

### 5. Output Generation and Visualization
* **Purpose**: Compile and present results to the user through various interfaces.
* **Output Formats**:
    * Structured JSON data (bounding box coordinates, width, class label, unique ID).
    * Annotated images (full image with detected ships, individual ship images by ID).
    * Statistical insights (total ships detected, estimated sizes, pixel areas, surface estimates).
    * Downloadable reports in PNG/JPEG formats.
    * Comprehensive ship characteristics table with metadata.
* **User Interfaces (Streamlit-based BlueGuard Dashboard)**:
    * **Home Interface** (`FullApp/home.py`): Landing page with branding and navigation.
    * **Raw SAR Processing** (`FullApp/pages/app.py`): Upload interface for processing user-provided SAR images.
    * **Google Earth Engine Integration** (`FullApp/pages/earthEngine.py`): Interactive map for area-of-interest selection.
    * **Main Selection** (`FullApp/pages/main.py`): Choice between raw data processing and GEE integration.

### 6. System Management and User Experience
* **Queue Management**: Implement a FIFO queue for incoming processing requests to manage server load. Provide user feedback on queue position and estimated wait time.
* **Error Handling**: Gracefully handle unexpected inputs and provide informative error messages.
* **Data Integrity**: Ensure integrity of processed data and output files.
* **Concurrency**: Handle a limited number of concurrent users.
* **Session State Management**: Streamlit session state for maintaining user data across interactions.
* **File Management**: Automatic cleanup of temporary uploaded files and processing results.
* **Asset Management**: Organized asset structure with logos, backgrounds, and example images.

## Current Implementation Status

### Completed Components

1. **Core System Structure** (`main.py`):
   * `SARShipDetectionSystem` class with modular architecture
   * `ConfigurationManager`, `DataHandler`, `ShipDetector`, `AISDetector` classes
   * `OutputGenerator`, `ProcessingQueue`, `StreamlitInterface` classes
   * Functional image processing pipeline with noise reduction and land masking

2. **Preprocessing Pipeline** (`preprocessing/` directory):
   * `noise_filter.py`: Gamma correction and alpha-based enhancement
   * `Land_masking.py`: Advanced land-sea segmentation with multiple algorithms
   * `whole_preprocessing.py`: Complete preprocessing workflow
   * `Yan_segmentation.py`: Additional segmentation capabilities

3. **AIS Integration** (`ais_detector.py`):
   * Complete AISDetector class with real-time streaming
   * WebSocket connection to aisstream.io API
   * Spatial-temporal matching algorithms
   * Data structures for AIS records and SAR detections
   * Command-line testing interface

4. **Web Application** (`FullApp/` directory):
   * `home.py`: BlueGuard landing page with branding
   * `pages/main.py`: Selection interface for processing modes
   * `pages/app.py`: Complete raw SAR processing interface
   * `pages/earthEngine.py`: Google Earth Engine integration
   * `pages/infer2.py`: Inference module with YOLOv11m integration

5. **Inference System**:
   * `InfSlicer.py`: Image slicing and inference coordination
   * `infer2.py`: Complete inference pipeline with crop extraction
   * Integration with Roboflow's inference SDK

### File Structure
```
├── main.py                     # Main system orchestrator
├── ais_detector.py            # AIS integration module
├── config.json                # System configuration
├── requirements.txt           # Python dependencies
├── InfSlicer.py              # Inference coordination
├── preprocessing/             # Image preprocessing modules
│   ├── noise_filter.py
│   ├── Land_masking.py
│   ├── whole_preprocessing.py
│   └── Yan_segmentation.py
├── tracking/                  # Tracking algorithms (development)
├── FullApp/                   # Streamlit web application
│   ├── home.py               # Landing page
│   ├── requirements.txt      # Web app dependencies
│   ├── assets/               # Images, logos, backgrounds
│   └── pages/
│       ├── main.py          # Mode selection
│       ├── app.py           # Raw data processing
│       ├── earthEngine.py   # GEE integration
│       └── infer2.py        # Inference module
└── YOLOv11m/                 # Model artifacts and metrics
    └── Graphs/               # Training metrics
```

## AIS Integration Testing

The AIS integration module can be tested independently using the aisstream.io API:

### Prerequisites
* Get free API key from https://aisstream.io/
* Install dependencies: `websockets`, `requests`

### Test Usage

**Command Line Test:**
```bash
python ais_detector.py --test-api YOUR_API_KEY
```

**Python Function Test:**
```python
from ais_detector import test_aisstream_api

# Test with default San Francisco Bay area
test_aisstream_api('your_api_key_here')

# Test with custom geographic area (min_lat, min_lon, max_lat, max_lon)
bbox = (40.0, -74.5, 41.0, -73.5)  # New York Harbor
test_aisstream_api('your_api_key_here', bbox)
```

The test function collects AIS data for 1 minute and displays sample vessel records including MMSI, position, speed, course, and vessel information.

## First Time Setup

### Prerequisites
* Python 3.8 or higher
* Git (for cloning the repository)

### Installation Steps

1. **Clone or Pull the Repository**:
   ```bash
   git clone <repository-url>
   cd SAR-SHIP-DETECTION
   ```
   Or if you already have the repo:
   ```bash
   git pull origin main
   ```

2. **Create a Virtual Environment (Recommended)**:
   ```bash
   python -m venv sar_env
   
   # Activate the virtual environment:
   # On Windows:
   sar_env\Scripts\activate
   # On macOS/Linux:
   source sar_env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   # Install main project dependencies
   pip install -r requirements.txt
   
   # OR install web application dependencies specifically
   pip install -r FullApp/requirements.txt
   ```

4. **Configuration Setup**:
   * Copy the example config below
   * Add your AIS stream API key to `config.json`:
     ```json
     {
       "aisstream": {
         "api_key": "your_aisstream_api_key_here"
       },
         "ais_detector": {
            "spatial_threshold_meters": 500.0,
            "temporal_threshold_seconds": 3600.0,
            "min_match_confidence": 0.7
         },
         "demo_locations": {
            "san_francisco_bay": [37.0, -122.5, 38.0, -121.5],
            "new_york_harbor": [40.0, -74.5, 41.0, -73.5],
            "english_channel": [50.0, -1.0, 51.0, 2.0],
            "singapore_strait": [1.0, 103.5, 1.5, 104.5]
         }
     }
     ```
   * Alternatively, set environment variable:
     ```bash
     export AISSTREAM_API_KEY="your_api_key_here"
     ```

5. **Test the Installation**:
   ```bash
   # Test the main system
   python main.py
   
   # Test AIS integration (requires API key)
   python ais_detector.py --test-api YOUR_API_KEY
   
   # Run the web application
   cd FullApp
   streamlit run home.py
   ```

### Key Dependencies Explained
* **streamlit**: Web application framework
* **opencv-python**: Computer vision and image processing
* **numpy, scipy**: Numerical computing
* **matplotlib**: Plotting and visualization
* **folium, streamlit_folium**: Interactive maps
* **websockets**: AIS real-time data streaming
* **inference-sdk**: YOLOv11m model inference
* **pillow**: Image processing
* **pandas**: Data manipulation and analysis

### Running the Application

1. **Web Application (Recommended)**:
   ```bash
   cd FullApp
   streamlit run home.py
   ```
   Access at `http://localhost:8501`

2. **Command Line Processing**:
   ```bash
   python main.py  # Runs main processing pipeline
   ```

3. **AIS Testing**:
   ```bash
   python ais_detector.py --test-api YOUR_API_KEY
   ```

### Troubleshooting

#### Import Errors in VSCode
* **Issue**: VSCode shows "Import cannot be resolved (Pylance)" even after pip install
* **Solution**: 
  1. Ensure you're using the correct Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
  2. Select the interpreter from your virtual environment (should show `venv` in the path)
  3. Restart VSCode after selecting the correct interpreter
  4. Ensure `__init__.py` files exist in module directories (now included)

#### AIS Connection Issues
* **Issue**: "Connected to AISStream.io" followed immediately by "WebSocket error: Connection to remote host was lost"
* **Root Cause**: Wrong WebSocket package - use `websockets` not `websocket-client`
* **Common Causes & Solutions**:
  1. **Wrong Package**: Install `websockets==12.0` instead of `websocket-client`
  2. **Invalid API Key**: Ensure you're using a valid API key from https://aisstream.io/
  3. **Placeholder API Key**: Don't use "your_api_key_here" - replace with actual key
  4. **Network/Firewall**: Corporate firewalls may block WebSocket connections
  5. **API Key Format**: Ensure no extra spaces or characters in the API key
* **Testing**: The updated code uses async/await with proper `websockets` package

#### Other Issues
* **Streamlit Issues**: Try `pip install --upgrade streamlit`
* **OpenCV Issues**: Try `pip install opencv-python-headless` instead of `opencv-python`
* **Module Not Found**: Ensure you're running commands from the correct directory
* **Virtual Environment**: Always activate your virtual environment before running commands

### Development Setup

For development, install additional tools:
```bash
pip install black flake8 pytest  # Code formatting and testing
```

### Asset Requirements
Ensure the following assets exist in `FullApp/assets/`:
* `logo.png`: BlueGuard logo
* `home_background.png`: Landing page background
* `ship.png`: Ship icon
* `engine1.png`: Google Earth Engine illustration
* `defaultcontent.png`: Default content placeholder
* `functionalities.png`: Features overview

If assets are missing, the application will show error messages for missing files.