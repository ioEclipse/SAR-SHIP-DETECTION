## Project Overview

The SAR Ship Detection Project (BlueGuard) is an AI-powered ship detection system using Synthetic Aperture Radar (SAR) data from Sentinel-1. The system integrates machine learning models for land-sea segmentation, ship detection (YOLOv11m), tracking algorithms, and AIS data integration. It operates through a single Streamlit-based web application host that provides multiple interaction methods: Google Earth Engine area-of-interest selection, user SAR file upload, and future Jetson Nano integration for on-board processing.

## New Architecture Overview:

The system has been refactored to use a single-host architecture where Streamlit serves as both the frontend and backend. All core functionality is centralized in `FullApp/functions.py`, which is called directly by various Streamlit pages. This eliminates the need for separate hosted backend services.

## Core System Components and Workflow:

### 1. Data Acquisition and Management
* **Purpose**: Handle the ingestion of SAR imagery and AIS data from various sources.
* **SAR Sources**:
    * Google Earth Engine integration (user-selected coordinates and time range from interactive map)
    * User-uploaded SAR images (.tiff, .png, .jpeg) with timeframe specification
    * Future: Jetson Nano on-board processing with minimal downlink transmission
* **AIS Sources**:
    * NOAA 2024 AIS data (daily zip folders, downloaded on-demand)
    * Automatic cleanup of cached AIS data to manage storage
* **Key Operations**: Data download, file type handling, temporary storage management, AIS data caching and cleanup

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
* **Data Source**: NOAA 2024 AIS historical data (daily zip archives, downloaded on-demand based on user-specified timeframe).
* **Key Functions**:
    * On-demand download of relevant AIS data based on spatial and temporal requirements
    * Spatial-temporal matching between SAR detections and AIS records
    * Classification of ships as AIS-equipped or potential dark vessels
    * Vessel information enrichment (MMSI, vessel name, type, dimensions)
    * Automatic cleanup of cached AIS files to manage storage
* **Implementation**: Functions in `FullApp/functions.py` for AIS data management, download, and matching

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
    * **Google Earth Engine Integration** (`FullApp/pages/earthEngine.py`): Interactive map for area-of-interest selection with automatic SAR and AIS data download.
    * **SAR File Upload** (`FullApp/pages/app.py`): Upload interface for user-provided SAR images with timeframe specification for AIS matching.
    * **Main Selection** (`FullApp/pages/main.py`): Choice between Google Earth Engine integration and SAR file upload.
    * **Future: Jetson Nano Integration**: On-board processing with minimal data transmission for remote deployment.

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

1. **Preprocessing Pipeline** (`preprocessing/` directory):
   * `noise_filter.py`: Gamma correction and alpha-based enhancement
   * `Land_masking.py`: Advanced land-sea segmentation with multiple algorithms
   * `whole_preprocessing.py`: Complete preprocessing workflow
   * `Yan_segmentation.py`: Additional segmentation capabilities

2. **Legacy AIS Integration** (`ais_detector.py`):
   * Previous real-time streaming implementation (deprecated)
   * Reference implementation for spatial-temporal matching algorithms
   * Data structures for AIS records and SAR detections

3. **New Core Backend** (`FullApp/functions.py`):
   * **Status**: In Development
   * **Purpose**: Centralized backend functionality called directly by Streamlit pages
   * **AIS Functions**: NOAA 2024 data download, caching, and matching (to be implemented)
   * **Integration**: Direct function calls from Streamlit pages without separate API layer

4. **Web Application** (`FullApp/` directory):
   * `home.py`: BlueGuard landing page with branding
   * `pages/main.py`: Selection interface for processing modes
   * `pages/app.py`: SAR file upload interface (updated for timeframe specification)
   * `pages/earthEngine.py`: Google Earth Engine integration with area-of-interest selection
   * `pages/infer2.py`: Inference module with YOLOv11m integration

5. **Inference System**:
   * `InfSlicer.py`: Image slicing and inference coordination
   * `infer2.py`: Complete inference pipeline with crop extraction
   * Integration with Roboflow's inference SDK

### Complete File Structure
```
SAR-SHIP-DETECTION/
├── .dockerignore                                    # Docker ignore file
├── .env                                            # Environment variables file
├── .gitignore                                      # Git ignore file
├── .vscode/                                        # VSCode configuration
│   └── settings.json                               # VSCode settings
├── BlueGuard_Documentation_FirstDraft.pdf         # Project documentation
├── CLAUDE.md                                       # Claude project instructions
├── Dockerfile                                      # Docker container configuration
├── docker-compose.yml                             # Docker compose configuration
├── LICENSE                                         # Project license
├── README.md                                       # Project README
├── ais_detector.py                                 # Legacy AIS detection module
├── config.json                                     # System configuration
├── core_api.py                                     # Core API module (depreciated)
├── InfSlicer.py                                    # Inference coordination
├── requirements.txt                                # Python dependencies
├── test_ais.py                                     # AIS testing module
├── __pycache__/                                    # Python cache directory
│   └── ais_detector.cpython-312.pyc              # Compiled Python file
│
├── FullApp/                                        # Streamlit web application (primary system)
│   ├── Test_image.png                             # Test image file
│   ├── functions.py                               # Core backend functions (NEW - centralized functionality)
│   ├── home.py                                    # Landing page
│   ├── requirements.txt                           # Web app dependencies
│   ├── ship_metadata.json                        # Ship metadata file
│   ├── assets/                                    # Web app assets
│   │   ├── boundingboxes.png                     # UI asset - bounding box illustration
│   │   ├── defaultcontent.png                    # Default content placeholder
│   │   ├── engine1.png                           # Google Earth Engine illustration
│   │   ├── home_background.png                   # Landing page background
│   │   ├── logo.png                              # BlueGuard logo
│   │   ├── preprocessing.png                     # Preprocessing workflow illustration
│   │   ├── raw_img.png                           # Raw image example
│   │   ├── statisticalinsights.png               # Statistics illustration
│   │   └── subimages.png                         # Subimages/crop illustration
│   └── pages/                                     # Streamlit pages
│       ├── app.py                                 # SAR file upload interface
│       ├── earthEngine.py                        # Google Earth Engine integration
│       └── main.py                                # Mode selection page
│
├── Homework/                                       # Development/test directory
│   └── Homework.py                                # Development script
│
├── preprocessing/                                  # Image preprocessing modules
│   ├── __init__.py                                # Package initialization
│   ├── Land_masking.py                           # Advanced land-sea segmentation
│   ├── noise_filter.py                           # Gamma correction and alpha enhancement
│   ├── whole_preprocessing.py                     # Complete preprocessing workflow
│   ├── Yan_segmentation.py                       # Additional segmentation capabilities
│   └── __pycache__/                              # Python cache directory
│       ├── __init__.cpython-312.pyc             # Compiled Python file
│       ├── Land_masking.cpython-312.pyc          # Compiled Python file
│       └── noise_filter.cpython-312.pyc          # Compiled Python file
│
├── tracking/                                       # Tracking algorithms (development)
│   ├── __init__.py                                # Package initialization
│   ├── image_preprocessing.py                     # Image preprocessing for tracking
│   ├── Land_masking.py                           # Land masking for tracking
│   ├── noise_filter.py                           # Noise filtering for tracking
│   ├── processed_image.jpg                       # Processed image example
│   ├── wake_test.jpg                             # Wake detection test image
│   ├── wakedet_ver1.ipynb                        # Wake detection notebook
│   └── __pycache__/                              # Python cache directory
│       ├── Land_masking.cpython-313.pyc          # Compiled Python file
│       └── noise_filter.cpython-313.pyc          # Compiled Python file
│
├── YOLOv11m/                                      # Model artifacts and metrics
│   ├── Metrics-YOLOv11m.docx                     # Model performance metrics document
│   └── Graphs/                                   # Training metrics visualizations
│       ├── BoxF1_curve.png                       # F1 score curve
│       ├── BoxPR_curve.png                       # Precision-Recall curve
│       ├── BoxP_curve.png                        # Precision curve
│       ├── BoxR_curve.png                        # Recall curve
│       ├── confusion_matrix.png                  # Confusion matrix
│       └── confusion_matrix_normalized.png       # Normalized confusion matrix
│
└── sar_env/                                       # Python virtual environment
    └── [virtual environment contents]             # Complete Python environment
```

### File Structure Notes
* **Missing Referenced Files**: The documentation previously referenced `FullApp/pages/infer2.py` as an inference module, but this file does not exist in the current structure, renamed to `functions.py`
* **Docker Support**: Full containerization setup with Dockerfile, docker-compose.yml, and .dockerignore
* **Development Environment**: Includes VSCode configuration, virtual environment, and Python cache files
* **Asset Completeness**: All required assets are present in `FullApp/assets/` including additional UI illustrations
* **Duplicate Modules**: Some preprocessing modules appear in both `preprocessing/` and `tracking/` directories for specialized use cases
* **Git Integration**: Complete git repository with .git/ directory (not shown) containing full version history

## AIS Data Management (NOAA 2024)

The new AIS integration uses NOAA's historical AIS data from 2024, stored in daily zip archives. The system downloads data on-demand based on user requirements.

### AIS Data Characteristics
* **Source**: NOAA Marine Cadastre AIS 2024 data
* **Format**: Daily zip files containing CSV data
* **Coverage**: US coastal waters and inland waterways
* **Size**: Large files requiring on-demand download and cleanup

### AIS Function Requirements (FullApp/functions.py)
The following AIS-related functions need to be implemented:

1. **AIS Data Download**
   * Download specific date ranges of AIS data
   * Handle NOAA API endpoints and zip file extraction
   * Spatial filtering based on area-of-interest

2. **AIS Data Caching and Cleanup**
   * Temporary storage management
   * Automatic cleanup when storage limits reached
   * Cache efficiency optimization

3. **AIS-SAR Matching**
   * Spatial-temporal correlation between SAR detections and AIS records
   * Dark vessel identification (ships not transmitting AIS)
   * Confidence scoring for matches

## Setup Options

The project supports two setup methods: **Docker (Recommended)** for consistent, isolated deployment, and **Local Installation** for development purposes.

### Option 1: Docker Setup (Recommended)

Docker provides a consistent, isolated environment that eliminates dependency conflicts and ensures reproducible deployments across different systems.

#### Prerequisites
* Docker Engine (20.10.0 or higher)
* Docker Compose (2.0.0 or higher)
* Git (for cloning the repository)

#### Quick Start with Docker

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd SAR-SHIP-DETECTION
   ```

2. **Build and Start the Application**:
   ```bash
   # Build and start the container in detached mode
   docker-compose up -d --build
   ```
   
   This command will:
   - Build the Docker image with all dependencies
   - Start the Streamlit web application
   - Make it available at `http://localhost:8501`

3. **Access the Application**:
   Open your web browser and navigate to `http://localhost:8501`

#### Docker Development Workflow

**Starting the Application**:
```bash
# Start in detached mode (runs in background)
docker compose up -d

# OR start with logs visible (useful for debugging)
docker compose up
```

**Viewing Logs**:
```bash
# View real-time logs
docker compose logs -f

# View logs for specific service
docker compose logs -f web
```

**Making Code Changes**:
The Docker setup includes volume mounting, so code changes are automatically reflected:
```bash
# No rebuild needed - changes are automatically picked up
# Just refresh your browser at http://localhost:8501
```

**Rebuilding After Dependency Changes**:
```bash
# Rebuild if you modify requirements.txt or Dockerfile
docker compose up -d --build
```

**Accessing the Container Shell**:
```bash
# Access container for debugging or manual operations
docker compose exec web bash

# OR for a quick command
docker compose exec web python --version
```

**Stopping the Application**:
```bash
# Stop containers (preserves data)
docker compose stop

# Stop and remove containers (clean shutdown)
docker compose down

# Stop, remove containers, and clean up images
docker compose down --rmi local
```

#### Docker Troubleshooting

**Port Already in Use**:
```bash
# If port 8501 is busy, modify docker-compose.yml:
ports:
  - "8502:8501"  # Use port 8502 instead
```

**Container Won't Start**:
```bash
# Check container status
docker compose ps

# View detailed logs
docker compose logs web

# Rebuild completely
docker compose down --rmi local
docker compose up -d --build
```

**Permission Issues (Linux/macOS)**:
```bash
# Fix ownership issues
sudo chown -R $USER:$USER .
```

### Option 2: Local Installation (Development)

For developers who need direct access to the Python environment or want to modify dependencies.

#### Prerequisites
* Python 3.8 or higher
* Git (for cloning the repository)

#### Installation Steps

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

#### Docker (Recommended)
```bash
# Start the application
docker-compose up -d

# Access at http://localhost:8501
# Logs: docker-compose logs -f
# Stop: docker-compose down
```

#### Local Installation
1. **Web Application**:
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

## Streamlit Development Conventions

**IMPORTANT**: Contributors must follow these Streamlit conventions to ensure proper navigation and Docker compatibility.

### File Structure Convention
The Streamlit application follows this **current** structure:

```
FullApp/
├── home.py                    # Landing page entry point (application start point)
├── assets/                    # Static assets (images, CSS, etc.)
├── pages/                     # Additional pages directory
│   ├── main.py               # Mode selection page (moved from root)
│   ├── app.py                # SAR upload interface
│   ├── earthEngine.py        # Google Earth Engine integration
│   └── [other_pages].py      # Additional application pages
└── functions.py               # Core backend functions
```

### Navigation Rules

1. **Application Entry Point**: `FullApp/home.py` serves as the landing page
2. **Main Selection Page**: Located at `FullApp/pages/main.py` (moved from root level)
3. **Page Navigation**: Use `st.switch_page()` with these path patterns:
   - From home.py → main: `st.switch_page("pages/main.py")`
   - From pages/main.py → other pages: `st.switch_page("pages/app.py")`
   - Between pages: `st.switch_page("pages/other_page.py")`

3. **Asset References**: Use relative paths from FullApp/ root:
   - ✅ Correct: `"assets/logo.png"`
   - ❌ Wrong: `"../assets/logo.png"` or `"FullApp/assets/logo.png"`

### Docker Working Directory
The Docker container sets `WORKDIR /app/FullApp`, meaning:
- All relative paths are resolved from `/app/FullApp/`
- Asset references work correctly with `"assets/filename.png"`
- Page navigation follows Streamlit's standard conventions

### Why This Structure?
- **Streamlit Standard**: Follows official Streamlit multipage app conventions
- **Docker Compatibility**: Ensures proper file path resolution in containers
- **Navigation Reliability**: Prevents `StreamlitAPIException` page not found errors
- **Asset Loading**: Consistent relative path resolution for images and resources

### Common Mistakes to Avoid
1. ❌ Using absolute paths for assets: `"/app/FullApp/assets/logo.png"`
2. ❌ Incorrect navigation paths: `st.switch_page("../main.py")` or `st.switch_page("main.py")`
3. ❌ Running Streamlit from wrong working directory in Docker
4. ❌ Confusing entry points: `home.py` is the landing page, `pages/main.py` is the mode selection

### Testing Navigation
Before committing changes that affect navigation:
```bash
# Test locally
cd FullApp
streamlit run home.py

# Test in Docker
docker compose up -d --build
# Verify navigation works at http://localhost:8501
```

### Asset Requirements
The following assets exist in `FullApp/assets/`:
* `logo.png`: BlueGuard logo
* `home_background.png`: Landing page background
* `engine1.png`: Google Earth Engine illustration
* `defaultcontent.png`: Default content placeholder
* `boundingboxes.png`: UI asset - bounding box illustration
* `preprocessing.png`: Preprocessing workflow illustration
* `raw_img.png`: Raw image example
* `statisticalinsights.png`: Statistics illustration
* `subimages.png`: Subimages/crop illustration

All required assets are present and the application should display correctly without missing file errors.

## System Architecture

The BlueGuard system architecture follows a containerized single-host design:

```
                     +------------------------+
                     |  User's Web Browser    |
                     |  (Accesses via URL)    |
                     +-----------+------------+
                                |
                                v
     +------------------------------------------------------+
     |   Docker Container                                   |
     |                                                      |
     |  +----------------------+                            |
     |  | Streamlit Frontend   |  <-- Handles user requests |
     |  +----------+-----------+                            |
     |             |                                        |
     |  +----------v-----------+                            |
     |  | Backend Logic        |  <-- Downloads AIS/SAR,    |
     |  | (Python)             |      imports to InfluxDB,  |
     |  |                      |      queries data          |
     |  +----------+-----------+                            |
     |             |                                        |
     |  +----------v-----------+                            |
     |  |      (local DB)      |  <-- Stores AIS temp data  |
     |  +----------------------+                            |
     |                                                      |
     +------------------------------------------------------+
```

### Architecture Benefits
* **Single Container**: Streamlined deployment with all components in one Docker container
* **Local Processing**: Eliminates external API dependencies for core functionality
* **Scalable Storage**: Efficient handling of large-scale AIS datasets with automatic cleanup 