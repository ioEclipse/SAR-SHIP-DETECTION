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

### 4. AIS Integration Module
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

   **Local Model Inference** (`FullApp/local_inference.py`):
   * **Status**: Completed - Replaces Roboflow model
   * **Purpose**: Local YOLO inference using trained YOLOv11m model (best.pt)
   * **Features**: Roboflow-compatible interface, fallback mode, automatic model loading
   * **Dependencies**: ultralytics>=8.0.0

4. **Web Application** (`FullApp/` directory):
   * `home.py`: BlueGuard landing page with branding
   * `pages/main.py`: Selection interface for processing modes
   * `pages/app.py`: SAR file upload interface (updated for timeframe specification)
   * `pages/earthEngine.py`: Google Earth Engine integration with area-of-interest selection
5. **Inference System**:
   * `utilities/InfSlicer.py`: Large image slicing and inference coordination (modular, CLI-ready)
   * `FullApp/local_inference.py`: Local YOLO model inference replacing Roboflow SDK
   * `utilities/test_local_model.py`: Integration testing for local model setup
   * `utilities/onnxtrasnform.py`: Model conversion utility (PyTorch to ONNX)

### Complete File Structure
```
SAR-SHIP-DETECTION/
├── .dockerignore                                    # Docker ignore file
├── .gitignore                                      # Git ignore file
├── BlueGuard_Documentation_FirstDraft.docx        # Project documentation (Word)
├── BlueGuard_Documentation_FirstDraft.pdf         # Project documentation (PDF)
├── CLAUDE.md                                       # Claude project instructions
├── Dockerfile                                      # Docker container configuration (web_app service)
├── docker-compose.yml                             # Docker compose configuration (dual-container)
├── LICENSE                                         # Project license
├── README.md                                       # Project README
├── config.json                                     # System configuration (Google Earth Engine, AIS data)
├── utilities/                                      # Utility scripts and tools
│   ├── InfSlicer.py                               # Large image slicing and inference coordination
│   ├── onnxtrasnform.py                          # Model conversion utility (PyTorch to ONNX)
│   └── test_local_model.py                       # Local YOLO model integration test
├── requirements.txt                                # Python dependencies (includes ultralytics>=8.0.0)
├── test_ais.py                                     # AIS testing module
│
├── Ais_data/                                       # AIS data storage (auto-managed)
│   └── *.zip                                      # Downloaded NOAA AIS data files
│
├── FullApp/                                        # Streamlit web application (primary system)
│   ├── Test_image.png                             # Test image file
│   ├── auth.py                                    # Authentication utilities
│   ├── engineAPI1.py                             # Google Earth Engine API integration
│   ├── functions.py                               # Core backend functions (centralized functionality)
│   ├── home.py                                    # Landing page entry point
│   ├── local_inference.py                        # Local YOLO model inference (replaces Roboflow)
│   ├── noise_filter.py                           # Noise filtering utilities
│   ├── ship_metadata.json                        # Generated ship metadata (user results)
│   ├── assets/                                    # Web app static assets
│   │   ├── boundingboxes.png                     # UI asset - bounding box illustration
│   │   ├── defaultcontent.png                    # Default content placeholder
│   │   ├── ee_earth_satellite.png                # Earth Engine satellite illustration
│   │   ├── home_background.png                   # Landing page background
│   │   ├── insights_background.png               # Insights page background
│   │   ├── logo.png                              # BlueGuard logo
│   │   ├── preprocessing.png                     # Preprocessing workflow illustration
│   │   ├── raw_img.png                           # Raw image example
│   │   ├── raw_img_2.png                         # Raw image example 2
│   │   ├── statisticalinsights.png               # Statistics illustration
│   │   ├── stats4.png                            # Statistics chart example
│   │   └── subimages.png                         # Subimages/crop illustration
│   ├── images/                                    # Generated output images
│   │   └── sar_sentinel1_jpg/                    # SAR processing results
│   │       ├── *.jpg                             # Original, corrected, and detection images
│   │       ├── *_metadata.json                   # Processing metadata
│   │       └── crops/                            # Individual ship crop images
│   └── pages/                                     # Streamlit pages
│       ├── app.py                                 # SAR file upload interface
│       ├── earthEngine.py                        # Google Earth Engine integration
│       ├── earthEngine1.py                       # Alternative Earth Engine interface
│       ├── earthEngineDesign.py                  # Earth Engine design variations
│       ├── insights.py                           # Statistical insights page
│       └── main.py                                # Mode selection page
│
├── Jetson AGX Orin/                               # Edge deployment container (jetson_app service)
│   ├── Dockerfile                                 # Jetson-optimized container configuration
│   ├── Final_function.py                         # Edge processing pipeline
│   ├── Test_image.png                            # Test image for edge inference
│   ├── best1.pt                                  # Optimized model weights for Jetson
│   ├── inference_with_ONNX.py                    # ONNX-based inference for GPU acceleration
│   ├── onnxtrasnform.py                          # Model conversion utility (PyTorch to ONNX)
│   └── ourmodel_inference_function.py            # Edge-optimized inference functions
│
├── debug_images/                                   # Debug and test images (organized)
│   └── preprocessing_tests/                       # Preprocessing debug outputs
│       ├── blured.jpg                            # Blur filter test
│       ├── dark*.jpg                             # Dark enhancement tests
│       ├── denoise.jpg                           # Denoising test
│       ├── enlighten*.jpg                        # Enlightenment filter tests
│       ├── masked.jpg                            # Land masking test
│       ├── morph.jpg                             # Morphological operations test
│       ├── no_land.jpg                           # Land removal test
│       └── tresh.jpg                             # Thresholding test
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
│   ├── test.jpg                                  # Test image
│   ├── ts.png                                    # Test image PNG
│   └── vv2.jpg                                   # VV polarization test image
│
├── tracking/                                       # Tracking algorithms (development)
│   ├── wakedet_ver1.ipynb                        # Wake detection notebook
│   ├── Original Image/                           # Original test images
│   │   ├── *.jpg                                 # Test images (11.jpg, 22.jpg, etc.)
│   │   ├── Land_masking.py                       # Land masking for tracking
│   │   ├── det.py                                # Detection utilities
│   │   ├── detector.ipynb                        # Detection notebook
│   │   ├── image_preprocessing.py                # Image preprocessing for tracking
│   │   ├── noise_filter.py                       # Noise filtering for tracking
│   │   ├── processed_image.jpg                   # Processed image example
│   │   ├── wake_test.jpg                         # Wake detection test image
│   │   └── *.png                                 # Processing result images
│   └── phase/                                    # Phase analysis
│       ├── phase*.png                            # Phase analysis results
│       └── radon.ipynb                           # Radon transform notebook
│
├── YOLOv11m/                                      # Model artifacts and metrics
│   ├── best.pt                                   # Trained model weights
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
* **Dual-Container Architecture**: The project implements a dual-container Docker setup with `web_app` and `jetson_app` services for different deployment scenarios
* **Edge Deployment**: `Jetson AGX Orin/` directory contains self-contained edge processing optimized for NVIDIA Jetson AGX Orin with GPU acceleration
* **Missing Referenced Files**: The documentation previously referenced `FullApp/pages/infer2.py` as an inference module, but this file does not exist in the current structure, renamed to `functions.py`
* **Docker Support**: Full containerization setup with Dockerfile, docker-compose.yml, and .dockerignore
* **Development Environment**: Includes VSCode configuration, virtual environment, and Python cache files
* **Asset Completeness**: All required assets are present in `FullApp/assets/` including additional UI illustrations
* **Duplicate Modules**: Some preprocessing modules appear in both `preprocessing/` and `tracking/` directories for specialized use cases
* **Git Integration**: Complete git repository with .git/ directory (not shown) containing full version history

## Docker Container Architecture

The BlueGuard system implements a dual-container architecture designed to support both web-based operations and edge deployment scenarios:

### Container Layout Structure

The repository contains two distinct Docker services defined in `docker-compose.yml`:

#### 1. **web_app Service** (Primary Web Application)
```yaml
web_app:
  build:
    context: .
    args:
      BASE_IMAGE: python:3.11-slim
  ports:
    - "8501:8501"
```

**Container Contents:**
- **Inclusion**: Everything in the repository root except `.dockerignore` exclusions and the `Jetson AGX Orin/` folder
- **Purpose**: Streamlit web application with full feature set
- **Components**: 
  - Complete `FullApp/` directory with web interface
  - All preprocessing modules in `preprocessing/`
  - Tracking algorithms in `tracking/`
  - Utility scripts in `utilities/`
  - Model artifacts in `YOLOv11m/`
  - Configuration files (`config.json`, `requirements.txt`)

#### 2. **jetson_app Service** (Edge Deployment Container)
```yaml
jetson_app:
  build:
    context: "./Jetson AGX Orin"
    dockerfile: Dockerfile
  runtime: nvidia
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
```

**Container Contents:**
- **Inclusion**: Only files within the `Jetson AGX Orin/` folder
- **Purpose**: Self-contained edge processing optimized for NVIDIA Jetson AGX Orin
- **Components**:
  - `Final_function.py`: Complete edge processing pipeline
  - `best1.pt`: Optimized model weights for Jetson hardware
  - `inference_with_ONNX.py`: GPU-accelerated ONNX inference
  - `ourmodel_inference_function.py`: Edge-optimized inference functions
  - `onnxtrasnform.py`: Model format conversion utilities
  - `Test_image.png`: Test image for validation

### Container Isolation Benefits

This architecture provides several key advantages:

1. **Resource Optimization**: The jetson_app container contains only essential files for edge deployment, minimizing memory footprint and transfer requirements

2. **Hardware Specialization**: Each container is optimized for its target hardware:
   - `web_app`: CPU-focused with full Python ecosystem
   - `jetson_app`: GPU-accelerated with NVIDIA runtime support

3. **Deployment Flexibility**: Containers can be deployed independently:
   - Web application for cloud/server deployment
   - Edge container for on-device processing

4. **Development Isolation**: Changes to web application don't affect edge deployment and vice versa

5. **Security Separation**: Edge container has minimal attack surface with only essential processing components

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

### Docker Setup

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

2. **Create Configuration File**:
   **CRITICAL**: The application will not work without this step. Create `config.json` in the root directory using the template provided in step 4 below.

3. **Build and Start the Application**:
   ```bash
   # Build and start the container in detached mode
   docker compose up -d --build
   ```
   
   This command will:
   - Build the Docker image with all dependencies
   - Start the Streamlit web application
   - Make it available at `http://localhost:8501`

4. **Configuration Setup**: 
   * IMPORTANT WILL NOT WORK WITHOUT CONFIG
   * Copy the example config below to `config.json`:
     ```json
     {
         "google-earth": {
             "service_account": {
                 "type": "service_account",
                 "project_id": "PUT_PROJECT_ID_HERE",
                 "private_key_id": "PUT_PRIVATE_KEY_ID_HERE",
                 "private_key": "PUT_PRIVATE_KEY_HERE",
                 "client_email": "PUT_SERVICE_ACCOUNT_EMAIL_HERE",
                 "client_id": "PUT_CLIENT_ID_HERE",
                 "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                 "token_uri": "https://oauth2.googleapis.com/token",
                 "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                 "client_x509_cert_url": "PUT_CLIENT_CERT_URL_HERE"
             }
         },
         "ais_data": {
             "base_url": "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/",
             "url_pattern": "AIS_2024_{date}.zip",
             "year": 2024
         }
     }
     ```

5. **Access the Application**:
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

**Managing Docker Volumes**:
```bash
# List all Docker volumes (useful for troubleshooting storage issues)
docker volume ls

# Remove unused volumes (cleanup)
docker volume prune
```

#### Docker Troubleshooting

**Port Already in Use**:
```bash
# If port 8501 is busy, modify docker compose.yml:
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

#### Local Environment Setup
To ensure a clean, reproducible, and isolated environment for running the application locally:

1. **Create and Activate Virtual Environment**  
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting

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

#### Import Errors in VSCode
* **Issue**: VSCode shows "Import cannot be resolved (Pylance)" even after pip install
* **Solution**: 
  1. Ensure you're using the correct Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
  2. Select the interpreter from your virtual environment (should show `venv` in the path)
  3. Restart VSCode after selecting the correct interpreter
  4. Ensure `__init__.py` files exist in module directories (now included)

#### Google Earth Engine Issues
* **Issue**: "Service account key file not found" or authentication errors
* **Common Causes & Solutions**:
  1. **Missing Config**: Ensure `config.json` exists with proper Google Earth Engine service account credentials
  2. **Invalid Credentials**: Verify all service account fields are correctly filled in config.json
  3. **JSON Format**: Check that the service account JSON structure is valid
  4. **Permissions**: Ensure the service account has Earth Engine access enabled
* **Testing**: The system creates temporary credential files from config.json for authentication

#### Other Issues
* **Streamlit Issues**: Try `pip install --upgrade streamlit`
* **OpenCV Issues**: Try `pip install opencv-python-headless` instead of `opencv-python`
* **Module Not Found**: Ensure you're running commands from the correct directory
* **Virtual Environment**: Always activate your virtual environment before running commands

### Development Setup

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
docker compose up -d
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

The BlueGuard system architecture follows a dual-container design supporting both web-based and edge deployment scenarios:

```
                        +------------------------+
                        |  User's Web Browser    |
                        |  (Accesses via URL)    |
                        +-----------+------------+
                                   |
                                   v
        +----------------------------------------------------------+
        |               Docker Container: web_app                  |
        |                                                          |
        |  +----------------------+                                |
        |  | Streamlit Frontend   |  <-- Handles user requests     |
        |  +----------+-----------+                                |
        |             |                                            |
        |  +----------v-----------+                                |
        |  | Backend Logic        |  <-- Downloads AIS/SAR,        |
        |  | (Python)             |      Outside APIs arrive       |
        |  |                      |      here, queries data        |
        |  +----------+-----------+                                |
        |             |                                            |
        |  +----------v-----------+                                |
        |  |      (local DB)      |  <-- Stores AIS temp data      |
        |  +----------------------+                                |
        |                                                          |
        +----------------------------------------------------------+

                                 =====
                                   | Manual
                                   | Data/Image Transfer
                                   v

        +----------------------------------------------------------+
        |           Docker Container: jetson_app                   |
        |           (NVIDIA Jetson AGX Orin Optimized)             |
        |                                                          |
        |  +----------------------+                                |
        |  | Edge Processing      |  <-- GPU-accelerated inference |
        |  | Pipeline (ONNX)      |      Minimal resource usage    |
        |  +----------+-----------+                                |
        |             |                                            |
        |  +----------v-----------+                                |
        |  | Preprocessing        |  <-- GPU-accelerated where     |
        |  | (GPU-optimized)      |      possible for performance  |
        |  +----------+-----------+                                |
        |             |                                            |
        |  +----------v-----------+                                |
        |  | Ship Detection       |  <-- YOLO inference on GPU     |
        |  | (YOLO + GPU)         |      Optimized model weights   |
        |  +----------------------+                                |
        |                                                          |
        +----------------------------------------------------------+
```

### Architecture Benefits
* **Dual-Container Design**: Separate containers for web application and edge deployment with optimized resource allocation
* **Edge Computing**: Jetson AGX Orin container optimized for GPU-accelerated inference with minimal resource requirements
* **Hardware Specialization**: Each container optimized for its target hardware (CPU vs GPU acceleration)
* **Local Processing**: Eliminates external API dependencies for core functionality
* **Scalable Storage**: Efficient handling of large-scale AIS datasets with automatic cleanup
* **Deployment Flexibility**: Independent deployment of web and edge components

### Jetson AGX Orin Edge Processing

The `jetson_app` container is specifically optimized for the NVIDIA Jetson AGX Orin platform:

**GPU Acceleration Features:**
* **Model Inference**: YOLO detection runs entirely on GPU using ONNX runtime
* **Preprocessing Optimization**: GPU-accelerated preprocessing operations where possible for enhanced performance
* **Memory Efficiency**: Optimized model weights (`best1.pt`) specifically tuned for Jetson hardware constraints
* **Real-time Processing**: Designed for real-time SAR image processing with minimal latency

**Container Specifications:**
* **NVIDIA Runtime**: Full GPU access with `runtime: nvidia`
* **Resource Allocation**: Dedicated GPU device capabilities for compute operations
* **Self-contained**: Complete processing pipeline independent of web application
* **Minimal Footprint**: Only essential files included for optimal memory usage 