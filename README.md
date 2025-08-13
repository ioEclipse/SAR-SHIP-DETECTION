# SAR Ship Detection Project

## Overview

The SAR Ship Detection Project is an end-to-end system designed for the automated detection and tracking of maritime vessels using Synthetic Aperture Radar (SAR) imagery. This project leverages advanced deep learning models to process SAR data, primarily from Sentinel-1 satellites, providing crucial insights for maritime domain awareness, safety, and security. A key focus of this project is the detection of **"dark vessels," by integrating AIS (Automatic Identification System) data with SAR observations to identify vessels that are not broadcasting their location.** Optimized for efficient inference on edge devices like the NVIDIA Jetson Nano, this solution aims to bridge the gap between powerful machine learning and deployable satellite-based applications.

This project is still a Work In Progress, so you may see a few placeholders.

-----

## 🚀 Features

  * **Automated Ship Detection:** Utilizes state-of-the-art deep learning models (e.g., YOLOv8) for accurate ship identification in complex SAR environments.
  * **Multi-Object Tracking:** Implements robust tracking algorithms (e.g., DeepSort) to maintain persistent ship identities across sequential SAR images.
  * **Dark Vessel Detection:** Integrates AIS data with SAR detections to identify vessels not transmitting their position, crucial for illegal activity monitoring.
  * **SAR Data Preprocessing Pipeline:** Includes essential steps like speckle noise reduction, land-sea segmentation, and radiometric correction.
  * **Edge Device Optimization:** Models are optimized for high-performance, low-latency inference on resource-constrained hardware (e.g., NVIDIA Jetson Nano).
  * **Versatile Interfaces:** Designed for integration into various operational modalities, including edge deployment, web-based map interfaces, and direct image upload for processing.
  * **Applications:** Supports maritime surveillance, illegal activity monitoring (e.g., illegal fishing, smuggling), search and rescue operations, and general maritime intelligence.

-----

## 💻 Getting Started

The project supports two setup methods: **Docker (Recommended)** for production-ready deployment and **Local Installation** for development.

### 🐳 Quick Start with Docker (Recommended)

Docker provides a consistent, isolated environment that eliminates dependency conflicts.

#### Prerequisites
* Docker Engine (20.10.0+) and Docker Compose (2.0.0+)
* Git

#### Installation & Launch
```bash
# Clone the repository
git clone https://github.com/ioEclipse/SAR-SHIP-DETECTION.git
cd SAR-SHIP-DETECTION

# Build and start the application
docker compose up -d --build

# Access the application at http://localhost:8501
```

#### Docker Development Commands
```bash
# View logs
docker compose logs -f

# Access container shell for debugging
docker compose exec web bash

# Stop gracefully
docker compose down

# Rebuild after code changes to requirements
docker compose up -d --build
```

#### Docker Troubleshooting
```bash
# Check container status
docker compose ps

# View detailed logs
docker compose logs web

# Complete rebuild
docker compose down --rmi local
docker compose up -d --build
```

### 🔧 Local Installation (Development)

For developers who need direct Python environment access.

#### Prerequisites
* Python 3.8+
* Git
* NVIDIA GPU (for training and optimized inference)
* NVIDIA Jetson Nano (for edge deployment testing)
* Conda or `venv` for virtual environment management

#### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ioEclipse/SAR-SHIP-DETECTION.git
    cd SAR-SHIP-DETECTION
    ```

2.  **Set up a Python virtual environment:**

    ```bash
    python3 -m venv venv_sardetection
    source venv_sardetection/bin/activate
    ```

    *(Alternatively, using Conda: `conda create -n sardetection python=3.8` then `conda activate sardetection`)*

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Ensure you have the correct NVIDIA drivers and CUDA toolkit installed for GPU acceleration, as specified in the documentation.*

4. **Create config.json**
   * IMPORTANT WILL NOT WORK WITHOUT CONFIG
   * Copy the example config below
   * Add your AIS stream API key to `config.json`:
     ```json
     {
      "google-earth": {
        "api_key": "PUT_API_KEY_HERE"
      },
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

5.  **Run the application:**
    ```bash
    cd FullApp
    streamlit run home.py
    ```
    Access at `http://localhost:8501`

6.  **Download pre-trained models:**
    *Detailed instructions for downloading and placing pre-trained model weights will be provided here.*
    YOLOv11m folder contains the weights.

## 📦 Application

### Overview

The `FullApp` directory encapsulates a comprehensive Synthetic Aperture Radar (SAR) ship detection application, integrating all functionalities developed throughout this project. This application provides an end-to-end pipeline for processing, tracking, and analyzing ships using radar imagery. It enables users to upload raw SAR data, explore Google Earth Engine SAR collections, and leverage cloud-based AI for robust inference. The system features an intuitive web-based dashboard for visualizing detailed reports, including global annotated images, sub-images of detected ships, and ship metadata (e.g., pixel area and surface area in square meters). Additionally, it supports export of results and is optimized for edge deployment on platforms like NVIDIA Jetson Nano, facilitating real-time inference without requiring prior land-ocean segmentation. This dual-targeted architecture ensures both operational efficiency in edge environments and user-friendly interaction for maritime surveillance.

### 🛠️ Environment Setup

Choose your preferred setup method:

#### Docker Setup (Recommended)
```bash
# One-time setup - builds and starts the application
docker-compose up -d --build

# Application available at http://localhost:8501
# No additional environment configuration needed
```

#### Local Environment Setup
To ensure a clean, reproducible, and isolated environment for running the application locally:

1. **Create a Virtual Environment**  
   Create a Python virtual environment (e.g., named `.venv`) to isolate dependencies and avoid conflicts with system-wide packages:

   ```bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment**  
   Activate the virtual environment to use its isolated Python interpreter and packages:  
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies**  
   Install the required Python packages listed in `requirements.txt` to ensure compatibility and reproducibility:

   ```bash
   pip install -r requirements.txt
   ```

   This command installs dependencies such as `streamlit`, `pandas`, and other libraries essential for the application's functionality.

### 🚀 Launching the Application

#### Docker Launch (Recommended)
```bash
# Start the application (builds automatically if needed)
docker-compose up -d

# View logs (optional)
docker-compose logs -f

# Access at http://localhost:8501
```

#### Local Launch
Navigate to the `FullApp` directory and launch the Streamlit application by running the main entry point:

```bash
# Ensure virtual environment is activated first
cd FullApp
streamlit run home.py
```

This command starts the SAR Ship Detection Dashboard, which opens in your default web browser (e.g., at `http://localhost:8501`).

### 📂 Directory Structure and File Descriptions

The `FullApp` directory is structured as follows, with each file and directory serving a specific role:

```
FullApp/
├── assets/                    # Static images, stylesheets, icons, and visual assets
├── pages/
│   ├── main.py                # The main entry point of the Streamlit dashboard
│   ├── earthengine.py         # Interface with Google Earth Engine SAR collections
│   ├── infer2.py              # Local inference with slicing, bounding boxes, and subimages
│   └── app.py                 # Core logic and UI for image prediction and download
├── home.py                    # Optional: Landing page or welcome module
├── Testimage.png              # A sample radar image for demonstration or testing
├── requirements.txt           # Python dependency file for reproducibility
```

- **assets/**: Contains static files such as images (e.g., `logo.png`, `ship.png`), stylesheets, and icons used to enhance the application’s user interface.
- **pages/main.py**: Serves as the primary Streamlit script, orchestrating the dashboard’s navigation and user interactions.
- **pages/earthengine.py**: Manages integration with Google Earth Engine, enabling access to and processing of SAR data collections.
- **pages/infer2.py**: Implements the inference pipeline, handling local SAR image processing, including slicing, bounding box detection, and sub-image generation.
- **pages/app.py**: Contains the core application logic and UI components for image prediction, visualization, and result downloading.
- **home.py**: Provides an optional landing page or welcome module to enhance user onboarding.
- **Testimage.png**: A sample radar image for testing and demonstration purposes.
- **requirements.txt**: Specifies all Python packages required to run the application, ensuring consistent setup across environments.

### ✅ Features Included

The application offers a comprehensive set of functionalities for SAR ship detection and analysis:

- **Raw SAR Image Processing**: Supports upload and processing of raw SAR images in `.png` or `.jpg` formats for ship detection.
- **Interactive Dashboard**: Provides a user-friendly Streamlit interface for visualizing global annotated images, individual ship sub-images, and metadata tables.
- **Automatic Sub-Image Cropping**: Extracts sub-images around detected ships with configurable margins, facilitating detailed inspection.
- **Ship Characteristics Analysis**: Computes and displays metadata, including ship ID, pixel area, and surface area in square meters, presented in an interactive table with a "See More" option for extended results.
- **Google Earth Engine Integration**: Enables access to SAR data collections for advanced data exploration and analysis.
- **Export Capabilities**: Allows downloading of annotated images, sub-images, and ship metadata for further analysis or reporting.
- **Persistent Session State**: Maintains user interactions across sessions, ensuring a seamless experience.
- **Edge Deployment Support**: Optimized for deployment on NVIDIA Jetson Nano, enabling real-time inference on raw SAR imagery without requiring land-ocean segmentation.

## 📊 Dataset

The primary dataset used for training and evaluation can be downloaded from:
[**Google Drive Dataset Download**](https://drive.google.com/file/d/1mJmn4Ad-oJ66HVHNrRK-BpDlxzXvF-0H/view?usp=sharing)

*This link will be updated with the actual dataset download location once available. Currently contains the SSDD Dataset.*

-----

## 💡 Usage

*Detailed instructions on how to run the detection and tracking pipeline, including command-line arguments and configuration options, will be provided in the full documentation.*

**Example (placeholder):**

```bash
python run_pipeline.py --input_dir ./data/raw_sar_images --output_dir ./results --model_path ./models/yolov8_sardetect.pt
```

-----

## 📖 Documentation & Project Roadmap

  * **Project Documentation:**
      * [**PDF Document**](https://drive.google.com/file/d/182cVIUZ71hkKXD2s3pYEsNLtcBHFN5_A/view?usp=sharing) - A detailed guide covering the project's architecture, methodology, and technical specifications.
  * **Project Task List:**
      * [**Google Sheet**](https://docs.google.com/spreadsheets/d/1uLw39G2AHuvqWG3Va8wvNI3f9989mslL4P4FfVq1el4/edit?usp=sharing) - Track the project's progress, upcoming tasks, and assigned responsibilities.

-----

## 🤝 Contributing

We welcome contributions to the SAR Ship Detection Project\! Please see our [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md) for guidelines on how to submit pull requests, report issues, and contribute to the development.

-----

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.


