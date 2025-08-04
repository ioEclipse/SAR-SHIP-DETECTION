## Project Overview

The SAR Ship Detection Project aims to detect and track ships using Synthetic Aperture Radar (SAR) data. It involves machine learning models for land-sea segmentation, ship detection (YOLOv11m), and tracking (DeepSort). The system offers three user interfaces: a Jetson Nano interface for on-orbit inference, a web-based interface for on-demand analysis via a map, and an image upload feature for user-provided raw SAR imagery.

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

### 5. Output Generation and Visualization
* **Purpose**: Compile and present results to the user through various interfaces.
* **Output Formats**:
    * Structured JSON data (bounding box coordinates, width, class label, unique ID).
    * Annotated images (full image with detected ships, individual ship images by ID).
    * Statistical insights (total ships detected, estimated sizes).
* **User Interfaces (Streamlit-based dashboard)**:
    * **Jetson Nano Interface**: Pre-loaded images, local display/streaming, statistical insights (no specific user requests).
    * **Web-based Map Interface**: User selects ROI on a map (Google Earth style), system fetches SAR data, processes, and displays results.
    * **Image Upload Interface**: User uploads raw SAR images for processing and display.

### 6. System Management and User Experience
* **Queue Management**: Implement a FIFO queue for incoming processing requests to manage server load. Provide user feedback on queue position and estimated wait time.
* **Error Handling**: Gracefully handle unexpected inputs and provide informative error messages.
* **Data Integrity**: Ensure integrity of processed data and output files.
* **Concurrency**: Handle a limited number of concurrent users.

## main.py Skeleton Requirements:

Based on the above, the `main.py` should define:

* **Main Application Class/Functions**: A central entry point (e.g., `main()` function) that orchestrates the overall workflow.
* **Configuration Handling**: Mechanisms to load and manage system parameters and model configurations.
* **Data Handling**: Functions/classes for `data acquisition`, `temporary storage`, and `cleanup`.
* **Preprocessing Module**: A class or set of functions (e.g., `preprocess_sar_image`) encapsulating the entire preprocessing pipeline.
* **Detection Module**: A class or function (e.g., `detect_ships`) for running the YOLOv11m inference.
* **Tracking Module**: A class or function (e.g., `track_ships`) for multi-object tracking.
* **Output Module**: Functions (e.g., `generate_output_json`, `generate_annotated_image`) for creating various output formats.
* **User Interface Integration**: High-level functions or a class (e.g., `run_streamlit_app`) to manage the Streamlit web application and its different modes.
* **Queue Management**: A class or functions (e.g., `ProcessingQueue`) to handle user requests and queueing.
* **Error Handling and Logging**: Basic structure for error capturing and logging.

The skeleton should use placeholders for internal logic (e.g., `pass` or `...`) and focus solely on the modular structure and function/method signatures.