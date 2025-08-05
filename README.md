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

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

  * Python 3.8+
  * Git
  * NVIDIA GPU (for training and optimized inference)
  * NVIDIA Jetson Nano (for edge deployment testing)
  * Conda or `venv` for virtual environment management

### Installation

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

3.  **Install dependencies: (Currently missing)**

    ```bash
    pip install -r requirements.txt
    ```

    *Ensure you have the correct NVIDIA drivers and CUDA toolkit installed for GPU acceleration, as specified in the documentation.*

4.  **Download pre-trained models:**
    *Detailed instructions for downloading and placing pre-trained model weights will be provided here.*

-----

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
