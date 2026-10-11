# Brain Tumor Classification Deep Learning Pipeline

An advanced computer vision pipeline built using Python and TensorFlow/Keras to analyze and classify MRI brain scan imagery into diagnostic categories.

Project Overview:
This project implements a convolutional neural network (CNN) architecture to process medical imaging data. By structuring high-resolution MRI scans into dedicated training and testing directories, the model learns spatial features to accurately detect and classify brain anomalies.

Dataset Structure:
The dataset used for this project contains brain MRI scan images categorized into diagnostic classes. Due to file size limits, the raw image dataset is not stored directly in this repository.

* Data Source: A standard brain MRI classification dataset (sourced from Kaggle).
* Expected Directory Structure: Ensure your local project workspace is organized with your training and testing folders in the root directory:
  - `BrainTumorProject/`
    - `brain_tumor_dataset_project.py`
    - `Training/`
    - `Testing/`


Technologies Used:
- Python (Core machine learning script logic)
- TensorFlow & Keras (Deep learning framework, model building, and tensor manipulation)
- OpenCV / Pillow (Image preprocessing and array conversion)
- NumPy & Matplotlib (Data matrix operations, loss curves, and evaluation plotting)

Key Features & Architecture:
- Structured Dataset Handling: Organized directory pipeline mapping data into explicit `Training/` and `Testing/` subsets.
- Image Preprocessing: Automated resizing, normalization, and pixel scaling for neural network optimization.
- CNN Feature Extraction: Multi-layer convolutional and pooling operations designed to isolate complex tumor patterns from MRI scans.
- Model Evaluation: Performance metrics tracking validation accuracy and loss across training epochs.

# How to Run the Code:
1. Clone or download the repository.
2. Install the required deep learning dependencies:
   ```bash
   pip install tensorflow numpy matplotlib pillow opencv-python

   python brain_tumor_dataset_project.py
