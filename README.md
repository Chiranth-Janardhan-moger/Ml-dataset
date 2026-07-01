# The Silent Cry Decoder 

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.8.0-red.svg)](https://pytorch.org/)
[![FastAPI Version](https://img.shields.io/badge/fastapi-1.0.0-green.svg)](https://fastapi.tiangolo.com/)
[![License: ODbL](https://img.shields.io/badge/License-ODbL-lightgrey.svg)](http://opendatacommons.org/licenses/odbl/1.0/)

An artificial intelligence system that classifies infant cry reasons from audio signatures. It identifies 5 primary drivers: **hunger, belly pain, fatigue (sleepy), discomfort, and the need to burp**.

To achieve high generalization and feature robustness, this project implements a novel **3-stream feature fusion architecture (CNN + BiLSTM + Whisper Feature Fusion)** that captures visual spectrogram mappings, sequential handcrafted features, and semantic speech transformer representations.

---

##  Key Features

*   **3-Stream Ensemble Neural Network**: Hybrid model combining visual, sequential, and pretrained transformers.
*   **Aesthetic PWA Web App**: Clean, responsive dark-themed dashboard featuring real-time audio recording via the Web Audio API and offline installation support.
*   **FastAPI Backend**: Microservice endpoint with async model loading, single-instance singleton pipeline, and GPU-accelerated inference.
*   **Digital Signal Processing Pipeline**: Dynamic resampling (16kHz), log-Mel spectrogram generation, and handcrafted feature calculation (MFCCs, Deltas, ZCR, RMS, Centroid).
*   **Stratified Training Workflow**: Pipeline designed to handle imbalanced datasets using stratified train/val splits and advanced optimization configurations.

---

## ️ Repository Directory Structure

Below is the file tree of the project. Click any file link to view the code directly:

*   [requirements.txt](file:///D:/Antigravity/Ml-project-cry/requirements.txt): List of python dependencies.
*   **`src/`** (Source Code)
    *   [config.py](file:///D:/Antigravity/Ml-project-cry/src/config.py): Center point for hyperparameters, audio configurations, and directory maps.
    *   [features.py](file:///D:/Antigravity/Ml-project-cry/src/features.py): Signal processing utilities for audio resampling, Mel-spectrogram resizing, and handcrafted feature extractions.
    *   [dataset.py](file:///D:/Antigravity/Ml-project-cry/src/dataset.py): PyTorch dataset mapping files and performing stratified validation splits.
    *   [train.py](file:///D:/Antigravity/Ml-project-cry/src/train.py): Optimizer setup, epoch metrics tracking, and checkpoint saving loops.
    *   [evaluate.py](file:///D:/Antigravity/Ml-project-cry/src/evaluate.py): Generates validation scores, classification reports, and confusion matrix plots.
    *   [inference.py](file:///D:/Antigravity/Ml-project-cry/src/inference.py): Singleton pipeline mapping raw files to predictions and pediatrician guidelines.
    *   [app.py](file:///D:/Antigravity/Ml-project-cry/src/app.py): REST API server built with FastAPI.
    *   **`models/`** (Neural Network Modules)
        *   [visual_stream.py](file:///D:/Antigravity/Ml-project-cry/src/models/visual_stream.py): Pretrained EfficientNet-B0 visual feature extractor (Stream 1).
        *   [sequential_stream.py](file:///D:/Antigravity/Ml-project-cry/src/models/sequential_stream.py): Handcrafted BiLSTM sequence model with temporal self-attention (Stream 2).
        *   [pretrained_stream.py](file:///D:/Antigravity/Ml-project-cry/src/models/pretrained_stream.py): OpenAI Whisper-small SSL semantic extractor (Stream 3).
        *   [fusion_model.py](file:///D:/Antigravity/Ml-project-cry/src/models/fusion_model.py): Fuses all three embedding networks into the dense feed-forward classifier.
*   **`frontend/`** (PWA Web Client)
    *   [index.html](file:///D:/Antigravity/Ml-project-cry/frontend/index.html): Semantic dashboard wrapper, structured for SEO compliance.
    *   [style.css](file:///D:/Antigravity/Ml-project-cry/frontend/style.css): Modern dark-theme glassmorphism styling, animations, and layouts.
    *   [app.js](file:///D:/Antigravity/Ml-project-cry/frontend/app.js): Handles recording inputs, timer ticks, API payloads, and UI loading transitions.
    *   [manifest.json](file:///D:/Antigravity/Ml-project-cry/frontend/manifest.json): Configuration file satisfying PWA desktop/mobile installation criteria.
    *   [sw.js](file:///D:/Antigravity/Ml-project-cry/frontend/sw.js): Active service worker enabling asset caching.
*   **`docs/`** (Study Material & Deep-Dives)
    *   [placement_guide.md](file:///D:/Antigravity/Ml-project-cry/docs/placement_guide.md): Study cheat sheet, structure workflows, and detailed mock interview Q&As.
    *   [technical_deep_dive.md](file:///D:/Antigravity/Ml-project-cry/docs/technical_deep_dive.md): Signal processing math, block configurations, fusion analysis, and backend design patterns.

---

##  The 3-Stream Model Architecture

This system decodes audio input from three complementary perspectives before predicting a label:

1.  **Visual Stream (CNN)**: Audio is converted into a 128-band Mel-Spectrogram image $(224 \times 224 \times 3)$ and passed through **EfficientNet-B0**. The final classification head is replaced by a 128-dim linear projection. Early blocks are frozen, and only the final 3 Blocks (6, 7, 8) and the projection head are fine-tuned.
2.  **Sequential Stream (BiLSTM)**: Standard acoustic features do not capture temporal contours. We extract 40 MFCCs along with their 1st and 2nd derivatives (120 features), and combine them with 5 handcrafted frame-wise descriptors (ZCR, RMS energy, spectral centroid, bandwidth, rolloff) to produce a 125-dim vector per frame. These features are processed through a BiLSTM (128 hidden), a Self-Attention layer, and a final BiLSTM (64 hidden), returning a 128-dim temporal embedding.
3.  **Pretrained Stream (Whisper)**: OpenAI's **Whisper-small** encoder serves as a powerful acoustic feature extractor. We freeze its parameters, extract the sequence representations from raw waveforms, and pass them through a global average pooling layer and a linear head to output a 128-dim semantic embedding.

All three embeddings are concatenated into a **384-dimensional joint representation** before being classified.

---

##  Dataset Details

The dataset directory is located at [cry-dataset](file:///D:/Antigravity/Ml-project-cry/cry-dataset). It contains user-uploaded baby cry samples categorized into 5 subfolders:
*   [hungry](file:///D:/Antigravity/Ml-project-cry/cry-dataset/hungry) -> Hunger (`hu`)
*   [belly_pain](file:///D:/Antigravity/Ml-project-cry/cry-dataset/belly_pain) -> Pain (`bp`)
*   [tired](file:///D:/Antigravity/Ml-project-cry/cry-dataset/tired) -> Sleepy (`ti`)
*   [discomfort](file:///D:/Antigravity/Ml-project-cry/cry-dataset/discomfort) -> Discomfort (`dc`)
*   [burping](file:///D:/Antigravity/Ml-project-cry/cry-dataset/burping) -> Burping (`bu`)

Audio files follow the naming conventions of the campaign. For example:
`0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.wav`
Represents: `app-uuid` - `timestamp` - `app-version` - `gender` - `age-bracket` - `reason-tag`.

---

##  Quick Start Instructions

### 1. Setup Environment & Install Dependencies
Ensure you have Python 3.13+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Fusion Network
To run the training loop, prepare the dataset, and output models checkpoints:
```bash
python -m src.train
```

### 3. Evaluate the Model
To compute classification scores and generate the confusion matrix:
```bash
python -m src.evaluate
```
This prints the detailed validation reports and saves `confusion_matrix.png` and `classification_report.txt` inside the `checkpoints/` directory.

### 4. Start the Backend API
Start the FastAPI server on port 8000:
```bash
python -m src.app
```

### 5. Launch the Web Client (PWA)
Run a local web server inside the root directory to access the frontend:
```bash
# Start a simple HTTP server on port 3000
python -m http.server 3000 --directory frontend/
```
Open your browser and navigate to `http://localhost:3000`. You can record cries directly or upload samples. The client will communicate with the API backend at `http://localhost:8000/predict`.

*Note: If the FastAPI server is not running, the web application runs in simulated prediction mode so you can demonstrate the UI flow during interviews.*
