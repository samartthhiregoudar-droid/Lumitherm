# Lumitherm: Optical-Guided Thermal Super-Resolution System

This repository contains an optical-guided thermal super-resolution workflow for improving low-resolution thermal satellite imagery and a Streamlit-based demo for interactive inference.

## Overview

The project combines:
- a PyTorch model for thermal super-resolution,
- preprocessing utilities for Landsat-style imagery,
- a Streamlit interface for local exploration, and
- a FastAPI backend for serving inference requests.

The workflow is designed around the files and folders in this repository, including the preprocessing pipeline in the Pre processing directory, the model code, and the demo assets in the frontend folder.

## Project structure

- streamlit_app.py: interactive Streamlit UI entrypoint
- backend_app.py: FastAPI backend for local API deployment
- inference_v2.py: batch inference script for generating example outputs
- Pre processing/: preprocessing, dataloader, model, and training utilities
- frontend/: static assets for the web UI
- dataset_demo/ and dataset/: demo or full datasets used by the scripts
- checkpoints/: pretrained model weights

## Requirements

The project targets Python 3.11 (as defined in runtime.txt) and relies on:
- PyTorch for the model
- NumPy and SciPy for numerical processing
- Matplotlib and Pillow for visualization
- Streamlit for the UI
- FastAPI and Uvicorn for the backend API
- Rasterio for geospatial file handling

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell use: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the app

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

### FastAPI backend

```bash
uvicorn backend_app:app --host 0.0.0.0 --port 8000
```

### Batch inference example

```bash
python inference_v2.py
```

## Notes

- Use streamlit_app.py as the Streamlit entrypoint.
- backend_app.py starts a FastAPI server and should not be used as the Streamlit main file.
- If the full dataset is not present locally, the app will fall back to the demo dataset in dataset_demo/.
- The pretrained checkpoint is expected at checkpoints/best_model.pth.

## Typical workflow

1. Install dependencies with pip install -r requirements.txt.
2. Start the Streamlit app for interactive exploration.
3. Optionally start the FastAPI backend for API-based inference.
4. Run inference_v2.py to generate sample outputs from the available dataset.

