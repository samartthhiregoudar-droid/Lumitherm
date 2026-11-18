# Lumitherm: Sharpening Satellite Heat Maps with Light

Lumitherm uses high-resolution optical imagery to guide super-resolution of low-resolution thermal satellite bands, turning blurry heat maps into sharper ones.

![Low-resolution thermal input alongside the super-resolved output](docs/demo.gif)
<!-- Replace with a screenshot or GIF of the Streamlit UI showing input vs. output -->

## Motivation

Thermal infrared bands on Landsat-style satellites capture surface temperature, which is essential for tracking urban heat islands, drought stress, and irrigation. But those bands are captured at roughly 100 m per pixel, while the optical bands on the same satellite are at 30 m. Lumitherm uses the sharper optical imagery as a guide, so the thermal map can be reconstructed at a much finer resolution. [Add your own reason for building this here: what problem you ran into, and why you wanted to solve it.]

## Features

- **Optical-guided thermal super-resolution** with a PyTorch model
- **Landsat preprocessing pipeline** for aligning and preparing bands
- **Interactive Streamlit UI** for uploading scenes and comparing results
- **FastAPI backend** for serving inference over HTTP
- **Batch inference script** for generating sample outputs

## Quick Start

**Requirements:** Python 3.11 and the pretrained checkpoint at `checkpoints/best_model.pth`.

1. Clone the repository and set up a virtual environment:

   ```bash
   git clone https://github.com/<your-username>/lumitherm.git
   cd lumitherm
   python -m venv .venv
   source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Launch the app:

   ```bash
   streamlit run streamlit_app.py
   ```

Open the URL Streamlit prints (usually `http://localhost:8501`) and explore the sample scenes. If the full dataset isn't present locally, the app falls back to the sample data in `dataset_demo/`.

## Usage

### Interactive UI

```bash
streamlit run streamlit_app.py
```

### API server

```bash
uvicorn backend_app:app --host 0.0.0.0 --port 8000
```

Once it's running, FastAPI's interactive docs are at `http://localhost:8000/docs`.

### Batch inference

Generate example outputs from the available dataset:

```bash
python inference_v2.py
```

## Project Structure

```
streamlit_app.py     # Streamlit UI entrypoint
backend_app.py       # FastAPI backend (don't use as the Streamlit main file)
inference_v2.py      # Batch inference script
Pre processing/      # Preprocessing, dataloader, model, and training code
frontend/            # Static assets for the web UI
dataset/             # Full dataset (used when present)
dataset_demo/        # Sample dataset (fallback)
checkpoints/         # Pretrained model weights
```

## Tech Stack

PyTorch · NumPy · SciPy · Rasterio · Matplotlib · Pillow · Streamlit · FastAPI · Uvicorn

## Contributing

Contributions are welcome! To get started:

1. Fork the repository and create a feature branch.
2. Make your changes and verify the Streamlit app and inference script still run.
3. Open a pull request describing what you changed and why.

For larger changes, open an issue first so we can discuss the approach.
## Usage

### Streamlit UI

Launch the interactive interface:

```bash
streamlit run streamlit_app.py
```

Upload a Landsat scene, run super-resolution, and compare the low-resolution input with the sharpened output side by side.

### FastAPI Backend

Start the inference server:

```bash
uvicorn backend_app:app --host 0.0.0.0 --port 8000
```

Interactive API docs are available at `http://localhost:8000/docs`. Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/scene.tif" \
  -o output.tif
```

> Adjust the endpoint path and form field name to match `backend_app.py`.

### Batch Inference

Generate example outputs for every scene in the dataset:

```bash
python inference_v2.py
```

Outputs are written to the output directory configured in the script. Set `dataset/` or `dataset_demo/` as the source depending on which data you have locally.

### Dataset Selection

The app reads from `dataset/` when it's present and falls back to `dataset_demo/` otherwise. To force the demo data, remove or rename `dataset/`.

### Model Checkpoint

The pretrained weights are loaded from `checkpoints/best_model.pth`. To use your own trained model, replace that file with a checkpoint that has the same architecture.

## Examples

Run the full pipeline on the sample dataset:

```bash
streamlit run streamlit_app.py
```

Generate sample outputs without the UI:

```bash
python inference_v2.py
```

Request a super-resolved tile from the API:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@dataset_demo/sample_scene.tif" \
  -o sharpened.tif
```

> Replace `sample_scene.tif` with an actual filename from `dataset_demo/`.
> ## Contributing

Contributions are welcome. To set up a local development environment:

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/lumitherm.git
cd lumitherm
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Verify the setup

Launch the Streamlit UI and confirm it loads the demo data:

```bash
streamlit run streamlit_app.py
```

Then run the batch script to confirm inference works end to end:

```bash
python inference_v2.py
```

### 4. Make your changes

- Create a feature branch: `git checkout -b feature/your-change`
- Keep changes focused, and update the README if you change setup steps, commands, or endpoints.
- Test the Streamlit app, the FastAPI backend, and `inference_v2.py` before opening a pull request, since they share the same model code in `Pre processing/`.

### 5. Submit a pull request

Push your branch and open a pull request against `main` with a description of what changed and why. For larger changes, open an issue first so we can discuss the approach.

## License

[Add license here]
