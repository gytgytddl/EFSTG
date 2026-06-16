# EFSTG

This repository contains the official implementation of the EFSTG model, a robust spatial-temporal graph neural network designed for traffic prediction tasks.

## 📂 Project Structure

- `util.py`: Contains core utility functions, including data loaders (PyTorch `DataLoader`), normalization scalers (`StandardScaler`), standard evaluation metrics (MAE, MAPE, RMSE, WMAPE), and robust loss functions (Huber Loss, Masked MAE).
- `dataprocess.py`: Script to process raw traffic data. It extracts specific traffic features (e.g., occupancy), injects temporal features (time-of-day, day-of-week), and generates sequence-to-sequence `(train/val/test)` datasets.
- `graphprocess.py`: Script to construct the spatial graph. It parses road network structures (from CSV) and computes all-pairs shortest paths using Dijkstra's algorithm to generate a spatial distance matrix.
- `train.py` & `model.py` *(already in repo)*: The training logic and the core model architectures.

## 📊 Datasets

This project utilizes two major categories of public spatial-temporal datasets:

### 1. PeMS Datasets (Highway Traffic)

The Caltrans Performance Measurement System (PeMS) datasets are widely used for highway traffic forecasting. We primarily use **PeMS03, PeMS04, PeMS07, and PeMS08**.

- **Source:** Data is collected in real-time from over 39,000 individual detectors spanning the freeway system across all major metropolitan areas of California.
- **Features:** Traffic flow, occupancy, and speed, aggregated into 5-minute intervals (288 time steps per day).
- **Graph:** The spatial graph is constructed based on the actual physical distance (cost) between sensor nodes on the highway network.

### 2. NYC Datasets (Urban Mobility)

New York City datasets capture complex urban mobility patterns.

- **Source:** Usually derived from the NYC Open Data portal, specifically the NYC Taxi and Limousine Commission (TLC) Trip Record Data or NYC Citi Bike datasets.
- **Features:** Pick-up/drop-off inflow and outflow sequences in different urban regions.

*(Note: Raw dataset files are not included in this repository due to size limitations. Please download them from standard benchmarks like [STSGCN](https://github.com/Davidham3/STSGCN) (including PEMS03, PEMS04, PEMS07, and PEMS08) and [ESG](https://github.com/LiuZH-19/ESG) (including NYCBike and NYCTaxi) and place them in the `data/` directory.)*

## 🚀 Getting Started

### 1. Environment Setup

Install the required Python packages using:
```bash
pip install -r requirements.txt
```
### 2. Prepare the Graph Matrix

Before training, generate the distance matrix for the target dataset (e.g., PeMS08). This script calculates the shortest paths between nodes:
```bash
python graphprocess.py
```
Output: `pems08_distance_matrix.npy` will be saved in your processed data folder.

### 3. Generate Sequence Data

Process the raw .npz data to generate the input/output sequences with temporal features (Time of Day, Day of Week):
```bash
python dataprocess.py --traffic_df_filename data/PEMS08/PEMS08.npz --output_dir data/processed/PEMS08/
```
Output: `train.npz`, `val.npz`, `test.npz` containing tensors of shape (B, T, N, 3).

### 4. Training

Run the training script (adjust parameters inside the script or via command line as needed):
```bash
python train.py
```
## 🛠️ Key Features

- **Robust DataLoader:** Highly optimized PyTorch DataLoader specifically designed for time-series forecasting, utilizing pinned memory and custom thread management to prevent deadlocks on Windows/Linux environments.
- **Advanced Loss Functions:** Includes standard Masked MAE and SOTA-recommended Masked Huber Loss to handle extreme anomalies and prevent gradient explosions during early training phases.
- **Multi-Feature Input:** Seamlessly integrates spatial readings with cyclic temporal features (day/week indicators) for enhanced predictive accuracy.
