# HydroGraphNet Pipeline Documentation

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Data Structure](#data-structure)
4. [Data Flow & Loading](#data-flow--loading)
5. [Graph Construction](#graph-construction)
6. [Node & Edge Features](#node--edge-features)
7. [Ground Truth Labels & Loss Functions](#ground-truth-labels--loss-functions)
8. [Physics Laws Applied](#physics-laws-applied)
9. [Model Architecture](#model-architecture)
10. [Training Pipeline](#training-pipeline)
11. [Inference & Evaluation](#inference--evaluation)
12. [Data Synthesis](#data-synthesis)
13. [Requirements](#requirements)
14. [Configuration Reference](#configuration-reference)

---

## Overview

HydroGraphNet is a **Physics-Informed Graph Neural Network (PINN-GNN)** designed for large-scale flood dynamics modeling. It combines:

- **Graph Neural Networks** for spatial message passing on unstructured meshes
- **Kolmogorov–Arnold Networks (KANs)** for enhanced interpretability
- **Physics-informed constraints** for mass conservation
- **Autoregressive forecasting** for multi-step predictions

The system predicts **water depth** and **volume changes** across a spatial domain during flooding events.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HydroGraphNet Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ Data Loading │ -> │   Graph      │ -> │   Model      │ -> │ Loss       │ │
│  │ & Processing │    │ Construction │    │ (MeshGraphKAN)   │ Computation │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                  │         │
│         v                   v                   v                  v         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ Static Data  │    │ k-NN Graph   │    │ Encoder      │    │ MSE Loss   │ │
│  │ + Dynamic    │    │ Connectivity │    │ Processor    │    │ + Physics  │ │
│  │   Data       │    │ + Features   │    │ Decoder      │    │   Loss     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### High-Level Workflow

1. **Data Preparation**: Load static terrain features and dynamic hydrograph time series
2. **Normalization**: Compute and apply z-score normalization to all features
3. **Graph Construction**: Build k-NN spatial graph from node coordinates
4. **Sliding Window Sampling**: Create training samples with temporal context
5. **Forward Pass**: Encode → Process (message passing) → Decode
6. **Loss Computation**: MSE on predictions + Physics-based continuity loss
7. **Autoregressive Rollout**: For inference, iteratively predict future states

---

## Data Structure

### Directory Structure

```
data/
├── M80_XY.txt           # Node coordinates (X, Y)
├── M80_CA.txt           # Cell area
├── M80_CE.txt           # Cell elevation
├── M80_CS.txt           # Cell slope
├── M80_A.txt            # Aspect
├── M80_CU.txt           # Curvature
├── M80_N.txt            # Manning's roughness coefficient
├── M80_FA.txt           # Flow accumulation
├── M80_IP.txt           # Infiltration parameter
├── M80_WD_{id}.txt      # Water depth time series (per hydrograph)
├── M80_V_{id}.txt       # Volume time series (per hydrograph)
├── M80_US_InF_{id}.txt  # Upstream inflow hydrograph
├── M80_Pr_{id}.txt      # Precipitation time series
├── train.txt            # List of training hydrograph IDs
├── test.txt             # List of test hydrograph IDs
├── static_norm_stats.json   # Cached static normalization stats
└── dynamic_norm_stats.json  # Cached dynamic normalization stats
```

### File Formats

| File | Format | Dimensions | Description |
|------|--------|------------|-------------|
| `M80_XY.txt` | Tab-delimited | `[num_nodes, 2]` | X, Y coordinates |
| `M80_WD_{id}.txt` | Tab-delimited | `[time_steps, num_nodes]` | Water depth at each node over time |
| `M80_V_{id}.txt` | Tab-delimited | `[time_steps, num_nodes]` | Volume at each node over time |
| `M80_US_InF_{id}.txt` | Tab-delimited | `[time_steps, 2]` | Time and inflow values |
| `M80_Pr_{id}.txt` | Tab-delimited | `[time_steps]` | Precipitation values |

### Automatic Data Download

If data is not present, it is automatically downloaded from Zenodo:
- **Record ID**: `14969507`
- The dataset is extracted and verified using MD5 checksums

---

## Data Flow & Loading

### HydroGraphDataset Class

The `HydroGraphDataset` (located in `physicsnemo/datapipes/gnn/hydrographnet_dataset.py`) handles all data operations:

```python
dataset = HydroGraphDataset(
    data_dir="./data",
    prefix="M80",
    split="train",           # "train" or "test"
    num_samples=500,         # Max number of hydrographs to load
    n_time_steps=2,          # Sliding window size
    k=4,                     # k-nearest neighbors
    noise_type="none",       # Noise injection type
    noise_std=0.01,          # Noise standard deviation
    return_physics=True,     # Return physics data for loss
    rollout_length=30,       # For test mode
)
```

### Processing Flow

1. **Ensure Data Available**: Download from Zenodo if not present
2. **Load Static Data**: Load and normalize terrain features
3. **Build Graph**: Construct k-NN graph from coordinates
4. **Load Dynamic Data**: For each hydrograph ID, load time series
5. **Normalize Dynamic Data**: Compute global statistics and normalize
6. **Build Sample Index**: Create sliding window indices for training

### Normalization

All features are z-score normalized:

$$x_{norm} = \frac{x - \mu}{\sigma + \epsilon}$$

where $\epsilon = 10^{-8}$ prevents division by zero.

Statistics are saved to JSON files for consistent normalization during testing.

---

## Graph Construction

### k-Nearest Neighbors Graph

The spatial graph is constructed using k-NN:

```python
from scipy.spatial import KDTree

kdtree = KDTree(xy_coords)
_, neighbors = kdtree.query(xy_coords, k=k+1)  # k+1 to exclude self

edge_index = np.vstack([
    (i, nbr) for i, nbrs in enumerate(neighbors) 
    for nbr in nbrs if nbr != i
]).T  # Shape: [2, num_edges]
```

- **Default k**: 4 (each node connects to 4 nearest neighbors)
- **Result**: Directed edges (both directions created)
- **Graph representation**: PyTorch Geometric `Data` object

---

## Node & Edge Features

### Node Features (16 dimensions by default)

| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 0-1 | `xy_coords` | Static | Normalized X, Y coordinates |
| 2 | `area` | Static | Cell area |
| 3 | `elevation` | Static | Cell elevation |
| 4 | `slope` | Static | Terrain slope |
| 5 | `aspect` | Static | Terrain aspect (direction) |
| 6 | `curvature` | Static | Terrain curvature |
| 7 | `manning` | Static | Manning's roughness coefficient |
| 8 | `flow_accum` | Static | Flow accumulation |
| 9 | `infiltration` | Static | Infiltration parameter |
| 10 | `inflow` | Dynamic | Current inflow hydrograph value |
| 11 | `precipitation` | Dynamic | Current precipitation value |
| 12-13 | `water_depth` | Dynamic | Water depth history (n_time_steps) |
| 14-15 | `volume` | Dynamic | Volume history (n_time_steps) |

### Edge Features (3 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0-1 | Relative coordinates | Normalized (x_i - x_j, y_i - y_j) |
| 2 | Distance | Normalized Euclidean distance between nodes |

### Feature Construction Code

```python
features = np.hstack([
    xy_coords,                    # [N, 2]
    area,                         # [N, 1]
    elevation,                    # [N, 1]
    slope,                        # [N, 1]
    aspect,                       # [N, 1]
    curvature,                    # [N, 1]
    manning,                      # [N, 1]
    flow_accum,                   # [N, 1]
    infiltration,                 # [N, 1]
    inflow_broadcast,             # [N, 1]
    precip_broadcast,             # [N, 1]
    water_depth_window.T,         # [N, n_time_steps]
    volume_window.T,              # [N, n_time_steps]
])  # Final shape: [N, 12 + 2*n_time_steps]
```

---

## Ground Truth Labels & Loss Functions

### Target Variables

The model predicts **differences** (residuals), not absolute values:

```python
# Target computation
target_depth = water_depth[t+n] - water_depth[t+n-1]   # Δ depth
target_volume = volume[t+n] - volume[t+n-1]            # Δ volume
target = np.stack([target_depth, target_volume], axis=1)  # [N, 2]
```

### Loss Functions

#### 1. MSE Loss (Data-Driven)

Standard Mean Squared Error between predictions and ground truth:

```python
mse_loss = nn.MSELoss()(pred, graph.y)
```

#### 2. Physics Loss (Continuity/Mass Conservation)

Ensures predictions satisfy mass balance constraints:

```python
def compute_physics_loss(pred, physics_data, graph, delta_t=1200.0):
    # Denormalize volumes
    past_volume_denorm = past_volume_norm * volume_std + num_nodes * volume_mean
    future_volume_denorm = future_volume_norm * volume_std + num_nodes * volume_mean
    
    # Predicted total volume
    pred_total_volume = past_volume_denorm + volume_std * pred_diff_sum
    
    # Effective precipitation (accounting for infiltration)
    new_precip_term = denorm_avg_precip * infiltration_area_sum
    
    # Continuity constraints (using ReLU for inequality enforcement)
    term1 = ReLU((pred_total_volume - (past_volume + Δt * (inflow + precip))) / area)²
    term2 = ReLU((future_volume - pred_total_volume - Δt * (next_inflow + next_precip)) / area)²
    
    return mean(term1 + term2)
```

#### 3. Pushforward Loss (Stability)

For autoregressive stability during training:

```python
loss = one_step_loss + stability_loss
```

Where `stability_loss` uses the model's own predictions to create input for a second forward pass.

### Combined Loss

```python
total_loss = mse_loss + physics_loss_weight * physics_loss
```

Default `physics_loss_weight = 1.0` and `delta_t = 1200.0` seconds.

---

## Physics Laws Applied

### Mass Conservation (Continuity Equation)

The physics loss enforces the **shallow water continuity equation**:

$$\frac{\partial h}{\partial t} + \nabla \cdot (h \mathbf{u}) = S$$

In discrete form for a control volume:

$$V_{t+1} = V_t + \Delta t \cdot (Q_{in} + P - Q_{out} - I)$$

Where:
- $V$ = Volume
- $Q_{in}$ = Inflow discharge
- $P$ = Precipitation contribution
- $Q_{out}$ = Outflow discharge
- $I$ = Infiltration losses
- $\Delta t$ = Time step (default: 1200 seconds = 20 minutes)

### Constraint Implementation

The physics loss uses **ReLU activation** to create soft inequality constraints:

$$\mathcal{L}_{physics} = \text{ReLU}\left(\frac{V_{pred} - V_{expected}}{A}\right)^2$$

This penalizes predictions that violate mass conservation while allowing for some flexibility in the transition dynamics.

### Denormalization for Physics

Physics calculations are performed in **denormalized (physical) units**:

```python
past_volume_denorm = past_volume_norm * volume_std + num_nodes * volume_mean
```

---

## Model Architecture

### MeshGraphKAN

The model uses an **Encoder-Processor-Decoder** architecture with KAN enhancement:

```
┌─────────────────────────────────────────────────────────────┐
│                      MeshGraphKAN                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐                                        │
│  │  Node Encoder   │  KolmogorovArnoldNetwork               │
│  │  (KAN Layer)    │  input_dim → hidden_dim (128)          │
│  └────────┬────────┘                                        │
│           │                                                  │
│  ┌────────┴────────┐                                        │
│  │  Edge Encoder   │  MLP: edge_dim → hidden_dim            │
│  │  (MLP)          │  with LayerNorm                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│  ┌────────┴────────┐                                        │
│  │   Processor     │  15 Message Passing Blocks             │
│  │  (15 layers)    │  Edge Block → Node Block (alternating) │
│  └────────┬────────┘                                        │
│           │                                                  │
│  ┌────────┴────────┐                                        │
│  │  Node Decoder   │  MLP: hidden_dim → output_dim (2)      │
│  │  (MLP)          │                                        │
│  └─────────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Kolmogorov–Arnold Network (KAN)

The node encoder uses a KAN layer instead of a standard MLP for enhanced interpretability:

```python
self.node_encoder = KolmogorovArnoldNetwork(
    input_dim=input_dim_nodes,      # 16
    output_dim=hidden_dim_processor, # 128
    num_harmonics=5,                 # Fourier harmonics
    add_bias=True,
)
```

KAN uses **spline-based learnable activation functions** that provide:
- Better function approximation
- Improved interpretability
- Smoother learned representations

### Message Passing

Each processor block consists of:
1. **Edge Block**: Updates edge features using connected node features
2. **Node Block**: Updates node features by aggregating neighbor messages

```python
# For each of 15 iterations:
edge_features = EdgeBlock(edge_features, node_features, graph)
node_features = NodeBlock(node_features, edge_features, graph, aggregation="sum")
```

---

## Training Pipeline

### Training Configuration

```yaml
# config.yaml
batch_size: 1
epochs: 100
lr: 0.0001
lr_decay_rate: 0.9999979
weight_decay: 0.0001
n_time_steps: 2
use_physics_loss: true
physics_loss_weight: 1.0
delta_t: 1200.0
```

### Training Loop

```python
for epoch in range(epochs):
    for batch in dataloader:
        # Unpack batch
        if use_physics_loss:
            graph, physics_data = batch
        else:
            graph = batch
        
        # Forward pass
        pred = model(graph.x, graph.edge_attr, graph)
        
        # Compute losses
        mse_loss = criterion(pred, graph.y)
        
        if use_physics_loss:
            physics_loss = compute_physics_loss(pred, physics_data, graph, delta_t)
            loss = mse_loss + physics_loss_weight * physics_loss
        else:
            loss = mse_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
```

### Noise Injection Strategies

| Noise Type | Description |
|------------|-------------|
| `none` | No noise added |
| `pushforward` | Two-step training for autoregressive stability |
| `only_last` | Noise added only to last time step |
| `correlated` | Same noise across all time steps |
| `uncorrelated` | Independent noise at each time step |
| `random_walk` | Cumulative noise simulating drift |

### Distributed Training

Supports multi-GPU training via `DistributedDataParallel`:

```python
if dist.world_size > 1:
    model = DistributedDataParallel(
        model,
        device_ids=[dist.local_rank],
        output_device=dist.device,
    )
```

---

## Inference & Evaluation

### Autoregressive Rollout

During inference, the model predicts future states iteratively:

```python
for t in range(rollout_length):
    # Extract current state
    static_part = X[:, :12]
    water_depth_window = X[:, 12:12+n_time_steps]
    volume_window = X[:, 12+n_time_steps:12+2*n_time_steps]
    
    # Predict differences
    pred = model(X, edge_features, graph)  # [N, 2]
    
    # Update state (residual connection)
    new_wd = water_depth_window[:, -1:] + pred[:, 0:1]
    new_vol = volume_window[:, -1:] + pred[:, 1:2]
    
    # Slide window
    water_depth_updated = torch.cat([water_depth_window[:, 1:], new_wd], dim=1)
    volume_updated = torch.cat([volume_window[:, 1:], new_vol], dim=1)
    
    # Update forcing (inflow, precipitation)
    static_part[:, 10:12] = [new_inflow[t], new_precip[t]]
    
    # Form new input
    X = torch.cat([static_part, water_depth_updated, volume_updated], dim=1)
```

### Evaluation Metrics

**Root Mean Square Error (RMSE)** for water depth predictions:

$$\text{RMSE}_t = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(h_{pred,i,t} - h_{gt,i,t})^2}$$

### Visualization

Inference generates four-panel animations:
1. **Predicted water depth** (node colors)
2. **Ground truth water depth**
3. **Absolute error** (|predicted - ground truth|)
4. **RMSE over time** (cumulative error curve)

---

## Data Synthesis

### How to Create Synthetic/Custom Data

To synthesize data for HydroGraphNet, you need to generate or simulate the following components:

#### 1. Terrain Data (Static)

Generate or extract from DEM (Digital Elevation Model):

```python
# Required static files
- XY coordinates (node positions)
- Cell area
- Elevation
- Slope (can derive: slope = gradient(elevation))
- Aspect (direction of steepest descent)
- Curvature (second derivative of elevation)
- Manning's roughness coefficient (from land cover)
- Flow accumulation (accumulated upstream area)
- Infiltration parameter (from soil type)
```

#### 2. Hydrodynamic Simulation (Dynamic)

Run HEC-RAS or similar 2D flood model to generate:

```python
# For each scenario/hydrograph:
- Water depth at each node over time
- Volume at each node over time
- Upstream inflow hydrograph (boundary condition)
- Precipitation time series (if applicable)
```

#### 3. Data Format Requirements

```python
# Static data: single file per feature
np.savetxt(f"{prefix}_XY.txt", xy_coords, delimiter='\t')

# Dynamic data: one file per hydrograph scenario
np.savetxt(f"{prefix}_WD_{hydro_id}.txt", water_depth, delimiter='\t')
```

#### 4. Dataset Guidelines

- **Minimum time steps**: `n_time_steps + rollout_length` per hydrograph
- **Consistent node ordering**: Same node indices across all files
- **Unit consistency**: Water depth in meters, volume in m³
- **Peak clipping**: Data is trimmed to 25 steps after peak inflow

### Synthetic Data Generation Example

```python
import numpy as np

# Create synthetic terrain
num_nodes = 1000
xy = np.random.rand(num_nodes, 2) * 1000  # 1km x 1km domain
elevation = 100 - np.sqrt(xy[:, 0]**2 + xy[:, 1]**2) / 100  # Bowl shape

# Simulate simplified flood dynamics
time_steps = 100
water_depth = np.zeros((time_steps, num_nodes))
inflow = np.exp(-((np.arange(time_steps) - 50) / 10)**2) * 100  # Gaussian pulse

for t in range(1, time_steps):
    # Simple diffusion model
    water_depth[t] = water_depth[t-1] + 0.01 * inflow[t] - 0.001 * water_depth[t-1]

# Save in required format
np.savetxt("data/SYNTH_XY.txt", xy, delimiter='\t')
np.savetxt("data/SYNTH_WD_001.txt", water_depth, delimiter='\t')
```

---

## Requirements

### Python Dependencies

```text
# requirements.txt
mlflow>=2.1.1
hydra-core
wandb
termcolor>=2.1.1
torch_geometric>=2.6.1
torch_scatter>=2.1.2
scipy
tqdm
numpy
matplotlib
networkx
```

### System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | >= 3.8 |
| PyTorch | >= 2.0 |
| CUDA | >= 11.8 (for GPU) |
| RAM | >= 16 GB |
| GPU Memory | >= 8 GB (recommended) |

### Optional Dependencies

```text
nvidia-apex          # For FusedAdam optimizer
transformer-engine   # For optimized layer norms
```

---

## Configuration Reference

### Full config.yaml Parameters

```yaml
# Hydra settings
hydra:
  job:
    chdir: True
  run:
    dir: ./outputs_phy/

# Data paths
data_dir: ./data
test_dir: ./data/Test

# Training parameters
batch_size: 1
epochs: 100
num_training_samples: 400
num_training_time_steps: 300
lr: 0.0001
lr_decay_rate: 0.9999979
weight_decay: 0.0001

# Model architecture
num_input_features: 16      # Node feature dimension
num_output_features: 2      # Output dimension (depth, volume diff)
num_edge_features: 3        # Edge feature dimension

# Temporal settings
n_time_steps: 2             # Sliding window size
noise_type: "none"          # Noise strategy

# Physics loss
use_physics_loss: true
delta_t: 1200.0             # Time step in seconds
physics_loss_weight: 1.0

# Performance
use_apex: True
amp: False                  # Automatic mixed precision
jit: False
num_dataloader_workers: 4
do_concat_trick: False
num_processor_checkpoint_segments: 0
recompute_activation: False

# Logging
wandb_mode: disabled
watch_model: False

# Checkpointing
ckpt_path: "./checkpoints_phy"

# Testing
num_test_samples: 10
num_test_time_steps: 30
```

---

## Quick Start Commands

### Training

```bash
cd physicsnemo/examples/weather/flood_modeling/hydrographnet
python train.py --config-path conf --config-name config
```

### Inference

```bash
python inference.py --config-path conf --config-name config
```

### With Custom Config Overrides

```bash
python train.py epochs=200 lr=0.0005 use_physics_loss=true
```

---

## Summary

HydroGraphNet implements a complete physics-informed machine learning pipeline for flood forecasting:

| Component | Implementation |
|-----------|----------------|
| **Data** | Graph-structured with static terrain + dynamic hydrograph features |
| **Model** | MeshGraphKAN (GNN with KAN encoder) |
| **Training** | MSE + Physics loss with optional pushforward |
| **Physics** | Mass conservation via continuity equation |
| **Inference** | Autoregressive rollout with residual updates |
| **Evaluation** | RMSE tracking with visualization |

The system achieves a balance between data-driven learning and physical constraints, enabling accurate and physically consistent flood predictions.
