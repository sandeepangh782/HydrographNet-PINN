# HydroGraphNet: Interpretable Physics-Informed Graph Neural Networks for Flood Forecasting

HydroGraphNet is a physics-informed graph neural network for
large-scale flood dynamics modeling. It integrates physical
consistency, autoregressive forecasting, and interpretability
through Kolmogorov–Arnold Networks (KANs) to deliver accurate
and explainable predictions of water depth and volume during
flooding events.

## Problem Overview

Floods, driven by climate-induced hydrologic extremes, pose
significant risks to communities and infrastructure. Accurate
and timely flood forecasts are critical for early warning systems
and resilience planning. However, traditional hydrodynamic models,
based on solving the shallow water equations, are computationally
expensive and unsuitable for real-time forecasting.

HydroGraphNet addresses this challenge by offering a fast, physically
consistent, and interpretable surrogate model using Graph Neural Networks.
It leverages unstructured spatial meshes and incorporates physical constraints
to maintain mass balance without the overhead of automatic differentiation.

## Model Overview and Architecture

### HydroGraphNet

HydroGraphNet uses an autoregressive encoder-processor-decoder GNN architecture
to predict water depth and volume across multiple future time steps. The
architecture comprises:

- **Encoder:** Initializes node and edge features from spatial and hydrologic inputs.
- **Processor:** A multi-layer message-passing network that refines node and edge features.
- **Decoder:** Outputs the predicted changes in depth and volume,
which are added to the previous state using residual connections.

The model integrates:

- **Physics-informed loss:** Ensures mass conservation using volume continuity
inequalities.
- **Pushforward trick:** Reduces autoregressive error propagation.
- **Kolmogorov–Arnold Networks (KAN):** Enhances model interpretability by
replacing MLPs with spline-based function networks.

The training and inference pipelines use node features that include both
static (e.g., elevation, slope, roughness) and dynamic (e.g., water depth,
volume history) attributes, along with global forcings such as inflow hydrograph
 and precipitation.

## Dataset

HydroGraphNet is validated on a real-world case study from the White River
near Muncie, Indiana. The dataset consists of:'

- A spatial graph of 4,787 nodes,
- Boundary inflow conditions and rainfall time series,
- Ground truth water depth and volume over time from high-fidelity HEC-RAS simulations.

The graph representation allows flexible modeling of both fluvial and
pluvial flood dynamics across urban and rural terrains.

## Training the Model

To train HydroGraphNet:

1. Prepare your dataset following the graph-based structure used in `HydroGraphDataset`.

2. Configure training parameters in `conf/config.yaml`.

3. Run the training script:


    # HydroGraphNet

    This repository contains code for training and inference of the HydroGraphNet model for flood forecasting using graph neural networks.

    ## Environment Setup for GPU (CUDA)

    To utilize GPU acceleration (CUDA) for training, follow these steps:

    ### 1. Install CUDA Toolkit and cuDNN

    - Ensure your system has a compatible NVIDIA GPU.
    - Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) matching your PyTorch version.
    - Add CUDA to your PATH and LD_LIBRARY_PATH as per NVIDIA instructions.

    ### 2. Create a Conda Environment (Recommended)

    ```bash
    conda create -n hydrographnet python=3.9
    conda activate hydrographnet
    ```

    ### 3. Install PyTorch with CUDA Support

    - Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select your CUDA version.
    - Example for CUDA 11.8:

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

    ### 4. Install Other Dependencies

    ```bash
    pip install -r requirements.txt
    ```

    ### 5. Verify CUDA Availability

    Run the following in Python:
    ```python
    import torch
    print(torch.cuda.is_available())  # Should return True
    print(torch.cuda.device_count())  # Number of GPUs detected
    ```

    ### 6. Configuration Changes

    - In `conf/config.yaml`, set:
      - `use_apex: True` (if using NVIDIA Apex for mixed precision; install Apex if needed)
      - `amp: True` (for PyTorch native mixed precision)
      - `num_dataloader_workers: >0` (for faster data loading, if supported)
    - Ensure your batch size and model fit in GPU memory.

    ### 7. Run Training on GPU

    ```bash
    python train.py
    ```

    The script will automatically use GPU if available. For distributed/multi-GPU training, refer to PyTorch Distributed documentation and ensure `DistributedManager` is properly configured.

    ### 8. Troubleshooting

    - If CUDA is not detected, check your driver, CUDA, and cuDNN installation.
    - For Apex, install via:
      ```bash
      git clone https://github.com/NVIDIA/apex
      cd apex
      pip install .
      ```
    - If using Mac, GPU acceleration is not supported; training will run on CPU.

    ## Standard Setup (CPU)

    If you do not have a GPU, follow the original steps:

    1. Clone the repository:
        ```bash
        git clone <repo-url>
        cd hydrographnet
        ```

    2. Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```

    3. Configure training parameters in `conf/config.yaml`.

    4. Run training:
        ```bash
        python train.py
        ```

    ## Inference

    Run inference using:
    ```bash
    python inference.py
    ```

    ## Notes

    - Training and inference scripts use Hydra for configuration management.
    - Outputs and checkpoints are saved in the `outputs_phy/` and `checkpoints_phy/` directories.
    - For GPU training, ensure CUDA and cuDNN are installed and PyTorch is built with CUDA support.

    ## Citation

    If you use this code, please cite:

    ...existing code...
optional physics-aware targets.
- **Test mode**: Returns a full graph and a rollout dictionary
for inference.

To use the dataset, simply instantiate:

```python
from hydrographnet_dataset import HydroGraphDataset

dataset = HydroGraphDataset(
    data_dir="./data",
    prefix="M80",
    split="train",  # or "test"
    n_time_steps=2,
    return_physics=True
)
```

This will ensure the data is downloaded, normalized, and ready for GNN training or evaluation.

## Logging

HydroGraphNet supports logging via [Weights & Biases (W&B)](https://wandb.ai/):

- Training and validation losses
- Physics-based loss contributions
- Learning rate schedule

Set up W&B by modifying `wandb_mode` and `watch_model` in `config.yaml`.

🍎 Mac CPU-Only Environment Setup
1. Create & Activate Virtual Environment

cd /Users/sandeepanghosh/Documents/hydrographnet

# Create virtual environment
python3 -m venv hydro_env

# Activate it
source hydro_env/bin/activate
2. Install PyTorch (CPU version for Mac)

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch for CPU (Mac compatible)
pip install torch torchvision torchaudio
3. Install PyTorch Geometric (CPU version)

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
4. Install Other Requirements

pip install -r requirements.txt
5. Install PhysicsNEMO/Modulus

pip install nvidia-modulus
Or if that fails:


pip install git+https://github.com/NVIDIA/modulus.git
6. Verify Installation

python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "import torch_geometric; print('✅ PyTorch Geometric installed')"
python -c "import wandb; print('✅ WandB installed')"
⚙️ Adjust Training for CPU
Since you're on CPU, training will be slower. I recommend starting with a smaller test run:

Quick Test (2 epochs, 50 samples):

source hydro_env/bin/activate
wandb login
python train.py epochs=2 num_training_samples=50
Expected time: ~10-20 minutes

Adjust for Faster CPU Training
You can modify the config for better CPU performance:

Edit config.yaml
Modified
Perfect! Config updated for Mac CPU.

📊 Training Expectations on Mac CPU
Full training (100 epochs, 400 samples):

Estimated time: 8-24 hours (depending on your Mac's CPU)
This will run overnight
Recommended approach:

Test run first (2 epochs): Verify everything works
Small training run (10-20 epochs): See if results are good
Full training (100 epochs): Run overnight or over weekend
🚀 Complete Workflow

# 1. Activate environment
cd /Users/sandeepanghosh/Documents/hydrographnet
source hydro_env/bin/activate

# 2. Login to WandB (one-time)
wandb login

# 3. Quick test (2 epochs)
python train.py epochs=2 num_training_samples=50

# 4. If successful, run small training
python train.py epochs=10 num_training_samples=100

# 5. Full training (leave running overnight)
python train.py
💡 Tips for Mac CPU Training
Monitor temperature:


# Mac will get warm during training
# You can check CPU usage with Activity Monitor
Run in background (optional):


# To keep training if you close terminal
nohup python train.py > training.log 2>&1 &

# Check progress
tail -f training.log
Save battery:

Keep Mac plugged in during training
Training is CPU-intensive
📈 What to Expect
Console output every epoch:


Epoch 0 | Loss: 1.234e-03 | RMSE (depth/vol): 0.0542/0.0398 | R²: 0.856 | LR: 1.00e-04 | Time: 450.3s
Note: Time: 450s (~7-8 min per epoch on CPU) vs Time: 125s (~2 min on GPU)

WandB Dashboard:

https://wandb.ai/sandeepangh/HydroGraphNet
Real-time metrics even from Mac!
You're all set! Start with the test run to verify everything works, then let it train overnight for the full 100 epochs. 🎉