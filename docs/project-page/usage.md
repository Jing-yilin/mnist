# Usage

## Environment Setup

The MNIST Training Experiment supports multiple computing environments through Pixi:

```bash
# Install CPU environment
pixi install

# Install CUDA environment (GPU acceleration)
pixi install --environment cuda

# Install MPS environment (Apple Silicon GPU acceleration)
pixi install --environment mps
```

## Data Preparation

Before training, you need to download and prepare the MNIST dataset:

```bash
# Download and prepare MNIST dataset
pixi run prepare-data

# Verify data integrity (optional but recommended)
pixi run verify-data-hash
```

## Training

You can train the model in different environments:

```bash
# Train with CPU
pixi run train-model

# Train with CUDA (if available)
pixi run --environment cuda train-model

# Train with MPS (Apple Silicon)
pixi run --environment mps train-model
```

Training parameters can be customized:

```bash
# Train with custom parameters
pixi run train-model -- --batch-size 128 --epochs 10 --lr 0.01
```

## Testing

After training, you can evaluate the model:

```bash
# Test with CPU
pixi run test-model

# Test with CUDA
pixi run --environment cuda test-model
```

## Experiment Tracking

The project uses MLflow for tracking experiments:

```bash
# View experiment results in MLflow UI
pixi run mlflow ui --backend-store-uri ./mlruns
```

Then open http://localhost:5000 in your browser to view training metrics and results.

## Comparing Results

You can compare results from different environments:

```bash
# Compare CPU and CUDA results
cat results/cpu/mnist_results.json
cat results/cuda/mnist_results.json
```

## Reproduction Workflow

For reproducible research, use the reproduction workflow:

```bash
# Run complete reproducible experiment workflow
pixi run reproduction
``` 