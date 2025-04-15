# Contributing

Thank you for considering contributing to the MNIST Training Experiment project!

## Development Environment Setup

This project uses Pixi for environment management. Follow these steps to set up your development environment:

```bash
# Clone the repository
git clone https://github.com/your-username/mnist.git
cd mnist

# Install dependencies (CPU environment)
pixi install

# Or install CUDA environment (GPU acceleration)
pixi install --environment cuda
```

## Data Preparation

Before starting development, make sure you have prepared the data:

```bash
# Download and prepare MNIST dataset
pixi run prepare-data

# Verify data hash
pixi run verify-data-hash
```

## Code Style

We use Black for code formatting and follow PEP 8 standards:

```bash
# Format code using Black
black train.py test.py scripts/*.py

# Check code style
flake8 train.py test.py scripts/*.py
```

## Testing

Ensure you run tests before submitting changes:

```bash
# Test with default CPU environment
pixi run test-model

# If you have a GPU, also test in CUDA environment
pixi run --environment cuda test-model
```

## Documentation

We use Quarto to generate documentation:

```bash
# Generate documentation
cd docs
quarto render
```

## Pull Request Process

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Submit a Pull Request

## Experiment Reproducibility

When modifying training code, ensure:

1. Random seeds are fixed to ensure reproducible results
2. Data and model hash values are updated
3. Tests are run in multiple environments (e.g., CPU and CUDA)
4. Results from different environments are recorded and compared
5. MLflow configuration is updated to track new metrics

## Guidelines

- Ensure your code runs in all supported environments (CPU, CUDA, and MPS)
- Add appropriate documentation and comments
- Follow existing code structure and style
- Update task definitions in pixi.toml when adding new features 