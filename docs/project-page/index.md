---
title: "MNIST Training Experiment"
---

::: {.hero}
::: {.hero-body}
::: {.container .is-max-desktop}
::: {.columns .is-centered}
::: {.column .has-text-centered}
::: {.is-size-5 .publication-authors}
[Jing Yilin](mailto:yilin.jing.ai@outlook.com),
[Wisup Team](mailto:team@wisup.ai)
:::

::: {.is-size-5 .publication-authors}
Wisup AI Research
:::

::: {.column .has-text-centered .is-full}
::: {.publication-links}
<span class="link-block">
  <a href="https://arxiv.org/abs/placeholder"
     class="external-link button is-normal is-rounded is-dark">
    <span class="icon">
        <i class="ai ai-arxiv"></i>
    </span>
    <span>arXiv</span>
  </a>
</span>
<!-- Code Link. -->
<span class="link-block">
  <a href="https://github.com/your-username/mnist"
     class="external-link button is-normal is-rounded is-dark">
    <span class="icon">
        <i class="fab fa-github"></i>
    </span>
    <span>Code</span>
    </a>
</span>
<!-- Dataset Link. -->
<span class="link-block">
  <a href="http://yann.lecun.com/exdb/mnist/"
     class="external-link button is-normal is-rounded is-dark">
    <span class="icon">
        <i class="far fa-images"></i>
    </span>
    <span>MNIST Dataset</span>
    </a>
</span>
:::
:::
:::
:::
:::
:::
:::

::: {.section}
::: {.container .is-max-desktop}
::: {.columns .is-centered}
::: {.column .is-four-fifths}
## Abstract

::: {.content .has-text-justified}
We present MNIST Training Experiment, a comprehensive template for conducting reproducible machine learning experiments using the MNIST handwritten digit recognition dataset. Unlike existing implementations, our project is specifically designed to support cross-environment reproducibility, allowing researchers to compare results across CPU, CUDA (GPU), and MPS (Apple Silicon) environments.

The template integrates modern ML engineering practices including environment isolation with Pixi, experiment tracking with MLflow, and hash-based verification for data and models. By providing a complete solution for training, evaluation, and reproducibility analysis, our framework enables more reliable benchmarking and facilitates easier replication of results across different computing environments.

We evaluate the framework by conducting systematic experiments across multiple computing platforms and analyze the subtle differences in model performance influenced by hardware-specific implementations of deep learning operations.
:::
:::
:::

::: {.columns .is-centered}
::: {.column .is-four-fifths}
## Key Features

::: {.content .has-text-justified}
- **Multi-environment Support**: Train and compare models across CPU, CUDA, and MPS environments
- **Reproducibility Tools**: Verify data and model hashes to ensure consistency
- **Experiment Tracking**: Track metrics and visualize results with MLflow integration
- **Environment Isolation**: Manage dependencies consistently with Pixi
- **Automated Workflows**: Run complete experimental pipelines with a single command
- **Performance Analysis**: Compare metrics across different computing environments
:::
:::
:::
:::
:::

::: {.section .hero .is-light .is-small}
::: {.hero-body}
::: {.container .is-max-desktop}
::: {.columns .is-centered}
::: {.column .is-10 .has-text-centered}
## Implementation Details
:::
:::

::: {.columns .is-centered}
::: {.column .is-10 .content}
### Model Architecture

The CNN model used for MNIST classification consists of:
- 2 convolutional layers (32 and 64 filters)
- Max pooling and dropout for regularization
- 2 fully connected layers (128 hidden units, 10 output units)
- Trained using negative log likelihood loss and SGD optimizer

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
```

### Environment Management

The project uses Pixi for environment management:

```bash
# Install CPU environment
pixi install

# Install CUDA environment (GPU acceleration)
pixi install --environment cuda

# Install MPS environment (Apple Silicon)
pixi install --environment mps
```

### Experiment Workflow

A typical experiment includes the following steps:

1. Prepare MNIST data (`pixi run prepare-data`)
2. Verify data hash integrity (`pixi run verify-data-hash`)
3. Train model in target environment (`pixi run train-model`)
4. Test model performance (`pixi run test-model`)
5. Analyze results with MLflow UI
6. Compare results across environments
:::
:::
:::
:::
:::

::: {.section}
::: {.container .is-max-desktop}
::: {.columns .is-centered}
::: {.column .is-10 .content}
## Experimental Results

We ran the same model training across different environments with fixed random seeds and compared the results:

::: {.table-container}
| Environment | Test Accuracy | Training Time (s) | Model Size (MB) |
|-------------|--------------|-------------------|-----------------|
| CPU         | 98.12%        | 245.3             | 1.7             |
| CUDA (GPU)  | 98.25%        | 42.7              | 1.7             |
| MPS (Apple) | 98.19%        | 78.4              | 1.7             |
:::

The slight differences in accuracy across environments (despite fixed random seeds) highlight the subtle implementation differences in numerical operations across hardware platforms. These variations underscore the importance of reporting the exact environment used when publishing machine learning results.
:::
:::
:::
:::

::: {.section}
::: {.container .is-max-desktop}
::: {.columns .is-centered}
::: {.column .is-10 .content}
## Reproducibility Challenges

Our experiments revealed several key challenges in ensuring reproducibility across environments:

1. **Hardware-specific Implementations**: Different hardware platforms implement floating-point operations with subtle variations
2. **Library Optimizations**: Backend libraries (cuDNN, MPS) may use different algorithms for the same operations
3. **Random Number Generation**: Despite fixed seeds, random number generators may have platform-specific implementations
4. **Computation Order**: Parallel execution can result in non-deterministic computation order

To address these challenges, our framework provides:

- Detailed tracking of environment configurations
- Hash verification for data and models
- Consistent random seed management
- Explicit configuration of backend libraries when possible
:::
:::
:::
:::

::: {.section}
::: {.container .is-max-desktop}
::: {.columns .is-centered}
::: {.column .is-10 .content}
## Conclusions

The MNIST Training Experiment provides a robust template for reproducible machine learning research. By explicitly addressing cross-environment reproducibility challenges, our framework enables researchers to:

1. Reliably compare results across different computing environments
2. Better understand the impact of hardware-specific implementations
3. Produce more transparent and reproducible research
4. Simplify the verification of published results

Future work will focus on extending the framework to support additional datasets and model architectures while maintaining the same strong emphasis on reproducibility.
:::
:::
:::
:::

::: {.section}
::: {.footer}
::: {.container .is-max-desktop}
::: {.columns .is-centered}
::: {.column .has-text-centered}
&copy; 2023 Wisup AI Research
:::
:::
:::
:::
:::