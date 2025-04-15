# MNIST Neural Network Documentation

This directory contains two separate documentation projects built with Quarto:

## 1. Project Page (`project-page/`)

An interactive website for demonstrating the MNIST neural network training and inference. It includes:

- Interactive visualizations of the training process
- A drawing tool to test the model in real-time
- Detailed methodology explanations
- Visual exploration of model internals (weights, activations)

**To build and serve the project page:**

```bash
cd project-page
quarto preview
```

To build for deployment:

```bash
cd project-page
quarto render
```

## 2. ArXiv Article (`arxiv-article/`)

A research article formatted for submission to arXiv, documenting our neural network training experiments on MNIST. It includes:

- Comprehensive methodology description
- Experimental results and analysis
- Mathematical formulations
- Citations and references

**To build the article:**

```bash
cd arxiv-article
quarto render
```

This will generate both PDF (for arXiv submission) and HTML versions.

## Requirements

- [Quarto](https://quarto.org/) (1.3.0+)
- Python 3.8+ with:
  - matplotlib
  - numpy
  - tensorflow (for code execution)
- LaTeX (for PDF rendering of the arXiv article)

## Directory Structure

```
docs/
├── project-page/           # Interactive website
│   ├── _quarto.yml         # Quarto configuration
│   ├── index.qmd           # Homepage
│   ├── methodology.qmd     # Methods description
│   ├── results.qmd         # Results page
│   ├── interactive-demo.qmd # Interactive demo
│   ├── team.qmd           # Team information
│   ├── styles.css         # Custom styling
│   └── assets/            # Images and resources
│
├── arxiv-article/          # ArXiv paper
│   ├── _quarto.yml         # Quarto configuration
│   ├── index.qmd           # Main article
│   ├── methods.qmd         # Detailed methods
│   ├── references.bib      # Bibliography
│   ├── ieee.csl            # Citation style
│   └── figures/            # Article figures
│
└── README.md               # This file 