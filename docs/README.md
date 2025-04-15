# MNIST Data Exploration Project

This directory contains MNIST dataset exploration and model analysis documentation created using Quarto.

## File Structure

- `_quarto.yml` - Quarto project configuration file
- `index.qmd` - Project homepage
- `data-exploration.qmd` - MNIST dataset exploration analysis
- `model-analysis.qmd` - Training model performance analysis
- `styles.css` - Custom stylesheet

## Usage Instructions

### Prerequisites

Before using, please ensure you have installed the required dependencies:

```bash
# Install project dependencies
pixi install

# Install Quarto (if not already installed)
# Please download and install from the official website according to your system: https://quarto.org/docs/get-started/

# Manually install Python packages required for Quarto documents (in the pixi environment)
pip install jupyter ipykernel matplotlib pandas scikit-learn seaborn
```

### Preparing Data and Models

Before viewing the Quarto documentation, you need to prepare the data and train the model:

```bash
# Download data
pixi run prepare-data

# Train model
pixi run train-model

# Test model
pixi run test-model
```

### Preview Quarto Documents

Use the following command to start the Quarto preview server:

```bash
pixi run quarto-preview
```

This will start a local server and automatically open the document preview in your browser. The preview will update automatically when you modify Quarto files.

### Render Quarto Documents

To render Quarto documents as a static HTML website, run:

```bash
pixi run quarto-render
```

The rendered files will be saved in the `_site` directory.

### Publish Quarto Documents

If you want to publish the documentation to GitHub Pages or other supported platforms:

```bash
pixi run quarto-publish
```

## Customization and Extension

You can customize and extend this project by editing the following files:

- Modify `_quarto.yml` to change the website theme and navigation
- Edit `.qmd` files to add new analysis content
- Customize styles in `styles.css`

## Notes

- This Quarto project depends on MNIST data and trained models
- Relative paths in code blocks are relative to the docs directory 