# MNIST Training Experiment Documentation

This directory contains the documentation for the MNIST Training Experiment project.

## Documentation Structure

The documentation is organized into the following sections:

- **Main Documentation**: General information, usage instructions, and API reference
- **Project Page**: A showcase website for the project
- **arXiv Article**: A research paper explaining the approach and findings

## Building Documentation Locally

### Prerequisites

- [Quarto](https://quarto.org/docs/get-started/) (v1.3.0 or higher)
- Python 3.10 or higher
- Required Python packages: jupyter, matplotlib, pandas, numpy

### Installing Quarto Extensions

Before building the documentation, install the required Quarto extensions:

```bash
cd docs
quarto add quarto-ext/fontawesome
quarto add grantmcdermott/quarto-revealjs-clean
quarto add mikemahoney218/quarto-arxiv
```

### Building the Documentation

To build the main documentation:

```bash
cd docs
quarto render
```

The output will be available in the `_output` directory.

### Building the Project Page

To build the project showcase page:

```bash
cd docs/project-page
quarto render
```

The output will be available in the `_site` directory.

### Building the arXiv Article

To build the research paper:

```bash
cd docs/arxiv-article
quarto render
```

The output will be available in the `_site` directory with both PDF and HTML versions.

## Continuous Deployment

The documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch. The deployment is handled by the GitHub Actions workflow defined in `.github/workflows/docs.yml`.

The published documentation is available at: `https://[username].github.io/mnist/`

## Contributing to Documentation

When contributing to the documentation:

1. Make changes to the relevant Markdown or Quarto files
2. If adding new sections, update the navigation in `_quarto.yml`
3. Build and test locally before pushing changes
4. Push changes to the main branch to trigger automatic deployment

## Documentation Format

The documentation uses [Quarto](https://quarto.org/), a scientific and technical publishing system that supports Markdown, Jupyter notebooks, and more.

For guidance on Quarto syntax and features, refer to the [Quarto documentation](https://quarto.org/docs/guide/). 