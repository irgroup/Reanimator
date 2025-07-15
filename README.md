# REANIMATOR: Reanimate Retrieval Test Collections with Extracted and Synthetic Resources

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview
REANIMATOR is a versatile framework for enhancing retrieval test collections, starting from a given set of DOIs or PDFs. It leverages a state-of-the-art PDF parsing pipeline ([docling](https://github.com/docling-project/docling)), [langchain](https://www.langchain.com/) for preprocessing, and generates [Umbrela](https://github.com/castorini/umbrela)-like synthetic relevance judgments. This allows for the creation of rich, new resources from existing test collections, enabling research on retrieval-augmented generation (RAG) and the impact of different data modalities like tables. For a detailed overview, please refer to our paper: [REANIMATOR: Reanimate Retrieval Test Collections with Extracted and Synthetic Resources](https://arxiv.org/pdf/2504.07584).

Key features include:
- Automated data extraction from PDFs (full-text and tables) from a given list of DOIs or local files.
- Synthetic relevance labeling using LLMs.
- Optional human-in-the-loop.
- Parallelized processing for efficiency.

## Project Structure
```
.
├── Docker_ARM/            # Docker configuration for ARM-based systems
├── Docker_CUDA/           # Docker configuration for NVIDIA GPUs
├── Docker_NO_GPU/         # Docker configuration for non-GPU environments
├── extensions/            # Setup for external services (e.g., labeling)
├── notebooks/             # Jupyter notebooks for examples and analysis
├── pyproject.toml         # Project configuration and dependencies
├── README.md              # This README file
└── src/
    └── reanimator/        # Source code for the reanimator package
        ├── core.py        # Main pipeline orchestration
        ├── downloaders.py # PDF downloading logic
        ├── extractors.py  # Content extraction from documents
        ├── labeling.py    # Human relevance labeling
        ├── labelers.py    # Synthetic query and label generation
        ├── models.py      # Data models (Document, Query, etc.)
        ├── retrieval.py   # Retrieval and ranking pipelines
        └── sources.py     # Data source wrappers
```

## Installation

### Local Development (with venv)
It is recommended to install the package in a virtual environment.

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    *   On Windows: `.\\venv\\Scripts\\activate`
    *   On macOS and Linux: `source venv/bin/activate`

3.  **Install the package:**
    For development, install in editable mode, which allows you to modify the source code and have the changes immediately reflected:
    ```bash
    pip install -e .
    ```
    For a standard installation, use pip to install directly from the GitHub repository:
    ```bash
    pip install git+https://github.com/irgroup/Reanimator.git
    ```

### Using Docker
This project also supports Docker for containerized environments. We provide configurations for different hardware setups:
-   `Docker_CUDA/`: For systems with NVIDIA GPUs.
-   `Docker_ARM/`: For ARM-based systems like Apple Silicon.
-   `Docker_NO_GPU/`: For environments without a dedicated GPU.

Each directory contains a `docker-compose.yml` and `Dockerfile` for that specific setup. To build and run the Docker container, navigate to the directory corresponding to your hardware and use the following commands:
```bash
# Build the Docker image
docker build -t reanimator .

# Start the services
docker-compose up
```

## Usage
For a practical guide on how to use the REANIMATOR framework, please see the example notebook:
[notebooks/example_reanimation.ipynb](./notebooks/example_reanimation.ipynb)

This notebook provides a step-by-step walkthrough of the data processing and reanimation pipeline.

## Data Resources
The original data resources for this project are available on [Google Drive](https://drive.google.com/drive/folders/1IqhijGWffGQ5ZjE7JrGTDAwPq_PGFVXD?usp=sharing). This data was generated with an older version of the REANIMATOR framework.

## Citation
If you use REANIMATOR in your research, please cite our paper, to be published at The 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), July 13-17, 2025 in Padua, Italy:

```
@inproceedings{engelmann2025reanimator,
  author    = {Björn Engelmann and Fabian Haak and Philipp Schaer and Mani Erfanian Abdoust and Linus Netze and Meik Bittkowski},
  title     = {{REANIMATOR:} Reanimate Retrieval Test Collections with Extracted and Synthetic Resources},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25)},
  year      = {2025},
  month     = {July},
  address   = {Padua, Italy},
  doi       = {10.1145/3726302.3730342}
}
```

## License
This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.


