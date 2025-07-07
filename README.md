# REANIMATOR: Reanimate Retrieval Test Collections with Extracted and Synthetic Resources

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview
REANIMATOR is a versatile framework designed to enhance and repurpose existing retrieval test collections by enriching them with extracted and synthetic resources. It enables the parsing of full texts, machine-readable tables, and contextual metadata from PDF files. Additionally, it leverages state-of-the-art large language models to generate synthetic relevance labels, with an optional human-in-the-loop validation step.

We showcase its potential by revitalizing the TREC-COVID test collection, demonstrating how retrieval-augmented generation (RAG) systems can be developed and evaluating the impact of tables on RAG performance. REANIMATOR lowers costs and broadens the utility of legacy resources, making them reusable for new applications.

## Features
- **Automated Data Extraction**: Parses full texts and structured tables from PDFs.
- **Synthetic Relevance Labeling**: Utilizes large language models to generate annotations.
- **Human-in-the-Loop Validation**: Optional verification step for quality assurance.
- **Parallelized Processing**: Efficient execution to handle large datasets.
- **RAG System Integration**: Enables research on retrieval-augmented generation.

## Project Structure
```
.
├── data/                  # Data files (original and processed)
├── docker-compose.yml     # Docker Compose configuration for CUDA
├── Docker_ARM/            # Docker configuration for ARM-based systems (e.g., Apple Silicon)
├── Docker_CUDA/           # Docker configuration for NVIDIA GPUs (default)
├── Docker_NO_GPU/         # Docker configuration for non-GPU environments
├── Dockerfile             # Main Dockerfile, sourced by compose files
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── pyproject.toml         # Project configuration and dependencies
├── README.md              # This README file
└── src/
    └── reanimator/        # Source code for the reanimator package
        ├── __init__.py
        ├── core.py        # Main pipeline orchestration
        ├── downloaders.py # PDF downloading logic
        ├── extractors.py  # Content extraction from documents
        ├── labelers.py    # Synthetic query and label generation
        ├── models.py      # Data models (Document, Query, etc.)
        ├── preprocessing/ # Scripts and notebooks for data preparation
        ├── retrieval.py   # Retrieval and ranking pipelines
        └── sources.py     # Data source wrappers (e.g., ir_datasets)
```

## Installation

You can install the `reanimator` package using pip. It's recommended to do this within a virtual environment.

### Local Development (with venv)

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        .\\venv\\Scripts\\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install the package in editable mode:**
    ```bash
    pip install -e .
    ```
    This allows you to make changes to the source code and have them immediately reflected in your environment.

### From GitHub

You can also install the package directly from the GitHub repository:

```bash
pip install git+https://github.com/irgroup/Reanimator.git
```

## Usage

The core functionality of this project is accessible through the `reanimate` command-line script. This script runs the entire pipeline, from downloading documents to generating labels and running retrieval experiments.

### Run the Full Pipeline
This single command orchestrates the entire process. You need to provide the name of an `ir_datasets` collection and an email address for the Unpaywall API.
```bash
reanimate <IRDS_NAME> --email YOUR_EMAIL@example.com
```

**Example:**
To run the pipeline on the `cranfield` dataset:
```bash
reanimate cranfield --email me@example.com
```

You can also limit the number of documents to process for a quicker test run using the `--max_docs` argument:
```bash
reanimate cranfield --email me@example.com --max_docs 100
```

The processed documents and intermediate files will be saved in the `data/` directory.

## Data Resources
The original data resources for this project are available via [Google Drive](https://drive.google.com/drive/folders/1IqhijGWffGQ5ZjE7JrGTDAwPq_PGFVXD?usp=sharing). The `reanimate` script will automatically handle the downloading and processing of necessary data.

## Citation
If you use REANIMATOR in your research, please cite our paper:

```
@inproceedings{reanimator2025,
  title={REANIMATOR: Reanimate Retrieval Test Collections with Extracted and Synthetic Resources},
  author={Björn Engelmann, Fabian Haak, Philipp Schaer, Mani Erfanian Abdoust, Linus Netze, Meik Bittkowski},
  booktitle={},
  year={2025}
}
```

## License
This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.


