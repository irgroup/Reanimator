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
├── docker-compose.yml     # Docker Compose configuration
├── Docker_NO_GPU/         # Docker configuration for non-GPU environments
├── Dockerfile             # Main Dockerfile for GPU environments
├── llm_models/            # Configuration for LLMs
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── pyproject.toml         # Project configuration and dependencies
├── README.md              # This README file
└── src/
    └── reanimator/        # Source code for the reanimator package
        ├── scripts/       # Command-line scripts
        ├── __init__.py
        ├── helpers.py
        ├── labeling/
        ├── parallel_exec/
        └── preprocessing/
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/irgroup/Reanimator.git
cd Reanimator
```

### 2. (Optional) Set up Docker Environment
This project is designed to run inside a Docker container to ensure reproducibility.

- **For NVIDIA GPU users:**
  Build and run the container using Docker Compose:
  ```bash
  docker compose up --build
  ```

- **For non-NVIDIA GPU users:**
  Replace the Docker files with the non-GPU versions before building:
  ```bash
  cp Docker_NO_GPU/Dockerfile .
  cp Docker_NO_GPU/docker-compose.yml .
  docker compose up --build
  ```

Once the container is running, you can attach your IDE (e.g., VS Code) to the container for development.

### 3. Install the Package
Inside the Docker container, or in your own Python environment (>= 3.10) with the prerequisites installed, install the `reanimator` package:
```bash
pip install .
```
This will also install all the required dependencies and make the command-line scripts available.

## Usage

The core functionality of this project is accessible through command-line scripts.

### 1. Get URLs for Document DOIs
This script uses the Unpaywall API to find direct PDF URLs for DOIs from the CORD-19 dataset.
```bash
get_urls_for_dois --email YOUR_EMAIL@example.com
```
The URLs will be saved to `data/next_pdf_urls.pkl`.

### 2. Download PDFs
This script downloads the PDFs from the URLs gathered in the previous step.
```bash
download_pdfs
```
The downloaded PDFs will be saved in the `data/pdfs` directory.

## Data Resources
The original data resources for this project are available via [Google Drive](https://drive.google.com/drive/folders/1IqhijGWffGQ5ZjE7JrGTDAwPq_PGFVXD?usp=sharing). After running the processing scripts, the `data` directory will be populated with the necessary files.

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


