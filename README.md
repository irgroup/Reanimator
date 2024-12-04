# cord19plus

This guide will walk you through setting up the **cord19plus** project using Docker and Docker Compose with NVIDIA GPU capabilities. If you do not have an NVIDIA GPU, alternative instructions are provided.

---

## Prerequisites

- **Docker** and **Docker Compose** installed on your system.
- **NVIDIA GPU** with the appropriate drivers (if available).
- An IDE like **Visual Studio Code** or **Cursor**.
- **Git** installed on your system.

---

## Installation Steps

### 1. Install Docker and Docker Compose with NVIDIA Capabilities

Ensure that Docker and Docker Compose are installed with NVIDIA GPU support.

- **For NVIDIA GPU users:**
  - Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to enable GPU support in Docker.
  - Verify the installation with:

    ```bash
    docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
    ```

- **For non-NVIDIA GPU users:**
  - Proceed to the next step.

### 2. Clone the Repository

Clone the **cord19plus** repository:

```bash
git clone https://github.com/your-username/cord19plus.git
```

### 3. Navigate to the Project Directory

```bash
cd cord19plus
```
### 4. (Optional) Replace Docker Files for Non-NVIDIA GPU Systems
If you do not have an NVIDIA GPU, replace the ```Dockerfile``` and ```docker-compose.yml``` with the versions suited for non-GPU systems:

```bash
cp Docker_NO_GPU/Dockerfile .
cp Docker_NO_GPU/docker-compose.yml .
```
Important: Do not commit these changes to the repository. To prevent accidental commits, add these files to your local .gitignore:

```bash
echo "Dockerfile" >> .gitignore
echo "docker-compose.yml" >> .gitignore
```
### 5. Build the Docker Image
Build the Docker image named ```cord19plus```:
```bash
docker build -t cord19plus .
```
### 6. Start the Docker Container
Run the container using Docker Compose:
```bash
docker compose up
```

### 7. Attach Your IDE to the Container

Open your preferred IDE (recommended: Visual Studio Code or Cursor) and attach it to the running Docker container.

### 8. Navigate to the Workspace Directory

Within your IDE, navigate to the ```/workspace``` directory inside the container.

### 9. Select Python Kernel

Set the Python interpreter to Python 3.10.12

Verification

To confirm that your setup is correct:

1. Open the Jupyter notebook ```/gen_docling_exports.ipynb```/ located in the ```/workspace``` directory.
2. Run all cells in the notebook:
3. Ensure that all cells execute without errors.


