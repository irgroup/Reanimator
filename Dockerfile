FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update and install dependencies
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    openjdk-17-jdk \
    openjdk-17-jre \
    python3 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure pip is up-to-date
RUN python3 -m pip install --upgrade pip

# Set the working directory
WORKDIR /workspace
ADD . /workspace

# Install Python dependencies
RUN pip install -r requirements.txt
