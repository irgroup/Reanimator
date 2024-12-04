FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils

# Update and install dependencies
RUN apt-get update && \
    apt-get install -y \
    openjdk-17-jdk \
    openjdk-17-jre \
    git \
    python3 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure pip is up-to-date
RUN python3 -m pip install --upgrade pip

# Set the working directory
WORKDIR /workspace
ADD requirements.txt /workspace

# Install Python dependencies
RUN pip install -r requirements.txt
