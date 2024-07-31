# Use the official Debian Bullseye image
FROM debian:bullseye-slim

# Set environment variables to prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install \
    torch \
    transformers \
    datasets \
    tqdm

# Set the working directory
WORKDIR /workspace

# Command to keep the container running and interactive
CMD ["/bin/bash"]
