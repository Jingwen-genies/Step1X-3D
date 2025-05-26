# Use NVIDIA CUDA development image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch with CUDA 12.4 support
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install torch-cluster for CUDA 12.4
RUN pip3 install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Copy the application code
COPY . .

# Install custom rasterizer components
RUN cd step1x3d_texture/custom_rasterizer && \
    python3 setup.py install && \
    cd ../differentiable_renderer && \
    python3 setup.py install && \
    cd ../../

# Create necessary directories
RUN mkdir -p model_cache temp output data/shape_autoencoder

# Expose the port the app runs on
EXPOSE 8001

# Set the entrypoint
ENTRYPOINT ["python3", "app-api.py"]
