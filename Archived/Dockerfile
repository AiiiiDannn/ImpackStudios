# Base Image: Includes CUDA 12.1, ready for GPU fine-tuning
# (On Mac, the CUDA parts are ignored, and it just uses the CPU)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Key: Cache all Hugging Face models to /workspace/hf_cache
    # You can volume-mount this directory later to avoid re-downloading
    HF_HOME=/workspace/hf_cache

# 1. Install system dependencies
# - python3.10 & pip: The basics
# - git & git-lfs: For pulling code and large models
# - tini: A lightweight "init" process to correctly manage signals
# (Note: We are NOT installing openssh-server, as VS Code Dev Containers don't need it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    git-lfs \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# 2. Set the working directory
WORKDIR /workspace

# 3. Copy requirements.txt
# (Docker caches this layer. As long as requirements.txt doesn't change,
# this step won't re-run)
COPY requirements.txt .

# 4. Install all Python dependencies
# Note: As your professor said, we are in a container, so we can "live...
# First, upgrade pip itself, which often fixes resolver bugs
RUN pip install --upgrade pip
# Now, install all the packages from requirements.txt
RUN pip install -r requirements.txt

# 5. Copy the rest of your project files (.py, .ipynb, etc.)
COPY . .

# 6. (Optional) Expose ports for VS Code to forward
# Expose Gradio default port
EXPOSE 7860
# Expose API port (if needed)
EXPOSE 8000

# 7. Use tini as the entrypoint for safe process management
ENTRYPOINT ["/usr/bin/tini", "--"]

# 8. Default command: Keep the container alive
# This command does nothing but "sleep",
# allowing the container to run so VS Code can "attach" to it.
CMD ["tail", "-f", "/dev/null"]