FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg wget unzip \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]