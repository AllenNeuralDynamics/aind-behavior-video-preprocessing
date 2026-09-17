# hash:sha256:f021ab0082d32d2954b63da31e988d47a581ca94879eb15b3627e51ed45c93a3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

# System deps: ffmpeg is the lossless encoder (libx264 -qp 0);
# libsm6/libxext6 are OpenCV runtime libs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python deps: headless OpenCV for frame transforms, pytest for the
# parity tests. Deliberately minimal -- no torch/LP/pandas here.
RUN pip install -U --no-cache-dir \
    numpy==2.5.3 \
    opencv-python-headless==5.0.0.93 \
    pytest==9.1.1