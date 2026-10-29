# Preserve separate numerical runtimes used for submission19433.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
USER root
COPY requirements.txt requirements-sam.txt /opt/eurs-build/
RUN python -m pip install --no-cache-dir -r /opt/eurs-build/requirements.txt numpy==2.2.6 pillow==11.3.0 scipy==1.15.3 pyyaml==6.0.3 timm==1.0.20 einops==0.8.1 && \
    python -m venv /opt/eurs-sam && \
    /opt/eurs-sam/bin/python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 && \
    /opt/eurs-sam/bin/python -m pip install --no-cache-dir -r /opt/eurs-build/requirements-sam.txt
ENV EURS_SAM_PYTHON=/opt/eurs-sam/bin/python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/eurs-hf \
    TORCH_HOME=/tmp/eurs-torch \
    XDG_CACHE_HOME=/tmp/eurs-cache \
    TRITON_CACHE_DIR=/tmp/eurs-triton \
    CUDA_CACHE_PATH=/tmp/eurs-cuda \
    TORCH_EXTENSIONS_DIR=/tmp/eurs-extensions \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
COPY . /pkg
WORKDIR /pkg
RUN chmod -R a+rX /pkg && mkdir -p /pkg/weights
USER 10001:10001
ENTRYPOINT []
