# Hunyuan3D-2.1 -> RunPod serverless image.
# ponytail: -devel base needed for nvcc to compile the rasterizer. If this tag
# 404s on Docker Hub, swap to nvidia/cuda:12.4.1-devel-ubuntu22.04 + install torch below.
FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/hf \
    PYTHONUNBUFFERED=1 \
    # Compile CUDA ext for the target GPUs so build needs no local GPU.
    # 8.6=A6000/A40/A5000, 8.9=L40/L40S/4090, 8.0=A100. Trim to what you'll deploy on.
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 .

# torch is pinned by the base image; requirements + runpod on top.
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir runpod

# Native extensions (the #1 build failure point). Paths per 2.1 README.
RUN cd hy3dpaint/custom_rasterizer && pip install -e . && cd /app \
 && cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && cd /app

# Real-ESRGAN weight the paint pipeline expects.
RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -P hy3dpaint/ckpt

COPY handler.py /app/handler.py
CMD ["python", "-u", "handler.py"]
