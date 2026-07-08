# Fast-boot, persistent build of the Local Lab Hunyuan3D 2.1 ComfyUI template.
# The upstream image runs its whole setup in /app/start.sh at boot (~20 min every
# time). Here we bake that setup into image layers so boots just launch, and
# redirect models/output to the RunPod network volume so they persist.
FROM thelocallab/hunyuan3d-2.1-comfyui:latest

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/hf \
    # A6000 = 8.6. Lets the CUDA extensions compile at build time with no GPU.
    TORCH_CUDA_ARCH_LIST=8.6
WORKDIR /app

# --- CUDA toolkit (needed to compile the rasterizer/renderer) ---
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates libgl1 git \
 && wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get install -y cuda-toolkit-12-4 \
 && rm cuda-keyring_1.1-1_all.deb && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- ComfyUI + custom nodes ---
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git \
 && git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git ComfyUI/custom_nodes/ComfyUI-Manager \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-Hunyuan3DWrapper.git ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper \
 && git clone --depth 1 https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git ComfyUI/custom_nodes/ComfyUI-Hunyuan3d-2-1 \
 && git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials.git ComfyUI/custom_nodes/ComfyUI_essentials \
 && git clone --depth 1 https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git ComfyUI/custom_nodes/Derfuu_ComfyUI_ModdedNodes \
 && git clone --depth 1 https://github.com/WASasquatch/was-node-suite-comfyui.git ComfyUI/custom_nodes/was-node-suite-comfyui \
 && git clone --depth 1 https://github.com/KoreTeknology/ComfyUI-Universal-Styler.git ComfyUI/custom_nodes/ComfyUI-Universal-Styler \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI/custom_nodes/ComfyUI-KJNodes \
 && git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git ComfyUI/custom_nodes/rgthree-comfy

# --- Python deps (torch pinned to match the template: 2.6 / cu124) ---
WORKDIR /app/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir opencv_python==4.10.0.82 sageattention \
 && pip install --no-cache-dir torch==2.6.* torchvision==0.21.* torchaudio==2.6.* --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir -r ./custom_nodes/ComfyUI-Manager/requirements.txt \
 && pip install --no-cache-dir -r ./custom_nodes/ComfyUI-Hunyuan3DWrapper/requirements.txt \
 && pip install --no-cache-dir -r ./custom_nodes/ComfyUI-Hunyuan3d-2-1/requirements.txt \
 && pip install --no-cache-dir -r ./custom_nodes/ComfyUI_essentials/requirements.txt \
 && pip install --no-cache-dir -r ./custom_nodes/was-node-suite-comfyui/requirements.txt \
 && pip install --no-cache-dir -r ./custom_nodes/ComfyUI-KJNodes/requirements.txt \
 && pip install --no-cache-dir -r ./custom_nodes/rgthree-comfy/requirements.txt \
 && pip install --no-cache-dir onnxruntime==1.16.3 jupyterlab

# --- compile the CUDA extensions (no GPU; arch pinned above) ---
ENV CUDA_HOME=/etc/alternatives/cuda \
    PATH=/etc/alternatives/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/etc/alternatives/cuda/lib64
RUN cd custom_nodes/ComfyUI-Hunyuan3d-2-1/hy3dpaint/custom_rasterizer && python setup.py install \
 && cd ../DifferentiableRenderer && python setup.py install

# --- persist models + outputs on the network volume (dirs exist now) ---
RUN rm -rf /app/ComfyUI/models /app/ComfyUI/output \
 && ln -s /workspace/models /app/ComfyUI/models \
 && ln -s /workspace/output /app/ComfyUI/output

# --- slim launcher replaces the heavy boot script ---
COPY start.sh /app/start.sh
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

WORKDIR /app
CMD ["/app/start.sh"]
