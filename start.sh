#!/bin/sh
# Slim launcher: everything heavy (CUDA, ComfyUI, nodes, pip, compiled
# extensions) is already baked into the image. Here we only ensure models
# exist on the persistent volume, then launch. First boot downloads models
# once to /workspace; every boot after finds them and starts instantly.
set -e
cd /app/ComfyUI

mkdir -p /workspace/models/diffusion_models \
         /workspace/models/vae \
         /workspace/models/upscale_models \
         /workspace/output

DIT=/workspace/models/diffusion_models/hunyuan3d-dit-v2-1.fp16.ckpt
VAE=/workspace/models/vae/hunyuan3d-vae-v2-1.ckpt
UP=/workspace/models/upscale_models/4x_foolhardy_Remacri.pth

[ -f "$DIT" ] || wget -O "$DIT" "https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/hunyuan3d-dit-v2-1/model.fp16.ckpt?download=true"
[ -f "$VAE" ] || wget -O "$VAE" "https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/hunyuan3d-vae-v2-1/model.fp16.ckpt?download=true"
if [ ! -f "$UP" ]; then
  wget -O /workspace/models/upscale_models/1x-SwatKatsLite.pth "https://huggingface.co/Thelocallab/1x-SwatKatsLite/resolve/main/1x-SwatKatsLite.pth?download=true"
  wget -O /workspace/models/upscale_models/2xLexicaRRDBNet_Sharp.pth "https://huggingface.co/Thelocallab/2xLexicaRRDBNet_Sharp/resolve/main/2xLexicaRRDBNet_Sharp.pth?download=true"
  wget -O "$UP" "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth?download=true"
fi

export CUDA_HOME=/etc/alternatives/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --NotebookApp.token= --NotebookApp.password= --NotebookApp.allow_origin='*' \
  > /app/jupyter.log 2>&1 &

python main.py --listen
