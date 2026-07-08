# Persistence layer on top of the Local Lab Hunyuan3D 2.1 ComfyUI template.
# Points ComfyUI's models + output dirs at the RunPod network volume (/workspace)
# so downloaded models and generated .glb survive Stop/Start.
# The base image's entrypoint/ports are inherited unchanged.
FROM thelocallab/hunyuan3d-2.1-comfyui:latest

# huggingface_hub cache also lands on the volume.
ENV HF_HOME=/workspace/hf

# Redirect model + output dirs to the volume via symlinks (baked at build time).
# ponytail: first boot downloads models onto the empty volume once; every boot
# after reuses them. A network volume always shadows image content at its mount,
# so one initial download is unavoidable — the point is it never repeats.
RUN rm -rf /app/ComfyUI/models /app/ComfyUI/output \
 && ln -s /workspace/models /app/ComfyUI/models \
 && ln -s /workspace/output /app/ComfyUI/output
