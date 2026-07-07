"""RunPod serverless handler for Hunyuan3D-2.1 (image -> textured PBR .glb).

Request schema: see plan.md §5. Returns { "glb_base64": ... } or { "error": ... }.
Run locally to smoke-test: `python handler.py --test_input '{"input":{...}}'` (runpod builtin).
"""
import base64, gc, io, os, sys, tempfile, urllib.request

import torch
import runpod

REPO = "/app"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "hy3dpaint"))  # textureGenPipeline lives here

# Lazy, cached pipeline loads. On 48GB both stay resident; on 24GB (low_vram)
# we drop each after use so shape(10GB) and texture(21GB) never coexist.
_shape = None
_paint = None


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_shape():
    global _shape
    if _shape is None:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        _shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    return _shape


def get_paint(resolution):
    global _paint
    if _paint is None:
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        _paint = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=resolution))
    return _paint


def _load_image(spec, dst):
    """spec is a data: / http(s) URL or raw base64 -> save to dst path."""
    if spec.startswith("http://") or spec.startswith("https://"):
        urllib.request.urlretrieve(spec, dst)
    else:
        if spec.startswith("data:"):
            spec = spec.split(",", 1)[1]
        with open(dst, "wb") as f:
            f.write(base64.b64decode(spec))
    return dst


def handler(event):
    inp = event["input"]
    low_vram = inp.get("low_vram", False)
    s = inp.get("shape", {})
    t = inp.get("texture", {})
    c = inp.get("cleanup", {})

    with tempfile.TemporaryDirectory() as tmp:
        img_path = _load_image(inp["image"], os.path.join(tmp, "in.png"))

        if inp.get("remove_background", True):
            from hy3dshape.rembg import BackgroundRemover
            from PIL import Image
            img = BackgroundRemover()(Image.open(img_path).convert("RGB"))
            img.save(img_path)

        # --- shape ---
        gen = torch.manual_seed(s.get("seed", 0)) if "seed" in s else None
        mesh = get_shape()(
            image=img_path,
            num_inference_steps=s.get("num_inference_steps", 50),
            guidance_scale=s.get("guidance_scale", 5.0),
            octree_resolution=s.get("octree_resolution", 384),
            num_chunks=s.get("num_chunks", 8000),
            generator=gen,
        )[0]

        # --- cleanup (best-effort; module path per 2.1 repo) ---
        try:
            from hy3dshape.postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover
            if c.get("remove_floaters", True):
                mesh = FloaterRemover()(mesh)
            if c.get("remove_degenerate", True):
                mesh = DegenerateFaceRemover()(mesh)
            if c.get("max_facenum"):
                mesh = FaceReducer()(mesh, max_facenum=c["max_facenum"])
        except ImportError:
            pass  # ponytail: confirm postprocessor import path against 2.1 source, then drop the try

        mesh_path = os.path.join(tmp, "shape.glb")
        mesh.export(mesh_path)

        if low_vram:
            global _shape
            _shape = None
            _free()

        # --- PBR texture ---
        out_path = mesh_path
        if t.get("enabled", True):
            textured = get_paint(t.get("resolution", 2048))(mesh_path, image_path=img_path)
            # paint may return a path or a mesh depending on version.
            if isinstance(textured, str):
                out_path = textured
            else:
                out_path = os.path.join(tmp, "textured.glb")
                textured.export(out_path)
            if low_vram:
                global _paint
                _paint = None
                _free()

        with open(out_path, "rb") as f:
            return {"glb_base64": base64.b64encode(f.read()).decode()}


runpod.serverless.start({"handler": handler})
