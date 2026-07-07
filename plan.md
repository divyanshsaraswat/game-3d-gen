# Plan: 2D → 3D pipeline with Hunyuan3D-2.1 on RunPod

End-to-end: single image (or text) in → textured **PBR** `.glb` out. Runs on a RunPod GPU
pod (dev) or serverless endpoint (production). All tunable parameters exposed.

Targeting **Hunyuan3D-2.1** (`tencent/Hunyuan3D-2.1`) — bigger 3.3B shape DiT + PBR
material paint. Needs a bigger GPU than 2.0; see §3 for the 24 GB staged fallback.

> ponytail: reuse Tencent's `hy3dgen` package and its `api_server.py` instead of
> rewriting a pipeline. We only add: a Dockerfile, a serverless handler, and a thin
> param wrapper. Everything else is upstream.

---

## 1. What the models are

Source: https://huggingface.co/tencent/Hunyuan3D-2.1

Two sub-models, loaded by two pipeline classes (same class names as 2.0 — our plan carries over):

| Model | Role | Pipeline class |
|---|---|---|
| Hunyuan3D-Shape-v2-1 (3.3B) | image → mesh geometry (flow-matching DiT) | `Hunyuan3DDiTFlowMatchingPipeline` |
| Hunyuan3D-Paint-v2-1 (2B) | mesh + image → **PBR materials** (albedo / metallic / roughness) | `Hunyuan3DPaintPipeline` |

**Why 2.1:** PBR output lights correctly in real engines (Unreal/Unity/Blender/three.js)
instead of 2.0's baked-in RGB. Full weights **and training code** released → fine-tunable.

Cost: bigger model, higher VRAM (§3). No mini/turbo variants at 2.1 yet — if VRAM/latency
forces it, fall back to `tencent/Hunyuan3D-2mini` for shape (no PBR there).

---

## 2. Parameters to expose (the whole surface)

### Input / preprocessing
- `image` — path or PIL/base64. Alpha channel used as mask if present.
- `remove_background` (bool) — run `hy3dgen.rembg.BackgroundRemover` first. Required if image has a background.
- `text` (optional) — text→3D: run `HunyuanDiTPipeline` (t2i) → feed result as `image`.

### Shape generation — `Hunyuan3DDiTFlowMatchingPipeline(...)`
- `num_inference_steps` — 30–50. Default 50.
- `guidance_scale` — CFG strength. Default 5.0–7.5.
- `octree_resolution` — surface detail: 256 / 384 / 512. Higher = finer + more VRAM/time.
- `num_chunks` — VRAM control; higher = lower peak memory, slower.
- `mc_algo` — marching cubes algo: `'mc'` (fast) or `'dmc'` (dual MC, cleaner topology).
- `seed` / `generator` — reproducibility.
- `output_type` — `'trimesh'`.

### Mesh cleanup (`from hy3dgen.shapegen import ...`, chain in order)
- `FloaterRemover()` — drop disconnected junk.
- `DegenerateFaceRemover()` — drop zero-area faces.
- `FaceReducer()(mesh, max_facenum=...)` — decimate to a face budget (e.g. 40000) for game/web use.

### Texture generation (PBR) — `Hunyuan3DPaintPipeline(mesh, image=...)`
- Runs de-lighting + multiview diffusion + bake automatically.
- Outputs **PBR channels**: albedo (base color), metallic, roughness.
- Knobs: `num_inference_steps`, `guidance_scale`, texture resolution (1024/2048), `seed`.
- ~21 GB VRAM on its own (§3) — the heavy stage.

### Export
- `.glb` carries PBR channels natively (metallic-roughness) — use it. Also `.obj`+`.mtl`, `.ply`.
- `mesh.export('out.glb')`. Verify metallic/roughness maps survive the export.

All of the above become a single JSON request schema (§5).

---

## 3. RunPod setup

**VRAM budget (2.1):** shape **10 GB**, texture **21 GB**, combined **29 GB**.

**GPU choice:**
- **48 GB (A6000 / L40 / L40S)** — runs shape + PBR texture loaded together, simplest. Recommended.
- **A100 40 GB** — fits combined 29 GB with headroom.
- **24 GB (RTX 4090 / A5000)** — won't hold both at once. Use the **staged fallback**: run
  shape (10 GB) → free the shape pipeline (`del` + `torch.cuda.empty_cache()`) → load & run
  texture (21 GB). Slower per job (reload cost) but works. Set this via a `low_vram` flag.

**Two deploy modes:**
- **Dev / interactive** — GPU Pod from our Docker image, run `gradio_app.py` on port 7860, expose via RunPod proxy. Good for tuning parameters by hand.
- **Production** — RunPod **Serverless** endpoint with a `handler.py`. Pay per second, autoscales to zero.

**Storage:** attach a Network Volume, set `HF_HOME=/runpod-volume/hf` so model weights
(~15 GB for 2.1) download once and persist across cold starts instead of re-downloading.

---

## 4. Build steps (in order)

1. **Dockerfile** — base `runpod/pytorch` (CUDA 12.1). Clone `Tencent-Hunyuan/Hunyuan3D-2.1`,
   `pip install -r requirements.txt`, then **compile the two native extensions** (these are
   the only real gotcha — paths differ slightly in 2.1's repo, confirm on clone):
   - custom rasterizer → `pip install -e .`
   - differentiable renderer → `pip install -e .`
   Also `pip install runpod`.
2. **Pre-cache weights** into the network volume (`huggingface-cli download tencent/Hunyuan3D-2.1`)
   so the first job doesn't time out.
3. **`handler.py`** — RunPod serverless handler. On a 48 GB GPU load both pipelines once at
   module scope; on 24 GB load lazily per stage (staged fallback, §3). Maps the §2 params from
   `event["input"]`, runs shape → cleanup → PBR texture → export, returns the `.glb` as base64
   or an uploaded URL.
4. **`gradio_app.py`** — use upstream's for dev; expose all §2 params as sliders.
5. **Push image**, create the RunPod template + serverless endpoint pointing at the volume.
6. **Test** with `/runsync` for a small job, `/run` (async + polling) for `octree_resolution=512`.

---

## 5. Request schema (serverless contract)

```json
{
  "input": {
    "image": "<base64 or url>",
    "text": null,
    "model": "Hunyuan3D-2.1",
    "low_vram": false,
    "remove_background": true,
    "shape": {
      "num_inference_steps": 50,
      "guidance_scale": 5.0,
      "octree_resolution": 384,
      "num_chunks": 8000,
      "mc_algo": "mc",
      "seed": 1234
    },
    "cleanup": { "remove_floaters": true, "remove_degenerate": true, "max_facenum": 40000 },
    "texture": { "enabled": true, "pbr": true, "num_inference_steps": 30, "resolution": 2048, "seed": 1234 },
    "output_format": "glb"
  }
}
```

---

## 6. Gotchas / notes

- Native extension compile is the #1 failure point — pin CUDA/torch versions in the image, build at image-build time (not per cold-start).
- **29 GB combined** is the headline constraint. On <48 GB use the staged fallback (§3) or you OOM on the texture stage.
- Texture stage dominates VRAM; if OOM: lower texture resolution (2048→1024), raise `num_chunks`, or fall back to 2.0-mini for shape.
- Confirm the `.glb` actually carries metallic/roughness maps — the whole reason for 2.1 is PBR; a silent bake-to-RGB export throws it away.
- Cold start is longer than 2.0 (bigger 3.3B model). Keep min workers ≥1 if latency matters.
- License: `tencent-hunyuan-community` — has usage/commercial restrictions. Read `LICENSE` before shipping.

---

*Skipped:* actual code, retries/queueing, multi-format post-processing UI. Add when the
pod build in §4 is proven working end-to-end.
