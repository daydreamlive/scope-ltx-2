# scope-ltx-2

[![Available on Daydream](https://img.shields.io/badge/Daydream-Install_Node-FF6B35)](https://app.daydream.live/nodes/daydreamlive/ltx-2)

[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/mnfGR4Fjhp)

[Scope](https://github.com/daydreamlive/scope) plugin providing the LTX 2.3 audio-video generation pipeline from [Lightricks](https://github.com/Lightricks/LTX-2).

LTX 2.3 is a 22B-parameter DiT (Diffusion Transformer) that generates synchronized video and audio from text prompts. This plugin uses ComfyUI-derived model loading and inference code with [Kijai's separated FP8 checkpoints](https://huggingface.co/Kijai/LTX2.3_comfy), enabling it to run on a single **24GB GPU**.

## Features

- **Audio–video generation** — synchronized video and audio from text prompts
- **Modes** — **text** (default) and **video**; video mode supports guide conditioning using a reference video (e.g. depth / canny / pose / colorization) on the graph **video** input
- **Image-to-video (I2V)** — optional reference image (`i2v_image`) with adjustable **I2V strength** (`i2v_strength`) to condition the first frame; set strength to `0` for pure text-to-video
- **LoRA** — **permanent merge at load** into the FP8 transformer (dequantize → merge → requantize); zero runtime LoRA overhead; compatible with block streaming. LoRA files go under your Scope models directory (e.g. `models/lora/`) and are selected via the pipeline **LoRA** UI or `loras` in `/load`
- **IC-LoRA support** — IC-LoRAs (In-Context LoRAs) are loaded like any other LoRA. Switch to **video mode** and provide reference frames to activate guide conditioning. `reference_downscale_factor` is read automatically from safetensors metadata. Compatible IC-LoRAs include [Union Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control), [Motion Track](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control), [Colorizer](https://huggingface.co/DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer), [Cameraman](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v1), [Outpaint](https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint), and others. Use **control strength** (`control_strength`) to blend the guide
- **Sampling** — default **8-step distilled** Euler schedule; optional schedules: `linear`, `cosine`, `linear_quadratic`, `beta`; configurable **denoising steps** (`num_steps`). Advanced: custom **`sigmas`** list (API) overrides step count and schedule
- **Output constraints** — height/width snapped to **32**-pixel multiples; frame count snapped to **8×K+1** (minimum 9)
- **Runs on 24GB GPUs** — FP8 weights in checkpoints, CPU-resident transformer blocks with double-buffered streaming to GPU
- **Configurable output** — resolution, frame count, frame rate, seed / **randomize seed** per chunk, **FFN chunk size** for memory tuning

## Requirements

- **VRAM**: ~22GB (24GB GPU recommended, e.g. RTX 4090 / A5000)
  - Gemma 3 12B FP8 text encoder: ~13GB (offloaded after encoding)
  - Transformer 22B FP8: ~23GB total, CPU-resident with block streaming
  - Video VAE + Audio VAE + vocoder: ~1GB (GPU-resident)
- **Python**: 3.12+
- **CUDA**: 12.8+

### 32GB GPU (RTX 5090) mode

With `text_encoder_quant="int4"` (load param) the Gemma checkpoint is
re-quantized from FP8 to int4 via torchao (tinygemm, group_size=128),
dropping its footprint from ~13 GB to ~7.5 GB.  That saves enough room
for Gemma to stay **resident** alongside the transformer + VAE — so a
prompt change no longer triggers the ~40 GB CPU↔GPU swap that the FP8
flow uses to free VRAM for the text encoder.  Peak generation VRAM on a
32GB card is ~30 GiB; prompt change goes from multi-second reloads to
essentially free.  First-load adds ~20s of one-time FP8→bf16 dequant on
the CPU and needs ~25 GB of system RAM briefly.  Recommended on Blackwell
(RTX 5090 and up).

```bash
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{"pipeline_id": "ltx2", "params": {"text_encoder_quant": "int4"}}'
```

The matching `transformer_quant="int4"` option is also implemented (peak
drops to ~25 GiB with zero block-swap during VAE decode), but torchao's
int4 tinygemm kernel is not yet tensor-core accelerated on Blackwell and
runs the denoiser ~10× slower than the FP8 path; prefer it only when
strict zero-swap matters more than throughput.

## Supported Models

Weights are pulled from these Hugging Face repositories:

| Repository | Contents |
|------------|----------|
| [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy) | Transformer (22B distilled v3 FP8), text projection, video VAE, audio VAE (includes vocoder weights used at decode) |
| [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2) | Gemma 3 12B FP8 text encoder (includes embedded SentencePiece tokenizer) |

The Gemma model architecture config is bundled with this plugin — no separate download from `google/gemma-3-12b-it` is needed. The tokenizer is extracted at runtime from the FP8 checkpoint's embedded `spiece_model` tensor.

## Install

### Desktop App

1. Open **Settings** → **Plugins** tab.
2. Paste the following into the installation input field:

   ```
   https://github.com/daydreamlive/scope-ltx-2
   ```

3. Click **Install** and wait for the server to restart.

The plugin and its `ltx2` pipeline will appear automatically once the restart completes. See the [Plugins guide](https://github.com/daydreamlive/scope/blob/main/docs/plugins.md#using-plugins-from-the-desktop-app) for more detail.

### Manual Installation (CLI)

Follow the [manual installation](https://github.com/daydreamlive/scope/tree/main?tab=readme-ov-file#manual-installation) instructions for Scope, then install the plugin from the `scope` directory:

```
uv run daydream-scope install https://github.com/daydreamlive/scope-ltx-2
```

Confirm the plugin is installed:

```
uv run daydream-scope plugins
```

Confirm the `ltx2` pipeline is available:

```
uv run daydream-scope pipelines
```

## Upgrade

### Desktop App

Open **Settings** → **Plugins** tab. If an update is available, an **Update** button appears next to the plugin — click it and wait for the server to restart.

### CLI

```
uv run daydream-scope install --upgrade https://github.com/daydreamlive/scope-ltx-2
```

## Uninstall

### Desktop App

Open **Settings** → **Plugins** tab and click the trash icon next to the plugin.

### CLI

```
uv run daydream-scope uninstall scope-ltx-2
```

## Usage

### Step 1: Configure HuggingFace Token

Create a HuggingFace access token with **read** permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then set:

**Windows Command Prompt:**

```cmd
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Windows PowerShell:**

```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Unix/Linux/macOS:**

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> [!TIP]
> Add the export to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist the token.

### Step 2: Run Scope

```bash
uv run daydream-scope
```

The web UI defaults to `http://localhost:8000`. Select pipeline **`ltx2`** in Settings, or load it via the [API](https://github.com/daydreamlive/scope/blob/main/docs/server.md) ([load pipeline](https://github.com/daydreamlive/scope/blob/main/docs/api/load.md#load-a-pipeline)).

Prefetch weights without the UI:

```bash
uv run download_models --pipeline ltx2
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `height` / `width` | 384 / 320 | Output size in pixels (snapped to multiples of 32) |
| `base_seed` | 42 | Base seed when not randomizing |
| `randomize_seed` | true | New random seed each inference chunk |
| `num_frames` | 129 | Frame count (snapped to 8×K+1; e.g. 9, 17, …, 129) |
| `num_steps` | 8 | Euler denoising steps (1–20) |
| `schedule` | `"distilled"` | Sigma schedule: `distilled`, `linear`, `cosine`, `linear_quadratic`, `beta` |
| `frame_rate` | 24.0 | Metadata / output frame rate |
| `lora_merge_strategy` | `"permanent_merge"` | Only `permanent_merge` is supported for this FP8 model |
| `i2v_image` | — | Optional path or asset for image-to-video first-frame conditioning |
| `i2v_strength` | 1.0 | 0 = no I2V conditioning, 1 = full first-frame conditioning |
| `control_strength` | 1.0 | Video mode: guide conditioning strength (0 = off, 1 = full) |
| `ffn_chunk_size` | 4096 | FFN chunking for memory (smaller = less VRAM, more overhead; `null` disables) |
| `sigmas` | — | **API:** custom descending sigma list; overrides `num_steps` and `schedule` |

**LoRAs:** Pass `loras` as a list of `{ "path": "...", "scale": 1.0 }` in the `/load` body (paths are typically under your models tree). IC-LoRAs are treated as regular LoRAs — add them to `loras` and switch to video mode to activate guide conditioning.

> [!NOTE]
> Which parameters appear in the Scope UI depends on your Scope version. Anything exposed in the pipeline JSON schema can be set via `/load` if needed.

#### Example `/load` body

```bash
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_id": "ltx2",
    "params": {
      "height": 384,
      "width": 320,
      "num_frames": 129,
      "num_steps": 8,
      "schedule": "distilled",
      "randomize_seed": true,
      "frame_rate": 24.0,
      "ffn_chunk_size": 4096
    }
  }'
```

#### Frame count

Valid counts follow **8×K+1** (minimum 9): 9, 17, 25, 33, … Other values are snapped to the nearest valid count.

### Memory optimization

1. Lower resolution and/or **fewer frames**
2. Lower **`ffn_chunk_size`** (e.g. 2048 or 1024)
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** to reduce fragmentation

## Limitations

- **No frame-by-frame streaming** — each run produces a full clip; latency is batch generation time, not interactive streaming
- **LoRA strategy** — **permanent merge only** for FP8 (no runtime PEFT path for this pipeline)
- **Quantization** — transformer FP8 is fixed by the published checkpoint; there is no separate “pick your dtype” mode in-plugin

Exact Scope UI coverage for every parameter can lag the schema; use **`/load`** when a control is not in the UI yet.

## Architecture

- **Gemma 3 12B FP8** text encoder, aggregate embedding projection
- **22B** transformer for joint audio–video denoising (FP8 scaled matmul where applicable)
- **Video VAE** — 32× spatial, 8× temporal downsampling
- **Audio VAE + vocoder** — mel decode to waveform (aligned with ComfyUI-style audio stack)
- **Euler sampling** with configurable sigma schedules; default distilled 8-step schedule when `num_steps` is 8 and `schedule` is `distilled`
- **CPU→GPU block streaming** — transformer blocks in pinned host memory, double-buffered async copies during denoising

Details: [Lightricks LTX-2](https://github.com/Lightricks/LTX-2).

## Troubleshooting

### Out of memory (OOM)

1. Reduce `num_frames` (e.g. 33 instead of 129)
2. Reduce resolution
3. Reduce `ffn_chunk_size`
4. Close other GPU workloads

### Model download fails

**Invalid token:** set `HF_TOKEN` correctly; token needs at least read access.

**Repository not found:** confirm HF account email verification if required.

**General:** check network; `huggingface-cli login` to verify the token.

### Slow generation

Generation time scales with frames, resolution, and GPU/PCIe throughput (weight streaming).

## License

This plugin is licensed under the same terms as the [LTX-2 model](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for LTX-2
- [Kijai](https://huggingface.co/Kijai) for separated Comfy-format FP8 checkpoints
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for patterns this plugin adapts
- [Daydream](https://daydream.live/) for Scope
