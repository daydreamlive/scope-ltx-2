# scope-ltx-2

[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/mnfGR4Fjhp)

[Scope](https://github.com/daydreamlive/scope) plugin providing the LTX 2.3 audio-video generation pipeline from [Lightricks](https://github.com/Lightricks/LTX-2).

LTX 2.3 is a 22B-parameter DiT (Diffusion Transformer) that generates synchronized video and audio from text prompts. This plugin uses ComfyUI-derived model loading and inference code with [Kijai's separated FP8 checkpoints](https://huggingface.co/Kijai/LTX2.3_comfy), enabling it to run on a single **24GB GPU**.

> [!IMPORTANT]
> Plugin support is a preview feature in Scope right now and the APIs are subject to breaking change prior to official release.
> Be sure to be running v0.1.0-beta.3+

## Features

- **Audio-video generation** from text prompts (synchronized audio + video)
- **Non-autoregressive** - generates complete clips (33+ frames) at once
- **Runs on 24GB GPUs** - FP8 quantization baked into checkpoints, CPU-resident transformer with double-buffered weight streaming
- **8-step distilled inference** - fast generation with predefined sigma schedule
- **Configurable output** - resolution, frame count, frame rate, seed

## Requirements

- **VRAM**: ~22GB (24GB GPU recommended, e.g. RTX 4090 / A5000)
  - Gemma 3 12B FP8 text encoder: ~13GB (offloaded after encoding)
  - Transformer 22B FP8: ~23GB total, CPU-resident with block streaming
  - Video VAE + Audio VAE + Vocoder: ~1GB (GPU-resident)
- **Python**: 3.12+
- **CUDA**: 12.8+

## Supported Models

This plugin downloads weights from three HuggingFace repositories:

| Repository | Contents |
|------------|----------|
| [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy) | Transformer (22B distilled v3 FP8), text projection, video VAE, audio VAE |
| [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2) | Gemma 3 12B FP8 text encoder |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | Gemma tokenizer and config files |

## Install

Follow the [manual installation](https://github.com/daydreamlive/scope/tree/main?tab=readme-ov-file#manual-installation) instructions for Scope (plugin support for the desktop app is not available yet).

Install the plugin within the `scope` directory:

```bash
DAYDREAM_SCOPE_PREVIEW=1 uv run daydream-scope install git+https://github.com/daydreamlive/scope-ltx-2.git
```

Confirm that the plugin is installed:

```bash
DAYDREAM_SCOPE_PREVIEW=1 uv run daydream-scope plugins
```

Confirm that the `ltx2` pipeline is available:

```bash
DAYDREAM_SCOPE_PREVIEW=1 uv run daydream-scope pipelines
```

## Upgrade

Upgrade the plugin to the latest version:

```bash
DAYDREAM_SCOPE_PREVIEW=1 uv run daydream-scope install --upgrade git+https://github.com/daydreamlive/scope-ltx-2.git
```

## Usage

### Step 1: Accept Gemma License on HuggingFace

The Gemma tokenizer files are hosted in a **gated** repository. You must accept the license to download them.

> [!IMPORTANT]
> You must complete these steps while **logged in** to HuggingFace, or downloads will fail with "403 Forbidden" errors.

#### Google Gemma 3 12B

1. Go to [huggingface.co/google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)
2. Click **"Agree and access repository"** to accept Google's Gemma Terms of Use
3. You may need to:
   - Verify your HuggingFace account email
   - Provide your name and affiliation
   - Acknowledge the usage restrictions
4. Wait for access approval (usually instant, but can take a few minutes)

> [!NOTE]
> The other two repositories ([Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy) and [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2)) are not gated and do not require license acceptance.

> [!NOTE]
> Gemma models are subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Make sure you comply with the terms for your use case.

### Step 2: Configure HuggingFace Token

Create a HuggingFace access token with **read** permissions:

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"Create new token"**
3. Give it a name (e.g., "scope-ltx2")
4. Select **"Read"** access (minimum required)
5. Click **"Create token"** and copy the token

Set the token as an environment variable:

**Windows Command Prompt:**

```cmd
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Windows Powershell:**

```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Unix/Linux/macOS:**

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> [!TIP]
> Add the export command to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist the token across sessions.

### Step 3: Run Scope

Start the server:

```bash
uv run daydream-scope
```

The web frontend will be available at `http://localhost:8000` by default.

The `ltx2` pipeline will be available in:
- The frontend under the Pipeline ID dropdown in the Settings panel.
- The [API](https://github.com/daydreamlive/scope/blob/main/docs/server.md) by [loading the pipeline](https://github.com/daydreamlive/scope/blob/main/docs/api/load.md#load-a-pipeline) using the `ltx2` pipeline ID.

If you are not using the frontend, you can download the model weights first:

```bash
uv run download_models --pipeline ltx2
```

### Configuration Options

| Parameter | Default | UI Available | Description |
|-----------|---------|--------------|-------------|
| `height` | 512 | ✅ Yes | Output video height in pixels |
| `width` | 768 | ✅ Yes | Output video width in pixels |
| `base_seed` | 42 | ✅ Yes | Random seed for reproducible generation |
| `num_frames` | 33 | ⚠️ API only | Number of frames to generate (~1.3 seconds at 24fps) |
| `frame_rate` | 24.0 | ⚠️ API only | Output frame rate |
| `randomize_seed` | false | ⚠️ API only | Generate new random seed each chunk |
| `ffn_chunk_size` | 4096 | ⚠️ API only | Chunk size for FFN processing (lower = less memory, more overhead; `null` to disable) |

> [!NOTE]
> Parameters marked "API only" work but don't have UI controls in Scope's main branch.
> Use the `/load` API endpoint to set these parameters.

#### Using API-Only Parameters

```bash
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_id": "ltx2",
    "params": {
      "height": 512,
      "width": 768,
      "num_frames": 49,
      "randomize_seed": true
    }
  }'
```

#### Frame Count Formula

LTX 2.3 works best with frame counts following the formula `8×K+1`:
- 9, 17, 25, **33**, 41, 49, 57, 65, ...

Other values are automatically snapped to the nearest valid count.

### Memory Optimization Tips

1. **Lower resolution** - activations scale with resolution × frames
2. **Fewer frames** - reduces both latent size and activation memory
3. **Reduce `ffn_chunk_size`** - smaller chunks use less peak activation memory at the cost of more kernel launches
4. **Set PYTORCH_CUDA_ALLOC_CONF** - `expandable_segments:True` prevents fragmentation

## Limitations

### Current Limitations

- **No real-time streaming** - LTX 2.3 generates complete clips, not frame-by-frame
- **No image conditioning** - text-to-video only (image conditioning planned)
- **No LoRA support** - planned for future release

### Scope Main Branch Limitations

These features work in the plugin but lack UI controls in Scope's main branch:

| Feature | Status | Workaround |
|---------|--------|------------|
| `num_frames` slider | No UI | Use API to set value |
| `randomize_seed` toggle | No UI | Use API to set value |
| `ffn_chunk_size` control | No UI | Use API to set value |
| Fixed FPS playback | Partial | Video generated correctly, playback may vary |

## Architecture

LTX 2.3 uses a DiT (Diffusion Transformer) architecture with standalone ComfyUI-adapted inference code:

- **Gemma 3 12B FP8** text encoder for prompt understanding, with per-token RMS normalization across all 49 layers and aggregate embedding projection
- **22B parameter** transformer for joint audio-video denoising, loaded with FP8 scaled matmul (`torch._scaled_mm`) for quantized layers
- **Video VAE** for latent space decoding (32x spatial, 8x temporal downsampling)
- **Audio VAE + Vocoder** for synchronized audio generation (mel spectrogram decoding with bandwidth extension)
- **8-step distilled Euler sampling** with predefined sigma schedule
- **CPU-to-GPU weight streaming** - transformer blocks are CPU-resident in pinned memory and double-buffered onto GPU via async CUDA streams during inference, allowing the full 22B model to run within 24GB VRAM

For more details, see the [official LTX-2 documentation](https://github.com/Lightricks/LTX-2).

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
1. Reduce `num_frames` (try 17 instead of 33)
2. Reduce resolution (try 384×512)
3. Reduce `ffn_chunk_size` (try 2048 or 1024)
4. Ensure no other GPU processes are running

### Model Download Fails

**"403 Forbidden" or "Access to model is restricted":**
- You haven't accepted the Gemma license agreement on HuggingFace
- Go to [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) while logged in and click "Agree and access repository"

**"Invalid token" or "Token not found":**
- Verify `HF_TOKEN` environment variable is set correctly
- Regenerate your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Ensure the token has at least "Read" permissions

**"Repository not found" for Gemma:**
- Gemma access may require email verification on HuggingFace
- Check your email for a verification link from HuggingFace
- Some Gemma models require manual approval (usually instant)

**General tips:**
- Check network connectivity
- Try `huggingface-cli login` to verify your token works
- Ensure you're using the correct model names (case-sensitive)

### Slow Generation

LTX 2.3 generates complete clips rather than streaming frames. Generation time depends on:
- Number of frames
- Resolution
- GPU performance and CPU-to-GPU transfer bandwidth (weight streaming)

## License

This plugin is licensed under the same terms as the [LTX-2 model](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for the LTX-2 model
- [Kijai](https://huggingface.co/Kijai) for the separated ComfyUI-format FP8 checkpoints
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the model loading and VAE implementations this plugin adapts
- [Daydream](https://daydreamlive.ai/) for the Scope platform
