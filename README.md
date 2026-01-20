# scope-ltx-2

[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/mnfGR4Fjhp)

[Scope](https://github.com/daydreamlive/scope) plugin providing the LTX-2 text-to-video generation pipeline from [Lightricks](https://github.com/Lightricks/LTX-2).

LTX-2 is a DiT-based foundation model for high-quality video generation. Unlike autoregressive models, LTX-2 generates complete video clips in a single forward pass.

> [!IMPORTANT]
> Plugin support is a preview feature in Scope right now and the APIs are subject to breaking change prior to official release.
> Be sure to be running v0.1.0-beta.3+

## Features

- **High-quality video generation** from text prompts
- **Non-autoregressive** - generates complete clips (33+ frames) at once
- **FP8 quantization** support for reduced VRAM usage
- **Configurable output** - resolution, frame count, frame rate, seed

### Audio Support (Stubbed)

LTX-2 natively supports synchronized audio-video generation. However, **audio output is currently disabled** because Scope's main branch doesn't support audio channels via WebRTC yet.

Audio latents are generated during inference but not decoded. When Scope adds audio channel support, this plugin can be updated to enable full audio output. See [PLUGIN.md](PLUGIN.md) for technical details.

## Requirements

- **VRAM**: ~96GB minimum (H100 recommended)
  - Text encoder (Gemma 3 12B): ~20GB
  - Transformer (FP8): ~25GB
  - Video decoder: ~3GB
  - Activations: ~50GB at 512×768×33 frames
- **Python**: 3.12+
- **CUDA**: 12.8+

## Supported Models

- [LTX-2 19B Distilled](https://huggingface.co/Lightricks/LTX-2) (via the `ltx2` pipeline)

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

### Step 1: Accept Model Licenses on HuggingFace

Both models used by this plugin are **gated** on HuggingFace, meaning you must accept their license agreements before downloading.

> [!IMPORTANT]
> You must complete these steps while **logged in** to HuggingFace, or downloads will fail with "403 Forbidden" errors.

#### Lightricks/LTX-2

1. Go to [huggingface.co/Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2)
2. Click **"Agree and access repository"** to accept the license
3. You should see "You have been granted access" confirmation

#### Google Gemma 3 12B (Required for Text Encoder)

The Gemma model requires additional steps:

1. Go to [huggingface.co/google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)
2. Click **"Agree and access repository"** to accept Google's Gemma Terms of Use
3. You may need to:
   - Verify your HuggingFace account email
   - Provide your name and affiliation
   - Acknowledge the usage restrictions
4. Wait for access approval (usually instant, but can take a few minutes)

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
| `use_fp8` | true | ❌ Config only | Enable FP8 quantization for transformer |
| `num_frames` | 33 | ⚠️ API only | Number of frames to generate (~1.3 seconds at 24fps) |
| `frame_rate` | 24.0 | ⚠️ API only | Output frame rate |
| `randomize_seed` | false | ⚠️ API only | Generate new random seed each chunk |

> [!NOTE]
> Parameters marked "API only" work but don't have UI controls in Scope's main branch.
> Use the `/load` API endpoint to set these parameters. See [PLUGIN.md](PLUGIN.md) for details.

#### Using API-Only Parameters

```bash
# Set num_frames and randomize_seed via API
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

#### LTX-2 Frame Count Formula

LTX-2 works best with frame counts following the formula `8×K+1`:
- 1, 9, 17, 25, **33**, 41, 49, 57, 65, ...

Other values work but may be less optimal.

### Memory Optimization Tips

1. **Use FP8 quantization** (enabled by default) - reduces transformer from ~45GB to ~25GB
2. **Lower resolution** - activations scale with resolution × frames
3. **Fewer frames** - each frame at 512×768 uses ~1.5GB for activations
4. **Set PYTORCH_CUDA_ALLOC_CONF** - `expandable_segments:True` prevents fragmentation

## Limitations

### Current Limitations

- **No real-time streaming** - LTX-2 generates complete clips, not frame-by-frame
- **High VRAM requirement** - 96GB+ recommended
- **Audio disabled** - waiting for Scope audio channel support
- **No image conditioning** - text-to-video only (image conditioning planned)
- **No LoRA support** - planned for future release

### Scope Main Branch Limitations

These features work in the plugin but lack UI controls in Scope's main branch:

| Feature | Status | Workaround |
|---------|--------|------------|
| `num_frames` slider | No UI | Use API to set value |
| `randomize_seed` toggle | No UI | Use API to set value |
| Fixed FPS playback | Partial | Video generated correctly, playback may vary |

See [PLUGIN.md](PLUGIN.md) for the full compatibility matrix and what changes are needed in Scope main to enable these features.

## Architecture

LTX-2 uses a DiT (Diffusion Transformer) architecture with:
- **Gemma 3 12B** text encoder for prompt understanding
- **19B parameter** transformer for video denoising
- **Video VAE** for latent space encoding/decoding
- **Distilled inference** with 8 predefined sigma values

For more details, see the [official LTX-2 documentation](https://github.com/Lightricks/LTX-2).

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
1. Reduce `num_frames` (try 17 instead of 33)
2. Reduce resolution (try 384×512)
3. Ensure no other GPU processes are running
4. Check that FP8 is enabled (`use_fp8: true`)

### Model Download Fails

**"403 Forbidden" or "Access to model is restricted":**
- You haven't accepted the license agreement on HuggingFace
- Go to the model page while logged in and click "Agree and access repository":
  - [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2)
  - [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)

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

LTX-2 generates complete clips rather than streaming frames. Generation time depends on:
- Number of frames
- Resolution
- GPU performance

Typical times on H100: ~10-20 seconds for 33 frames at 512×768.

## Development

See [PLUGIN.md](PLUGIN.md) for detailed integration notes and the roadmap for audio support.

## License

This plugin is licensed under the same terms as the [LTX-2 model](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for the LTX-2 model
- [Daydream](https://daydreamlive.ai/) for the Scope platform
