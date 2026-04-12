# TextVACE: Glyph-Conditioned Video Scene Text Editing via VACE

## Implementation Plan

---

## 1. Project Overview

### 1.1 Task Definition

**Video Scene Text Editing (VSTE):** Given an input video containing scene text, a text mask indicating the text region, and an edit instruction ("Change X to Y"), generate an output video where:
- The specified text region is replaced with the target text
- The target text matches the original text's font style, color, size, and perspective
- Non-text regions remain **pixel-identical** to the input
- The edited text is temporally consistent across frames

### 1.2 Core Innovation: Glyph-Conditioned VACE

We extend VACE (ICCV 2025) with two key innovations:

1. **Font-Aware Glyph Rendering Condition:** Use a VLM to identify the font style of the original text, then render the target text using the matched font as a glyph condition image. This glyph image is VAE-encoded and injected as an additional channel in `vace_context`, providing explicit character-level guidance to the diffusion process.

2. **Pixel-Anchored Denoising (inference-time):** During denoising, force non-mask regions in latent space to remain anchored to the original video's latents, ensuring strict background preservation.

### 1.3 Why This is Novel

| Aspect | Prior Work | Ours |
|--------|-----------|------|
| Task scope | Image-only STE (TextCtrl, FLUX-Text) or non-diffusion video (STRIVE 2021) | **First diffusion-based video STE** |
| Glyph conditioning | FLUX-Text: glyph injection for images via FLUX VAE | Glyph injection in **video DiT** with temporal glyph propagation |
| Font matching | AnyText2: user-specified font file | **VLM-based automatic font recognition** from source video |
| Background fidelity | VACE: inactive/reactive split (soft) | Pixel-anchored latent compositing (hard guarantee) |

---

## 2. Architecture Design

### 2.1 Original VACE Data Flow (Wan2.1-VACE-1.3B)

```
                          vace_in_dim = 96
                    ┌──────────┴──────────┐
Original Video      │                     │
    │                │   Mask (8x8 patch)  │
    ├─ * (1-mask) ──→ VAE.encode → 16ch   │   64ch
    ├─ * mask ──────→ VAE.encode → 16ch   │     │
    │                │       │             │     │
    │                │  concat → 32ch      │     │
    │                │       │             │     │
    │                └───concat────────────┘     │
    │                         │                  │
    │                    vace_context (96ch)      │
    │                         │                  │
    │                  Conv3D(96→1536)            │
    │                         │                  │
    │                  15 VaceBlocks              │
    │                         │                  │
    │                    hints[0..14]             │
    │                         │                  │
    │              DiT blocks: x += hint * scale │
    │                         │                  │
    └─────────────────────────┘                  │
```

### 2.2 TextVACE Extended Data Flow

```
                       vace_in_dim = 112 (+16)
                 ┌──────────┴────────────────────┐
Original Video   │                               │
    │            │  Mask      Glyph Image(NEW)    │
    ├─*(1-m) → VAE→16ch   64ch    │              │
    ├─*m ────→ VAE→16ch    │   VAE.encode→16ch   │
    │            │  │       │      │              │
    │         concat→32ch   │   (NEW channel)     │
    │            │          │      │              │
    │            └──────concat─────┘              │
    │                    │                        │
    │             vace_context (112ch)             │
    │                    │                        │
    │             Conv3D(112→1536)  ← extended    │
    │                    │                        │
    │             15 VaceBlocks (reuse pretrained) │
    │                    │                        │
    │               hints[0..14]                  │
    │                    │                        │
    │         DiT blocks: x += hint * scale       │
    └─────────────────────────────────────────────┘
```

**Key change:** `vace_in_dim` from 96 to 112 (add 16 channels for glyph latent).

### 2.3 Glyph Condition Preparation Pipeline

```
Edit Instruction: "Change ATP to NAD"
                     │
                     ├─ Parse → source_text="ATP", target_text="NAD"
                     │
    Original Video Frame ──→ VLM (Qwen2-VL / GPT-4o)
                     │         "What font does this text most resemble?"
                     │              │
                     │         font_name = "Arial Bold"
                     │              │
                     ├── Render target_text with font_name ──→ glyph_image (white text on black)
                     │              │
                     │         Per-frame perspective transform (from mask geometry)
                     │              │
                     └────── glyph_video: [glyph_frame_0, glyph_frame_1, ...]
                                    │
                              VAE.encode → 16ch glyph latent
                                    │
                              Concat to vace_context
```

### 2.4 Pixel-Anchored Denoising (Inference Only)

During each denoising step t:
```python
# Standard denoising produces latents_t
latents_denoised = scheduler.step(noise_pred, timestep, latents)

# Anchor non-mask regions to original video latents
original_noisy = scheduler.add_noise(original_latents, noise, next_timestep)
mask_latent = downsample_mask_to_latent_space(text_mask)  # (1, 1, T', H', W')

# Composite: keep original in non-mask region, keep generated in mask region
latents_next = mask_latent * latents_denoised + (1 - mask_latent) * original_noisy
```

This is **training-free** and guarantees non-edited regions are perfectly preserved.

---

## 3. Data Preparation

### 3.1 Current Data Structure

```
data/
├── original_videos/     230 MP4 files, 1280x720, 24fps, 120 frames (5s)
├── edited_videos/       230 MP4 files (some with _overlay suffix)
├── text_masks/          230 MP4 mask videos
└── edit_instructions/   230 JSON files: {"instruction_en": "Change X to Y"}
```

### 3.2 Data Processing Steps

#### Step 1: Parse Edit Instructions
```python
# Input:  {"instruction_en": "Change ATP to NAD"}
# Output: source_text="ATP", target_text="NAD"
# Method: regex: r"Change (.+) to (.+)"
# Status: All 230 instructions parseable (verified)
```

#### Step 2: Font Recognition via VLM (offline preprocessing)

For each sample:
1. Extract the first frame from `original_videos/`
2. Extract the text region using `text_masks/` (crop to bounding box)
3. Send cropped region + source_text to VLM with prompt:
   ```
   This image shows the text "{source_text}" in a video frame.
   Identify the closest matching standard font name for this text.
   Return ONLY the font name (e.g., "Arial Bold", "Times New Roman", "Helvetica").
   ```
4. Map VLM output to available system font (fallback: Arial/DejaVu Sans)
5. Save to: `data/font_info/{id}.json` → `{"font_name": "Arial Bold"}`

**VLM options (by cost/quality):**
- Best: Qwen2.5-VL-72B (if available locally or via API)
- Good: GPT-4o via API
- Local free: Qwen2.5-VL-7B (can run on 32GB)

#### Step 3: Render Glyph Video

For each sample:
1. Load `font_name` from Step 2
2. Parse `target_text` from Step 1
3. Load `text_masks/{id}.mp4` → extract per-frame bounding boxes
4. For each frame:
   - Get bounding box from mask
   - Render `target_text` in white on black background, sized to fit bounding box
   - Apply perspective transform matching the mask's geometry
5. Save to: `data/glyph_videos/{id}.mp4` (same resolution/fps as originals)

**Rendering details:**
- Use PIL/Pillow `ImageFont.truetype(font_name, size)`
- Background: black (0,0,0), text: white (255,255,255)
- Size: fit to mask bounding box with padding
- Perspective: estimate from mask contour → apply `cv2.getPerspectiveTransform`

#### Step 4: Create metadata.csv

```csv
video,vace_video,vace_video_mask,glyph_video,prompt
edited_videos/0000007_00000_overlay.mp4,original_videos/0000007_00000.mp4,text_masks/0000007_00000.mp4,glyph_videos/0000007_00000.mp4,Change ATP to NAD
edited_videos/0000015_00000_overlay.mp4,original_videos/0000015_00000.mp4,text_masks/0000015_00000.mp4,glyph_videos/0000015_00000.mp4,Change Carb to Sugar
...
```

**Column semantics:**
| Column | Role | Maps to Training |
|--------|------|-----------------|
| `video` | **Ground truth** (edited video) | `input_video` → VAE encode → training target |
| `vace_video` | **Condition** (original video) | VAE encode → inactive/reactive latents |
| `vace_video_mask` | **Mask** (text region) | Rearranged to 64ch mask latent |
| `glyph_video` | **Glyph condition** (NEW) | VAE encode → 16ch glyph latent |
| `prompt` | **Text instruction** | T5 encode → cross-attention context |

---

## 4. Code Modifications

### 4.1 File Modification List

| File | Change | Description |
|------|--------|-------------|
| `diffsynth/models/wan_video_vace.py` | Modify | Extend `vace_in_dim` default, add glyph channel support |
| `diffsynth/pipelines/wan_video.py` | Modify | Extend `WanVideoUnit_VACE` to process glyph_video |
| `diffsynth/configs/model_configs.py` | Modify | Update VACE-1.3B config if needed |
| `diffsynth/diffusion/loss.py` | Add | New `TextEditSFTLoss` with mask-weighted loss |
| `examples/wanvideo/model_training/train.py` | Modify | Support `glyph_video` as extra input |
| `scripts/prepare_data.py` | **New** | Data preprocessing: font recognition, glyph rendering, metadata generation |
| `scripts/train_textvace.sh` | **New** | Training launch script |

### 4.2 Model Architecture Changes

#### 4.2.1 `diffsynth/models/wan_video_vace.py`

```python
class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=112,  # Changed: 96 → 112 (add 16ch for glyph)
        patch_size=(1, 2, 2),
        ...
    ):
        ...
        # Conv3d input channels: 112 instead of 96
        self.vace_patch_embedding = torch.nn.Conv3d(
            vace_in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
```

**Weight initialization strategy for the new Conv3D:**
```python
# Load pretrained weights for first 96 input channels
pretrained_weight = pretrained_state_dict["vace_patch_embedding.weight"]  # (1536, 96, 1, 2, 2)
new_weight = torch.zeros(1536, 112, 1, 2, 2)
new_weight[:, :96, :, :, :] = pretrained_weight  # Copy pretrained
new_weight[:, 96:, :, :, :] = 0  # Zero-init new channels (glyph starts as no-op)
```

This ensures the model starts from pretrained behavior and gradually learns to use glyph information.

#### 4.2.2 `diffsynth/pipelines/wan_video.py` — WanVideoUnit_VACE

Add glyph_video processing:

```python
class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "vace_video", "vace_video_mask", "vace_reference_image",
                "glyph_video",  # NEW
                "vace_scale", "height", "width", "num_frames",
                "tiled", "tile_size", "tile_stride"
            ),
            output_params=("vace_context", "vace_scale"),
            onload_model_names=("vae",)
        )

    def process(self, pipe, vace_video, vace_video_mask, vace_reference_image,
                glyph_video,  # NEW parameter
                vace_scale, height, width, num_frames, tiled, tile_size, tile_stride):
        ...
        # Existing: inactive (16ch) + reactive (16ch) = 32ch video latents
        vace_video_latents = torch.concat((inactive, reactive), dim=1)

        # NEW: Encode glyph video
        if glyph_video is not None:
            glyph_video_tensor = pipe.preprocess_video(glyph_video)
            glyph_latents = pipe.vae.encode(
                glyph_video_tensor, device=pipe.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            # glyph_latents shape: (1, 16, T', H', W')
            vace_video_latents = torch.concat(
                (vace_video_latents, glyph_latents), dim=1
            )
            # Now: (1, 48, T', H', W') → 32 original + 16 glyph

        # Existing: mask latents (64ch)
        vace_mask_latents = rearrange(...)

        # Final context: 48 + 64 = 112 channels (was 32 + 64 = 96)
        vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
        ...
```

#### 4.2.3 `diffsynth/diffusion/loss.py` — Optional Mask-Weighted Loss

```python
def TextEditSFTLoss(pipe, **inputs):
    """SFT loss with optional mask weighting for text editing."""
    # Standard flow matching setup
    timestep_id = torch.randint(min_boundary, max_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(...)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    # Compute per-pixel MSE
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
```

Note: The standard `FlowMatchSFTLoss` may suffice initially. The mask-weighted variant
is a secondary experiment to try if baseline results show the model struggles with text
region quality specifically.

### 4.3 Training Script Modifications

#### 4.3.1 `examples/wanvideo/model_training/train.py`

Add `glyph_video` to extra_inputs handling:

```python
def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
    for extra_input in extra_inputs:
        ...
        elif extra_input == "glyph_video":
            inputs_shared["glyph_video"] = data["glyph_video"]
        ...
```

### 4.4 Weight Loading Strategy

When initializing from pretrained Wan2.1-VACE-1.3B:

1. Load all pretrained weights normally (DiT, VAE, T5 — all frozen)
2. For VACE module:
   - All VaceWanAttentionBlock weights: load directly (architecture unchanged)
   - `vace_patch_embedding` Conv3D: expand input channels from 96 to 112
     - Copy pretrained weights for channels 0-95
     - Zero-initialize channels 96-111 (glyph channels)
     - Copy pretrained bias as-is

This needs a custom state dict converter or a post-load weight patching function.

---

## 5. Training Configuration

### 5.1 Hardware Constraints

- GPU: 32GB VRAM (single GPU)
- Model: Wan2.1-VACE-1.3B
- Training mode: SFT (full fine-tuning of VACE module only; DiT backbone frozen)

### 5.2 Memory Budget Estimation

| Component | Memory (bf16) | Trainable |
|-----------|--------------|-----------|
| DiT 1.3B (frozen) | ~2.6 GB | No |
| VACE module (~120M params) | ~0.24 GB | **Yes** |
| VACE optimizer states (AdamW, 2x) | ~0.48 GB | - |
| VACE gradients | ~0.24 GB | - |
| T5 encoder (frozen) | ~4.5 GB | No |
| VAE (frozen) | ~0.2 GB | No |
| Activations (grad checkpoint + CPU offload) | ~8-15 GB | - |
| Input tensors + latents | ~2-5 GB | - |
| **Total estimated** | **~18-28 GB** | |

**Conclusion: 32GB is sufficient for VACE SFT with the 1.3B model.**

### 5.3 Training Parameters

```bash
# scripts/train_textvace.sh
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/ \
  --dataset_metadata_path data/metadata.csv \
  --data_file_keys "video,vace_video,glyph_video" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/TextVACE_sft" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,glyph_video" \
  --use_gradient_checkpointing_offload
```

**Key parameter choices:**
- `--num_frames 17`: Start with fewer frames to fit 32GB (4*4+1=17 is valid for time_division_factor=4)
- `--height 480 --width 832`: Standard Wan2.1 resolution
- `--trainable_models "vace"`: Only train VACE module (SFT, not LoRA)
- `--learning_rate 5e-5`: Same as official VACE full training
- If OOM: reduce to `--height 320 --width 576` or `--num_frames 13`

### 5.4 Training Data Handling

The training video resolution (1280x720, 120 frames) will be:
- Spatially resized: 1280x720 → 480x832 (center crop + resize)
- Temporally subsampled: 120 frames → 17 frames (uniform sampling at ~3.4fps)
- Both original_video and edited_video undergo the same transform for alignment

### 5.5 Mask Handling During Training

The `vace_video_mask` (text_masks/) needs to be loaded alongside `vace_video`.

**Important:** In the training pipeline:
- `video` column = edited_videos (ground truth output)
- `vace_video` column = original_videos (condition input)
- `vace_video_mask` = text_masks (comes from vace_video input with mask)

The VACE pipeline already handles mask as part of vace_video processing.
We need to ensure the mask is passed correctly. Options:
1. Concatenate mask as alpha channel of vace_video (requires custom operator)
2. Pass mask as separate `vace_video_mask` parameter

Option 2 is cleaner — add `vace_video_mask` as an extra data key:
```
--data_file_keys "video,vace_video,vace_video_mask,glyph_video"
--extra_inputs "vace_video,vace_video_mask,glyph_video"
```

---

## 6. Implementation Phases

### Phase 1: Data Preparation (Week 1)

#### 1a. Font Recognition Script
```
scripts/prepare_data.py --step font_recognition
```
- Input: original_videos/, text_masks/, edit_instructions/
- Process: Extract text region → VLM query → font name
- Output: data/font_info/{id}.json

**VLM strategy:**
- Try Qwen2.5-VL-7B locally first (runs on 32GB)
- If quality insufficient, use API-based VLM
- Fallback: use a small set of common fonts (Arial, Times, Helvetica, Courier, etc.)
  and match via visual similarity

#### 1b. Glyph Video Rendering Script
```
scripts/prepare_data.py --step render_glyphs
```
- Input: font_info/, text_masks/, edit_instructions/
- Process: For each frame, render target text in matched font, apply perspective
- Output: data/glyph_videos/{id}.mp4

#### 1c. Metadata Generation
```
scripts/prepare_data.py --step create_metadata
```
- Output: data/metadata.csv

### Phase 2: Model Modification (Week 1-2)

#### 2a. Extend VACE Model
- Modify `wan_video_vace.py`: change `vace_in_dim` default to 112
- Modify `model_configs.py`: update VACE-1.3B config
- Write weight patching function for Conv3D expansion

#### 2b. Extend Pipeline
- Modify `WanVideoUnit_VACE.process()`: add glyph_video encoding
- Add `glyph_video` to input_params

#### 2c. Extend Training Script
- Add `glyph_video` handling in `parse_extra_inputs`
- Add custom data operator for mask video loading

#### 2d. Test Forward Pass
- Verify model loads with expanded weights
- Verify training pipeline runs for 1 step without errors
- Check memory usage

### Phase 3: Baseline Training (Week 2)

#### 3a. Baseline 1: Original VACE (no glyph condition)
- Train with vace_in_dim=96, no glyph_video
- This establishes the performance floor

#### 3b. TextVACE Training (with glyph condition)
- Train with vace_in_dim=112, glyph_video included
- Compare against baseline

### Phase 4: Pixel-Anchored Denoising (Week 2-3)

#### 4a. Implement in inference pipeline
- Modify denoising loop in `model_fn_wan_video` or inference script
- Add latent compositing after each denoising step

#### 4b. Evaluate
- Compare: TextVACE vs TextVACE + Pixel-Anchored
- Metrics: PSNR/SSIM on non-mask regions, OCR accuracy on text region

### Phase 5: Evaluation & Ablation (Week 3-4)

#### 5a. Evaluation Metrics
| Metric | Measures | Tool |
|--------|----------|------|
| OCR Accuracy | Text correctness | PaddleOCR / EasyOCR |
| PSNR (non-mask) | Background preservation | skimage |
| SSIM (non-mask) | Structural similarity of background | skimage |
| FVD | Video quality | pytorch-fvd |
| Temporal Consistency | Frame-to-frame smoothness | optical flow variance |
| LPIPS (text region) | Text visual quality | lpips |

#### 5b. Baselines for Comparison
1. **STRIVE** (ICCV 2021) — only prior work
2. **TextCtrl per-frame** — apply image STE to each frame independently
3. **VACE vanilla fine-tune** — VACE without glyph condition
4. **TextVACE (ours)** — full method
5. **TextVACE + Pixel-Anchored** — full method + inference trick

#### 5c. Ablation Studies
- w/o glyph condition (= baseline VACE fine-tune)
- w/o font matching (use default font for glyph)
- w/o pixel-anchored denoising
- Different glyph rendering strategies (binary vs grayscale vs colored)

---

## 7. Detailed Memory Optimization for 32GB

If training encounters OOM, apply these mitigations in order:

| Priority | Strategy | Impact |
|----------|----------|--------|
| 1 | `--use_gradient_checkpointing_offload` | Already enabled. Offloads activations to CPU. |
| 2 | Reduce `--num_frames` from 17 to 13 | ~25% less memory for activations |
| 3 | Reduce resolution to 320x576 | ~50% less memory |
| 4 | Use `--gradient_accumulation_steps 2` with smaller effective batch | Trades speed for memory |
| 5 | Enable FP8 for frozen DiT: `--fp8_models dit` | If supported by hardware |
| 6 | Fall back to LoRA (as last resort) | Much less memory but less expressiveness |

---

## 8. Key Technical Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| VLM font recognition inaccurate | Glyph condition wrong → model confused | Use small set of common fonts; ablate with/without font matching |
| 230 samples too few for SFT | Overfitting | High dataset_repeat (100+), strong augmentation, consider additional synthetic data |
| Glyph perspective transform inaccurate | Misaligned condition | Start with simple center-aligned glyph (no perspective), add perspective as improvement |
| Conv3D expansion breaks pretrained features | Poor initial convergence | Zero-init new channels (verified: standard practice in ControlNet) |
| 32GB OOM during training | Cannot train | Reduce resolution/frames, or fall back to LoRA as stepping stone |

---

## 9. File Structure After Implementation

```
DiffSynth-Studio-TextVACE/
├── data/
│   ├── original_videos/          (existing)
│   ├── edited_videos/            (existing)
│   ├── text_masks/               (existing)
│   ├── edit_instructions/        (existing)
│   ├── font_info/                (NEW: VLM font recognition results)
│   ├── glyph_videos/             (NEW: rendered glyph condition videos)
│   └── metadata.csv              (NEW: training metadata)
├── diffsynth/
│   ├── models/
│   │   └── wan_video_vace.py     (MODIFIED: vace_in_dim=112)
│   ├── pipelines/
│   │   └── wan_video.py          (MODIFIED: glyph_video processing)
│   ├── configs/
│   │   └── model_configs.py      (MODIFIED: config update)
│   └── diffusion/
│       └── loss.py               (OPTIONAL: TextEditSFTLoss)
├── scripts/
│   ├── prepare_data.py           (NEW: data preprocessing pipeline)
│   ├── train_textvace.sh         (NEW: training launch script)
│   ├── inference_textvace.py     (NEW: inference with pixel-anchored denoising)
│   └── evaluate.py               (NEW: evaluation metrics)
├── examples/wanvideo/model_training/
│   └── train.py                  (MODIFIED: glyph_video support)
└── docs/
    └── TextVACE_Implementation_Plan.md  (this document)
```

---

## 10. Summary: What We Submit to NeurIPS

### Paper Title (draft)
**"TextVACE: Font-Aware Glyph-Conditioned Video Diffusion for Scene Text Editing"**

### Contributions
1. **Task formulation:** First formal definition of diffusion-based video scene text editing, with a curated dataset and evaluation benchmark
2. **Method:** Glyph-Conditioned VACE with VLM-based font recognition, extending VACE with explicit character-level spatial guidance
3. **Technique:** Pixel-Anchored Denoising for strict background preservation in masked video editing
4. **Experiments:** Comprehensive evaluation showing SOTA on video STE, with ablation validating each component

### Key Experiments
- TextVACE vs STRIVE (2021) — show diffusion-based approach is superior
- TextVACE vs TextCtrl-per-frame — show temporal consistency advantage
- TextVACE vs vanilla VACE fine-tune — show glyph condition is essential
- Ablation on font matching, glyph rendering, pixel-anchoring
