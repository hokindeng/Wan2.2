# Wan2.2 TI2V-5B: Software Engineering Architecture

## Overview

The Text+Image-to-Video (TI2V) pipeline is a unified architecture that supports both text-to-video (T2V) and image-to-video (I2V) generation through a single model. This document provides a software engineering perspective on the system architecture.

## Core Architecture

### 1. Class Hierarchy

```
WanTI2V (wan/textimage2video.py)
├── T5EncoderModel (text encoding)
├── Wan2_2_VAE (video encoding/decoding)
└── WanModel (DiT backbone)
    └── 30x WanAttentionBlock
```

### 2. Pipeline Initialization (`WanTI2V.__init__`)

**File**: `wan/textimage2video.py:36-116`

**Component Loading Order**:

1. **T5 Text Encoder**
   - Model: `umt5-xxl` (Google's multilingual T5)
   - Checkpoint: `models_t5_umt5-xxl-enc-bf16.pth`
   - Output: `[seq_len=512, dim=4096]` embeddings
   - Options: FSDP sharding, CPU placement for memory efficiency

2. **VAE (Variational Autoencoder)**
   - Architecture: 3D Causal Convolutional VAE
   - Checkpoint: `Wan2.2_VAE.pth`
   - Latent channels: 48 (vs 16 in Wan 2.1)
   - Stride: `(4, 16, 16)` for (T, H, W)
   - Example: 121 frames @ 1280x704 → 31x44x80 latent

3. **DiT (Diffusion Transformer)**
   - Architecture: Flow-Matching DiT
   - Parameters: 5 billion
   - Layers: 30
   - Hidden dim: 3072
   - Attention heads: 24
   - Patch size: `(1, 2, 2)` - minimal temporal patching

**Memory Optimization Options**:
- `t5_fsdp`: FSDP (Fully Sharded Data Parallel) for T5
- `dit_fsdp`: FSDP for DiT model
- `use_sp`: Sequence parallelism with DeepSpeed Ulysses
- `t5_cpu`: Keep T5 on CPU, only move to GPU during encoding
- `convert_model_dtype`: Convert to bf16 for inference
- `offload_model`: Offload models to CPU between forward passes

---

## Entry Point Flow

### CLI → Pipeline

**File**: `generate.py:315-454`

```python
# 1. Parse arguments
args = _parse_args()

# 2. Load config
cfg = WAN_CONFIGS['ti2v-5B']  # from wan/configs/wan_ti2v_5B.py

# 3. Load image if provided
img = Image.open(args.image).convert("RGB") if args.image else None

# 4. Initialize pipeline
wan_ti2v = wan.WanTI2V(
    config=cfg,
    checkpoint_dir=args.ckpt_dir,
    device_id=device,
    rank=rank,
    t5_fsdp=args.t5_fsdp,
    dit_fsdp=args.dit_fsdp,
    use_sp=(args.ulysses_size > 1),
    t5_cpu=args.t5_cpu,
    convert_model_dtype=args.convert_model_dtype,
)

# 5. Generate video
video = wan_ti2v.generate(
    args.prompt,
    img=img,  # Key parameter: determines I2V vs T2V
    size=SIZE_CONFIGS[args.size],
    max_area=MAX_AREA_CONFIGS[args.size],
    frame_num=args.frame_num,
    shift=args.sample_shift,
    sample_solver=args.sample_solver,
    sampling_steps=args.sample_steps,
    guide_scale=args.sample_guide_scale,
    seed=args.base_seed,
    offload_model=args.offload_model
)
```

---

## Image-to-Video Path (I2V)

### Method: `WanTI2V.i2v()`
**File**: `wan/textimage2video.py:413-619`

### Phase 1: Image Preprocessing

```python
# 1. Calculate optimal output size
ow, oh = best_output_size(iw, ih, dw=32, dh=32, max_area=704*1280)

# 2. Resize with LANCZOS (high-quality)
img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

# 3. Center crop to exact dimensions
x1, y1 = (img.width - ow) // 2, (img.height - oh) // 2
img = img.crop((x1, y1, x1 + ow, y1 + oh))

# 4. Normalize to [-1, 1]
img = TF.to_tensor(img).sub_(0.5).div_(0.5)
# Output shape: [3, H, W]
```

**Key Insight**: The aspect ratio of the output video follows the input image, not the `size` parameter. The `max_area` parameter controls the total resolution.

### Phase 2: Latent Space Setup

```python
# 1. Calculate sequence length for DiT
F = frame_num  # e.g., 121
seq_len = ((F-1)//4 + 1) * (H//16) * (W//16) // (2*2)
seq_len = ceil(seq_len / sp_size) * sp_size  # Align for sequence parallel

# 2. Initialize random noise tensor
noise = torch.randn(
    48,  # z_dim (latent channels)
    (F-1)//4 + 1,  # temporal: 121 frames → 31 latent frames
    H//16,  # spatial height
    W//16,  # spatial width
    generator=seed_g
)
# Example: 121 frames @ 704x1280 → noise shape [48, 31, 44, 80]
```

### Phase 3: Text and Image Encoding

```python
# 1. Encode text prompt (conditional)
context = self.text_encoder([input_prompt], device)
# Shape: [512, 4096]

# 2. Encode negative prompt (unconditional)
context_null = self.text_encoder([n_prompt], device)
# Shape: [512, 4096]

# 3. Encode image to latent space
z = self.vae.encode([img])
# Input: [3, 1, H, W]
# Output: [48, 1, H//16, W//16]
```

### Phase 4: Masking Strategy

**File**: `wan/utils/utils.py:172-199`

```python
mask1, mask2 = masks_like([noise], zero=True)
# mask1: conditioning strength (all ones or with first frame special handling)
# mask2: binary mask [48, 31, 44, 80] where:
#   - First frame (t=0): zeros
#   - Remaining frames: ones

# Initial latent blending
latent = (1.0 - mask2[0]) * z[0] + mask2[0] * noise
# Interpretation:
#   - First frame: 100% image latent (z[0])
#   - Rest: 100% noise
```

**Critical Insight**: This creates a "guided" denoising where the first frame is always the input image latent, ensuring temporal consistency.

### Phase 5: Scheduler Setup

```python
if sample_solver == 'unipc':
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False
    )
    sample_scheduler.set_timesteps(sampling_steps=40, shift=5.0)
    
elif sample_solver == 'dpm++':
    sample_scheduler = FlowDPMSolverMultistepScheduler(...)
    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
    timesteps = retrieve_timesteps(sample_scheduler, sigmas=sampling_sigmas)
```

**Timestep Range**: 1000 → 0 (reverse diffusion process)

### Phase 6: Denoising Loop

```python
for _, t in enumerate(tqdm(timesteps)):  # 40-50 iterations
    # 1. Prepare input
    latent_model_input = [latent.to(self.device)]
    timestep = torch.stack([t])  # scalar → [1]
    
    # 2. Create spatially-varying timesteps
    # First frame has t=0 (no noise), rest have t=timestep
    temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
    temp_ts = torch.cat([
        temp_ts,
        temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
    ])
    timestep = temp_ts.unsqueeze(0)  # [1, seq_len]
    
    # 3. Conditional forward pass
    noise_pred_cond = self.model(
        latent_model_input,
        t=timestep,
        context=[context[0]],
        seq_len=seq_len
    )[0]
    
    if offload_model:
        torch.cuda.empty_cache()
    
    # 4. Unconditional forward pass
    noise_pred_uncond = self.model(
        latent_model_input,
        t=timestep,
        context=context_null,
        seq_len=seq_len
    )[0]
    
    if offload_model:
        torch.cuda.empty_cache()
    
    # 5. Classifier-Free Guidance
    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
    # guide_scale typically 5.0
    
    # 6. Scheduler step (ODE solver)
    temp_x0 = sample_scheduler.step(
        noise_pred.unsqueeze(0),
        t,
        latent.unsqueeze(0),
        generator=seed_g
    )[0]
    latent = temp_x0.squeeze(0)
    
    # 7. Re-blend with image latent (CRITICAL!)
    latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent
    # This ensures the first frame always remains the input image
```

**Key Observations**:
1. **Spatially-varying timesteps**: The model receives different timestep values for different spatial locations (first frame vs rest)
2. **Classifier-Free Guidance**: Two forward passes per step (2x compute)
3. **Re-blending**: After each denoising step, the first frame latent is re-injected from the encoded image
4. **Memory management**: Optional GPU cache clearing between passes

### Phase 7: VAE Decoding

```python
videos = self.vae.decode([latent])
# Input: [48, 31, 44, 80] latent
# Output: [3, 121, 704, 1280] video
# Range: [-1, 1]
```

**VAE Decoder Architecture** (`wan/modules/vae2_2.py`):
- 3D Causal Convolutional decoder
- Temporal upsampling: 31 → 121 frames (4x)
- Spatial upsampling: 44x80 → 704x1280 (16x)
- Channel denormalization with learned mean/std per channel

---

## Text-to-Video Path (T2V)

### Method: `WanTI2V.t2v()`
**File**: `wan/textimage2video.py:239-411`

**Key Differences from I2V**:

1. **No Image Encoding**:
   - Pure random noise initialization
   - No VAE encoding step

2. **No Masking**:
   ```python
   mask1, mask2 = masks_like(noise, zero=False)
   # Both masks are all ones (no first-frame conditioning)
   ```

3. **Simpler Denoising**:
   - No re-blending with image latent
   - Uniform timesteps across all spatial locations

4. **Fixed Output Size**:
   - Respects `size=(width, height)` parameter
   - Not constrained by input image aspect ratio

---

## DiT Model Architecture

### Class: `WanModel`
**File**: `wan/modules/model.py:294-485`

### Forward Pass Signature

```python
def forward(
    self,
    x: List[Tensor],      # Input latents [C_in=48, F, H, W]
    t: Tensor,            # Timesteps [B] or [B, seq_len]
    context: List[Tensor], # Text embeddings [512, 4096]
    seq_len: int,         # Max sequence length for padding
    y: List[Tensor] = None # Optional: image conditioning (unused in TI2V)
) -> List[Tensor]:        # Output: [C_out=48, F, H, W]
```

### Architecture Stages

1. **Patch Embedding**
   ```python
   self.patch_embedding = nn.Conv3d(
       in_channels=48,
       out_channels=3072,
       kernel_size=(1, 2, 2),
       stride=(1, 2, 2)
   )
   # [48, F, H, W] → [3072, F, H/2, W/2]
   ```

2. **Flattening and Padding**
   ```python
   x = x.flatten(2).transpose(1, 2)  # [B, 3072, F*H*W] → [B, F*H*W, 3072]
   # Pad to seq_len
   x = torch.cat([x, x.new_zeros(1, seq_len - x.size(1), 3072)], dim=1)
   # [B, seq_len, 3072]
   ```

3. **Time Embedding**
   ```python
   # Sinusoidal positional encoding
   e = sinusoidal_embedding_1d(freq_dim=256, t)
   # MLP projection
   e = self.time_embedding(e)  # → [B, 256]
   e0 = self.time_projection(e).unflatten(1, (6, 3072))  # [B, seq_len, 6, 3072]
   ```

4. **Text Embedding**
   ```python
   context = self.text_embedding(context)  # Linear(4096 → 3072)
   # [B, 512, 3072]
   ```

5. **Transformer Blocks (30 layers)**
   ```python
   for block in self.blocks:
       x = block(
           x,                  # [B, seq_len, 3072]
           e=e0,               # Time modulation
           seq_lens=seq_lens,  # Actual lengths (no padding)
           grid_sizes=grid_sizes,  # (F, H, W) for RoPE
           freqs=self.freqs,   # Precomputed RoPE freqs
           context=context,    # Text embeddings
           context_lens=None   # Use all 512 tokens
       )
   ```

### WanAttentionBlock Details

**File**: `wan/modules/model.py:183-259`

Each block contains:
1. **Self-Attention with RoPE**
   - QK normalization (improves training stability)
   - Rotary Position Embedding (better relative position encoding)
   - Window size: `(-1, -1)` = global attention

2. **Cross-Attention to Text**
   - Queries: from video latents
   - Keys/Values: from text embeddings
   - Optional normalization for cross-attention

3. **Feed-Forward Network**
   - `3072 → 14336 → 3072` (hidden expansion ~4.67x)
   - GELU activation

4. **Modulation**
   - Time embedding modulates both normalization and gating
   - AdaLN-Zero style: `norm(x) * (1 + scale) + shift`

### Output Head

```python
self.head = Head(dim=3072, out_dim=48, patch_size=(1,2,2))
# LayerNorm + Linear projection + Unpatchify
# [B, seq_len, 3072] → [B, 48, F, H/2, W/2]
```

---

## VAE Architecture

### Class: `Wan2_2_VAE`
**File**: `wan/modules/vae2_2.py:888-1051`

### Encoder

```python
self.encoder = Encoder3d(
    dim=160,
    z_dim=48*2,  # 96 (split into mean + logvar)
    dim_mult=[1, 2, 4, 4],  # 160 → 320 → 640 → 640
    num_res_blocks=2,
    temperal_downsample=[False, True, True]  # 1 → 2 → 4
)
```

**Downsampling**:
- Spatial: 3 stages of 2x → total 16x
- Temporal: 2 stages of 2x → total 4x
- Input: `[3, F, H, W]`
- Output: `[48, F//4, H//16, W//16]`

### Decoder

```python
self.decoder = Decoder3d(
    dec_dim=256,  # Larger decoder for better quality
    z_dim=48,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    temperal_upsample=[True, True, False]
)
```

**Upsampling**:
- Temporal: frame-by-frame decoding with causal convolutions
- Spatial: transposed convolutions
- Output: `[3, F, H, W]` in range `[-1, 1]`

### Normalization

```python
self.mean = tensor([...])  # 48 values, per-channel mean
self.std = tensor([...])   # 48 values, per-channel std

# Encoding
z = (z - self.mean) / self.std

# Decoding
z = z * self.std + self.mean
```

---

## Scheduler Algorithms

### FlowUniPCMultistepScheduler

**File**: `wan/utils/fm_solvers_unipc.py`

- **Type**: Unified Predictor-Corrector
- **Order**: Multi-step (typically 2-3)
- **Shift parameter**: Controls noise schedule
  - Higher shift (5.0) → more noise early, cleaner later
  - Helps with high-resolution video generation

**Timestep schedule**:
```python
timesteps = linspace(num_train_timesteps-1, 0, sampling_steps)
# Apply shift transformation
timesteps = timesteps / (shift + (1-shift)*timesteps/num_train_timesteps)
```

### FlowDPMSolverMultistepScheduler

**File**: `wan/utils/fm_solvers.py`

- **Type**: DPM-Solver++ (optimized ODE solver)
- **Order**: 2nd or 3rd order
- **Better for**: Fewer sampling steps (20-30)

---

## Memory Management

### Offload Strategy

```python
if offload_model:
    # Before each component usage:
    self.text_encoder.model.to(device)
    context = self.text_encoder([prompt], device)
    self.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    
    # In denoising loop:
    self.model.to(device)
    noise_pred = self.model(...)
    torch.cuda.empty_cache()  # After each forward pass
```

**Memory Breakdown** (approximate, single GPU):
- DiT model: ~19 GB (5B params × bf16)
- T5 encoder: ~10 GB (4.7B params × bf16)
- VAE: ~3 GB
- Activations per forward pass: ~5-15 GB (depends on resolution)

**With offloading**: Can run on 24GB GPU (e.g., RTX 4090)
**Without offloading**: Requires 80GB GPU (e.g., A100)

### Distributed Strategies

1. **FSDP (Fully Sharded Data Parallel)**
   ```python
   wan_ti2v = wan.WanTI2V(
       t5_fsdp=True,   # Shard T5 across GPUs
       dit_fsdp=True,  # Shard DiT across GPUs
   )
   ```
   - Shards model parameters across GPUs
   - All-gather during forward pass
   - Reduces memory per GPU

2. **Sequence Parallelism (Ulysses)**
   ```python
   torchrun --nproc_per_node=8 generate.py --ulysses_size 8
   ```
   - Splits sequence length across GPUs
   - Requires `num_heads % ulysses_size == 0`
   - Better for long sequences

---

## Key Software Engineering Insights

### 1. Unified Architecture Pattern

The TI2V model elegantly handles both T2V and I2V through:
- **Polymorphic `generate()` method**: Routes to `t2v()` or `i2v()` based on `img` parameter
- **Shared DiT backbone**: Same model type `'ti2v'`, different conditioning
- **Masking abstraction**: `masks_like()` with `zero` parameter controls conditioning strategy

### 2. Temporal Consistency Mechanisms

**I2V Mode**:
- First frame is **always** the encoded input image (re-blended every step)
- Spatially-varying timesteps: first frame has `t=0`, rest follow schedule
- This prevents drift from the input image

**T2V Mode**:
- No special first-frame treatment
- Uniform timesteps across all frames
- Model learns temporal consistency from training data

### 3. Memory-Compute Tradeoff

The `offload_model` parameter represents a classic engineering tradeoff:
- **Enabled**: ~3x slower, runs on 24GB GPU
- **Disabled**: ~3x faster, requires 80GB GPU
- Implementation: Simple CPU↔GPU transfers with cache clearing

### 4. Flow Matching vs DDPM

Unlike traditional DDPM (Denoising Diffusion Probabilistic Models), this uses **Flow Matching**:
- Learns ODE (Ordinary Differential Equation) from noise to data
- Continuous time formulation: `t ∈ [0, 1000]` rather than discrete steps
- Better sample quality with fewer steps
- Training on **velocity** prediction rather than noise prediction

### 5. Type Safety and Configuration

**EasyDict pattern** (`wan/configs/`):
```python
ti2v_5B = EasyDict(__name__='Config: Wan TI2V 5B')
ti2v_5B.dim = 3072
ti2v_5B.num_layers = 30
# ...
```
- Allows dot notation: `cfg.dim`
- Easier to read than nested dictionaries
- Still serializable for checkpointing

---

## Performance Characteristics

### Inference Speed (single GPU, bf16, offload_model=True)

- **RTX 4090 (24GB)**:
  - 720p (1280x704), 121 frames: ~8-10 minutes
  - Each denoising step: ~10-12 seconds

- **A100 (80GB, no offload)**:
  - 720p, 121 frames: ~3-4 minutes
  - Each denoising step: ~4-5 seconds

### Scaling with Multi-GPU (8x A100)

- **FSDP + Ulysses (ulysses_size=8)**:
  - 720p, 121 frames: ~60-90 seconds
  - Near-linear scaling up to 8 GPUs
  - Memory per GPU: ~15-20 GB

---

## Comparison: TI2V vs I2V Models

| Aspect | WanTI2V (Unified) | WanI2V (Separate) |
|--------|-------------------|-------------------|
| **Model Type** | `'ti2v'` | `'i2v'` |
| **Architecture** | Single DiT | Dual DiT (high/low noise) |
| **VAE** | Wan2_2_VAE (48 channels) | Wan2_1_VAE (16 channels) |
| **Masking** | Spatial mask for first frame | Boundary-based progressive masking |
| **Flexibility** | Can do T2V or I2V | I2V only |
| **Training** | Trained on both modalities | Trained on I2V only |

**File References**:
- TI2V: `wan/textimage2video.py`
- I2V: `wan/image2video.py`

---

## Conclusion

The Wan2.2 TI2V-5B system demonstrates several advanced software engineering principles:

1. **Unified Architecture**: One model, multiple modalities through conditional execution
2. **Scalable Design**: Supports single GPU to multi-node with FSDP and sequence parallelism
3. **Memory Efficiency**: Intelligent offloading and caching strategies
4. **Modular Components**: Clear separation of VAE, text encoder, and DiT backbone
5. **Flow Matching**: Modern diffusion approach for better sample efficiency

The codebase is well-structured for both research and production deployment, with careful attention to memory management and distributed training.

---

## Further Reading

- **Model Architecture**: `wan/modules/model.py`
- **VAE Details**: `wan/modules/vae2_2.py`
- **Flow Matching Schedulers**: `wan/utils/fm_solvers.py`, `wan/utils/fm_solvers_unipc.py`
- **Distributed Training**: `wan/distributed/fsdp.py`, `wan/distributed/sequence_parallel.py`
- **Configuration**: `wan/configs/wan_ti2v_5B.py`

