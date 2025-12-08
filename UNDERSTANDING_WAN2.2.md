# 🎬 Complete Guide to Understanding Wan2.2

This guide contains 3 comprehensive Mermaid diagrams to help you fully understand the Wan2.2 video generation system.

---

## 📚 Files Overview

### 1. **Wan2.2_Complete_Architecture.mmd** (8 Diagrams)
The comprehensive technical deep-dive covering:

- **Diagram 1:** Overall system architecture and model variants
- **Diagram 2:** Video generation pipeline (T2V example)
- **Diagram 3:** MoE expert switching mechanism
- **Diagram 4:** Image-to-video workflow
- **Diagram 5:** Speech-to-video workflow
- **Diagram 6:** Training vs inference comparison
- **Diagram 7:** Memory optimization strategies
- **Diagram 8:** Complete end-to-end pipeline

**Use this when:** You want detailed technical understanding of each component.

### 2. **Wan2.2_Quick_Reference.mmd** (1 Diagram)
A one-page visual summary showing:

- All 5 model variants at a glance
- Core architecture components
- MoE expert system
- Generation pipeline steps
- Output formats and performance metrics

**Use this when:** You need a quick mental model of the entire system.

### 3. **Wan2.2_Example_Walkthrough.mmd** (1 Diagram)
A concrete step-by-step example:

- Generating "A cat walking in snow"
- Each timestep explained
- Expert switching in action
- Tensor shapes and computations
- Performance breakdown

**Use this when:** You want to see exactly how the system works in practice.

---

## 🎯 How to View the Diagrams

### Option 1: GitHub (Easiest)
1. Push these files to your GitHub repo
2. Open any `.mmd` file in GitHub
3. GitHub automatically renders Mermaid diagrams

### Option 2: Mermaid Live Editor
1. Go to https://mermaid.live/
2. Copy the content of any `.mmd` file
3. Paste into the editor
4. Instant visualization!

### Option 3: VS Code
1. Install "Markdown Preview Mermaid Support" extension
2. Create a markdown file and embed:
   ````markdown
   ```mermaid
   (paste diagram content here)
   ```
   ````
3. Open preview (Ctrl+Shift+V)

### Option 4: Command Line (using mmdc)
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate PNG images
mmdc -i Wan2.2_Complete_Architecture.mmd -o architecture.png
mmdc -i Wan2.2_Quick_Reference.mmd -o quick_reference.png
mmdc -i Wan2.2_Example_Walkthrough.mmd -o example.png
```

---

## 🧠 Key Concepts Explained

### What is MoE (Mixture of Experts)?

Instead of one giant 27B model, Wan2.2 uses **two specialized 14B experts**:

```
Traditional Approach:
┌─────────────────────────┐
│   Single 27B Model      │  ← Used for ALL timesteps
│   (Always active)       │     Slow, memory-hungry
└─────────────────────────┘

Wan2.2 MoE Approach:
┌─────────────────────────┐
│  High-Noise Expert 14B  │  ← Active when t ≥ 0.875
│  (Layout, composition)  │     Specialized for early steps
└─────────────────────────┘

┌─────────────────────────┐
│  Low-Noise Expert 14B   │  ← Active when t < 0.875
│  (Details, refinement)  │     Specialized for late steps
└─────────────────────────┘

Result: 2× parameters, SAME speed!
```

**Why it works:**
- Early denoising needs different skills than late denoising
- Each expert becomes highly specialized
- Total 27B params, but only 14B active per step
- Better quality + same compute cost = WIN! 🎉

### What is Flow Matching?

Instead of predicting noise (like traditional diffusion), Wan2.2 predicts **velocity**:

```
Traditional Diffusion:
  x_noisy → [Model] → predicted_noise
  Then: x_less_noisy = x_noisy - noise

Flow Matching (Wan2.2):
  z_t → [Model] → predicted_velocity
  Then: z_{t-1} = z_t + Δt × velocity

Advantages:
  ✓ More stable training
  ✓ Better quality
  ✓ Fewer sampling steps needed
```

### Understanding the Pipeline

```
1. INPUT PROCESSING
   Text "A cat walking in snow"
   ↓
   T5 Encoder → embeddings [1, 512, 4096]

2. NOISE INITIALIZATION
   Random Gaussian noise
   ↓
   Shape: [1, 45, 16, 40, 40] (latent space)

3. ITERATIVE DENOISING (40 steps)
   For each timestep t from 40 to 1:
     ├─ Select expert (high-noise or low-noise)
     ├─ Forward pass with text context
     ├─ Apply classifier-free guidance
     └─ Update latent: z = z + Δt × velocity

4. VAE DECODING
   Latent [1, 45, 16, 40, 40]
   ↓
   Video [1, 45, 3, 720, 1280]

5. SAVE
   MP4 file, H.264 codec, 16 fps
```

---

## 📊 Model Comparison

| Model | Parameters | Active | VAE | Resolution | Speed (1×A100) |
|-------|-----------|--------|-----|------------|----------------|
| **T2V-A14B** | 27B | 14B | Wan2.1 | 480p-720p | ~40 min |
| **I2V-A14B** | 27B | 14B | Wan2.1 | 480p-720p | ~40 min |
| **TI2V-5B** | 5B | 5B | Wan2.2 | 720p | ~9 min |
| **S2V-14B** | 14B | 14B | Wan2.1 | 480p-720p | ~35 min |
| **Animate-14B** | 14B | 14B | Wan2.1 | 720p | ~30 min |

**With 8×A100 GPUs:**
- T2V-A14B: ~8 minutes (5× speedup with FSDP+Ulysses)
- Memory per GPU: ~60-80 GB

---

## 💡 Common Questions

### Q: Why two VAE versions?
**A:** 
- **Wan2.1 VAE:** 4×8×8 compression, used by 14B models
- **Wan2.2 VAE:** 4×16×16 compression (4× higher!), enables TI2V-5B to run on consumer GPUs

### Q: What is Ulysses sequence parallelism?
**A:** Splits attention across GPUs:
```
Single GPU:
  Attention heads: [H1, H2, H3, H4, H5, H6, H7, H8]
  ↓
  All 8 heads on one GPU

Ulysses (4 GPUs):
  GPU 0: [H1, H2]
  GPU 1: [H3, H4]
  GPU 2: [H5, H6]
  GPU 3: [H7, H8]
  
  Result: 4× memory reduction per GPU
```

### Q: What is classifier-free guidance (CFG)?
**A:** A technique to make the model follow the prompt better:
```python
# Two forward passes per step:
output_cond = model(latent, text, timestep)      # With prompt
output_uncond = model(latent, "", timestep)      # Without prompt

# Combine with guidance scale (e.g., 4.0):
final = output_uncond + 4.0 * (output_cond - output_uncond)

# Effect: Pushes output further in the "prompted" direction
```

### Q: How does expert switching work?
**A:** Based on timestep `t`:
```python
if t >= 0.875:
    active_expert = high_noise_model  # Early steps
else:
    active_expert = low_noise_model   # Late steps

# Only ONE expert is active per timestep
# Both experts share the same input/output format
# Switch happens seamlessly during generation
```

---

## 🚀 Quick Start Examples

### Generate a simple video:
```bash
python generate.py \
  --task t2v-A14B \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --prompt "A cat walking in snow" \
  --size 1280*720
```

### Generate with multiple GPUs:
```bash
torchrun --nproc_per_node=8 generate.py \
  --task t2v-A14B \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --prompt "A cat walking in snow"
```

### Generate on consumer GPU (RTX 4090):
```bash
python generate.py \
  --task ti2v-5B \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --prompt "A cat walking in snow"
```

---

## 🎓 Learning Path

### Beginner (30 minutes)
1. Read this README
2. View **Wan2.2_Quick_Reference.mmd**
3. View **Wan2.2_Example_Walkthrough.mmd**
4. Run a simple generation example

### Intermediate (2 hours)
1. Study **Wan2.2_Complete_Architecture.mmd** (all 8 diagrams)
2. Read the main codebase:
   - `wan/text2video.py` - T2V pipeline
   - `wan/modules/model.py` - DiT architecture
   - `wan/modules/vae2_1.py` - VAE implementation
3. Experiment with different parameters

### Advanced (1 week)
1. Understand distributed training code:
   - `wan/distributed/fsdp.py` - Model sharding
   - `wan/distributed/ulysses.py` - Sequence parallelism
2. Study the MoE implementation in detail
3. Read the GRPO training guide (if available)
4. Try fine-tuning on custom data

---

## 🔗 Additional Resources

- **Main Repo:** https://github.com/Wan-Video/Wan2.2
- **Paper:** https://arxiv.org/abs/2503.20314
- **HuggingFace:** https://huggingface.co/Wan-AI/
- **ModelScope:** https://modelscope.cn/organization/Wan-AI
- **Discord:** https://discord.gg/AKNgpMK4Yj

---

## 📝 Summary

**Wan2.2 is a state-of-the-art video generation system featuring:**

✅ **5 specialized models** for different tasks (T2V, I2V, TI2V, S2V, Animate)  
✅ **MoE architecture** with 2 experts for efficiency  
✅ **27B parameters** but only 14B active (fast inference)  
✅ **Multi-GPU support** via FSDP and Ulysses  
✅ **Consumer GPU friendly** with TI2V-5B model  
✅ **720p @ 16-24fps** high-quality video output  
✅ **Open source** with Apache 2.0 license  

**Key Innovation:** Expert specialization allows 2× more parameters at the same compute cost, resulting in superior quality without sacrificing speed.

---

**Happy video generating! 🎬✨**

