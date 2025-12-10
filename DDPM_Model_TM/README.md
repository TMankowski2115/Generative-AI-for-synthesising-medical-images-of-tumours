# Lung-Controlled Diffusion Model for Synthetic Nodule Generation

## Overview

This repository contains a complete pipeline for **diffusion-based generation of lung CT slices**, with explicit control over lung masks, nodule masks, and a **6-dimensional conditioning vector** describing synthetic nodule properties (including distance to pleura).  
The project uses a **ControlNet-enhanced UNet with Self-Attention** inside a **DDPM** framework and is built around preprocessed slices from the **LIDC-IDRI dataset**.

The system supports:

- Training a diffusion model conditioned on lung anatomy  
- Generating new CT slices with controllable synthetic nodules  
- Dataset handling (CT, lung masks, nodule masks, labels)  
- Tools for debugging, evaluation, visualization, and sampling  
- Oversampling for class-balanced training  
- Automatic preview grids + TensorBoard logging  

---

##  1. How the System Works

The pipeline follows the classical **DDPM (Denoising Diffusion Probabilistic Models)** approach:

1. **Forward diffusion:** Noise is added to clean CT slices inside lung regions.

2. **Reverse diffusion:** The UNet predicts noise at each timestep to reconstruct the image.

3. **Conditioning sources:**
   - 2-channel lung masks (left / right)  
   - 1-channel nodule mask  
   - **6-dimensional vector** for precise control:  
     ```
     (cx, cy, radius, side_left, side_right, dist_pleura)
     ```
     *Note: `dist_pleura` allows controlling how close the nodule is to the chest wall.*

4. A **ControlNet module** injects anatomical information into the UNet’s skip connections.

5. **Self-Attention Mechanism:** Integrated into the UNet to improve the generation of fine-grained textures (lung parenchyma vs. nodule tissue).

6. **Classifier-free guidance (CFG):** Enhances controllability during sampling by mixing conditional and unconditional estimates.

---

##  2. Description of Key Modules & Methods

### **DiffusionSchedule - `diffusion_utils.py`**
- Builds linear beta schedules  
- Computes alphas, cumulative products, and `sqrt_betas`  
- `q_sample(x0, t)` - applies forward diffusion (adds noise to CT slices)

### **LIDCDiffusionDataset — `lidc_diffusion_dataset.py`**
Loads LIDC-derived slices and returns:
- CT image in `[-1, 1]` (HU normalized)
- Lung mask  
- Nodule mask  
- **Conditioning vector `[6]`** (includes normalized distance to pleura)  
- Optional text prompt (for compatibility)  
- Slice metadata  

### **LIDCControlNetUNet — `lidc_controlnet_model.py`**
UNet enhanced with:
- **Self-Attention Layers:** Newly added to capture global dependencies and improve texture realism.
- Timestep & Conditional embeddings  
- 3-channel ControlNet for `(left_lung, right_lung, nodule_mask)`  
- ResBlocks with skip connections  
- A final noise prediction head  

### **Sampling Methods (in `generate_nodule_user.py` & `train_diffusion.py`)**
Key functions:
- `ddpm_sample(...)` - reverse diffusion with CFG  
- `draw_nodule_mask_from_lung_labels(...)` - generates a synthetic nodule inside a lung  
- `draw_bbox_on_tensor(...)` - draws a visual bounding box for debugging  

### **Oversampling — `sampler_utils.py`**
- `make_oversampling_sampler(...)`: Increases sampling frequency for positive (tumor) slices (default factor=5) to handle dataset imbalance.

### **Training — `train_diffusion.py`**
Implements:
- Full DDPM training loop with **EMA (Exponential Moving Average)** for weight stabilization.
- **Top-k Checkpointing:** Saves the top 3 models based on FID score.
- Loss = `MSE(noise)` + `0.1 * SSIM loss`  
- Real-time evaluation: FID, Inception Score (IS), PSNR, SSIM  
- Automatic sample grids per epoch  

### **Debug Utilities**
- `test_dataset.py` - visualize dataset samples and masks  
- `test_model.py` -  verify model forward pass and tensor shapes  

---

## 3. Repository Structure
├── diffusion_utils.py          # Diffusion schedule, betas & forward noising

├── generate_nodule_user.py     # Main script for user-based nodule generation

├── lidc_controlnet_model.py    # ControlNet + UNet architecture with Self-Attention

├── lidc_diffusion_dataset.py   # Custom LIDC dataset loader (6-dim conditioning)

├── sampler_utils.py            # Oversampling utilities (WeightedRandomSampler)

├── test_dataset.py             # Dataset visualization/debugging

├── test_model.py               # Model forward-pass testing

└── train_diffusion.py          # Full training pipeline with EMA & Metrics

---

##  4. Disclaimer

Portions of the code were formatted, reorganized, or stylistically refined using **Generative AI tools**.
