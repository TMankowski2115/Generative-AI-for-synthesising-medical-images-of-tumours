# Lung-Controlled Diffusion Model for Synthetic Nodule Generation

## Overview

This repository contains a complete pipeline for **diffusion-based generation of lung CT slices**, with explicit control over lung masks, nodule masks, and a **5-dimensional conditioning vector** describing synthetic nodule properties.  
The project uses a **ControlNet-enhanced UNet** inside a **DDPM** framework and is built around preprocessed slices from the **LIDC-IDRI dataset**.

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

1. **Forward diffusion:**  
   Noise is added to clean CT slices inside lung regions.

2. **Reverse diffusion:**  
   The UNet predicts noise at each timestep.

3. **Conditioning sources:**
   - 2-channel lung masks (left / right)  
   - 1-channel nodule mask  
   - 5-dimensional vector:  
     ```
     (cx, cy, scaled_diameter, left_flag, right_flag)
     ```

4. A **ControlNet module** injects anatomical information into the UNet’s skip connections.

5. **Classifier-free guidance (CFG)** enhances controllability during sampling.

The user can specify the lung side and nodule diameter or generate fully synthetic nodules placed inside anatomically plausible regions.

---

##  2. Description of Key Modules & Methods

### **DiffusionSchedule - `diffusion_utils.py`**
- Builds linear beta schedules  
- Computes alphas, cumulative products, sqrt terms  
- `q_sample(x0, t)` - applies forward diffusion (adds noise to CT slices)

### **LIDCDiffusionDataset — `lidc_diffusion_dataset.py`**
Loads LIDC-derived slices and returns:
- CT image in `[-1, 1]`
- Lung mask  
- Nodule mask  
- Conditioning vector `[5]`  
- Optional text prompt  
- Slice metadata  
- “Has nodule” flag  

Handles HU normalization and correct interpolation.

### **LIDCControlNetUNet — `lidc_controlnet_model.py`**
UNet enhanced with:
- Timestep embeddings  
- Conditional embeddings  
- 3-channel ControlNet for `(left_lung, right_lung, nodule_mask)`  
- ResBlocks with skip connections  
- A final noise prediction head  

### **Sampling Methods (in `generate_nodule_user.py` & `train_diffusion.py`)**
Key functions:
- `ddpm_sample(...)` - reverse diffusion with CFG  
- `draw_nodule_mask_from_lung_labels(...)` - generates a synthetic nodule inside a lung  
- `draw_bbox_on_tensor(...)` - draws a visual bounding box for debugging  

### **Oversampling — `sampler_utils.py`**
- `make_oversampling_sampler(...)`: increases sampling frequency for positive (tumor) slices.

### **Training — `train_diffusion.py`**
Implements:
- Full DDPM training loop  
- Classifier-free dropout  
- Loss = `MSE(noise)` + `0.1 * SSIM loss`  
- Quick evaluation: FID, IS, PSNR, SSIM  
- Automatic sample grids per epoch  
- TensorBoard logging  
- Model checkpointing  

### **Debug Utilities**
- `test_dataset.py` - visualize dataset samples and masks  
- `test_model.py` -  verify model forward pass  

---

## 3. Repository Structure
├── diffusion_utils.py # Diffusion schedule & forward noising

├── generate_nodule_user.py # Main script for user-based nodule generation

├── lidc_controlnet_model.py # ControlNet + UNet architecture

├── lidc_diffusion_dataset.py # Custom LIDC dataset loader

├── sampler_utils.py # Oversampling utilities

├── test_dataset.py # Dataset visualization/debugging

├── test_model.py # Model forward-pass testing

└── train_diffusion.py # Full training pipeline


---

##  4. Disclaimer

Portions of the code were formatted, reorganized, or stylistically refined using **Generative AI tools**.  



