# Lung CT Conditional Stable Diffusion (LIDC + ControlNet)

This project fine-tunes **Stable Diffusion 1.5** on lung CT slices from the **LIDC** dataset and extends it with **ControlNet** conditioned on **tumor masks**. The goal is to generate realistic CT slices where both:
- global lung appearance (texture, contrast, artifacts) and
- local nodule presence (position, shape, size)
are controlled by a textual prompt and an explicit spatial conditioning signal.

---

## 1. Data and conditioning signals

**Dataset**
- Source: LIDC (preprocessed to 2D axial CT slices): [Dataset] https://www.kaggle.com/datasets/jokerak/lidcidri
- Resolution used in training: **512 × 512** RGB (CT slices normalized and mapped to [-1, 1]).
- CSV file `captions_lidc_lora.csv` contains:
  - `file_name`: image file in `LIDC/image/` (e.g. `LIDC_2338.png`),
  - `prompt`: textual description including nodule presence, location and type.

**Conditioning modalities**
1. **Text prompt (caption)** – describes:
   - lung CT modality,
   - presence/absence of a nodule,
   - approximate location in normalized coordinates (x, y),
   - sometimes nodule type (solid, subsolid, etc.).
2. **Tumor mask (for ControlNet)** – binary mask for each CT slice:
   - stored as `LIDC_Mask_XXXX.png` in `LIDC/mask/`,
   - used as input to ControlNet (3-channel, normalized to [-1, 1]),
   - encodes shape and exact position of the lesion.

---

## 2. Model overview

### 2.1 Base model: Stable Diffusion 1.5

The starting point is `runwayml/stable-diffusion-v1-5`:
- Text encoder, tokenizer, and VAE are **frozen**.
- Only selected UNet blocks are fine-tuned on LIDC:
  - `down_blocks.3` (deepest downsampling block),
  - `mid_block` (bottleneck),
  - `up_blocks.2` and `up_blocks.3` (two highest-resolution upsampling blocks).

### 2.2 Fine-tuned UNet (text → CT)

The fine-tuned UNet learns a mapping:
- input:
  - noisy latent,
  - diffusion timestep,
  - text embedding of the prompt,
- output:
  - predicted noise (DDPM objective).

Effectively, it learns to reconstruct CT-like images consistent with the textual description (lung-view, nodule presence and location).

### 2.3 ControlNet on tumor masks

On top of the fine-tuned UNet, a **ControlNet** is trained:
- initialized from `lllyasviel/sd-controlnet-canny`,
- attached to the same UNet as in the fine-tuned SD,
- only ControlNet weights are trainable (UNet remains frozen).

ControlNet receives as additional input:
- the same noisy latent,
- the same timestep and text embedding,
- a **tumor mask** image as conditioning.
---

## 3. Training setup

### 3.1 UNet fine-tuning

- Base: SD 1.5 UNet.
- Optimizer: **AdamW**, LR ≈ 5e-6, cosine LR schedule, weight decay.
- Steps: up to **20k global steps** (with gradient accumulation).
- Loss: **MSE** on noise in latent space.
- Batch:
  - physical batch size = 2,
  - gradient accumulation factor = 4 → effective batch = 8.
- Validation:
  - same loss (MSE) computed on fixed val split,

### 3.2 ControlNet training (tumor masks)

On top of the fine-tuned UNet, a **ControlNet** is trained:
- base: fine-tuned SD 1.5 pipeline (UNet + VAE + text encoder),
- ControlNet initialized from canny-based ControlNet model,
- only ControlNet weights are updated, UNet is frozen.

Training setup:
- Optimizer: AdamW, LR ≈ 5e-6, cosine LR schedule.
- Steps: **5k global steps** (with gradient accumulation).
- Objective: DDPM noise prediction (MSE) conditioned on prompt + tumor mask.

---

## 4. Evaluation protocol

Three quantitative metrics are used on the **validation split**:

1. **PSNR [dB]**
   - pixel-wise metric comparing original CT slice and generated image.

2. **SSIM**
   - structural similarity (perceptual similarity, luminance/contrast/structure).

3. **FID**
   - Frechet Inception Distance between real CT slices and generated CT images.
   - computed in a **streaming** way using `torchmetrics.image.FrechetInceptionDistance`:

All metrics are computed on the same val split with a fixed random seed to make comparisons fair.

### 4.1 Quantitative results (example)

On a validation subset of ~200 slices:

| Model                         | PSNR [dB] | SSIM   | FID    |
|------------------------------|-----------|--------|--------|
| SD 1.5 baseline              |   ~8.7    | ~0.17  | ~239   |
| SD 1.5 fine-tuned (wide UNet)|  ~10.5    | ~0.31  |  ~96   |
| SD 1.5 FT + ControlNet (mask)|  ~10.8    | ~0.34  | ~120   |

Interpretation:
- Fine-tuning SD1.5 significantly improves similarity and distribution match to LIDC CT slices.
- Adding ControlNet with tumor masks improves **structural alignment** (SSIM) and local lesion control, at the cost of slightly worse global FID (model becomes more constrained and less diverse).

## 5. Qualitative experiments

Several visualization experiments are used to inspect behaviour of the model.

### 5.1 GT vs Baseline FT vs ControlNet+FT

For selected validation cases:

1. Show **ground-truth CT**.
2. Show **conditioning mask** (tumor region).
3. Generate image with **fine-tuned SD (prompt only)**.
4. Generate image with **ControlNet+FT (prompt + mask)**.

Observation:
- Fine-tuned SD captures general CT style and approximate nodule presence,
- ControlNet+FT better matches **position and extent** of the nodule defined by the mask.

<img width="1575" height="409" alt="pobrane (4)" src="https://github.com/user-attachments/assets/b260d9ec-1caa-4255-9c01-9bc16763a3c8" />


### 5.2 Shuffled masks (foreign mask test)

Test whether ControlNet really uses the mask:

- Choose sample A and sample B from val with masks.
- Use:
  - prompt from A, mask from A → generation 1,
  - prompt from A, mask from B → generation 2.
    

<img width="1976" height="410" alt="pobrane" src="https://github.com/user-attachments/assets/30802038-de1a-470b-9bf5-19fa260dd340" />

### 5.3 Multiple seeds (same mask, different noise)

- Fix prompt and tumor mask.
- Generate multiple samples with different random seeds.

<img width="2376" height="411" alt="pobrane (1)" src="https://github.com/user-attachments/assets/3c54deca-2b38-4782-8e6f-924f866d2d20" />


## 6. How to use the project

The project is organized around a Colab notebook with the following logical blocks:

1. **Environment & imports**
   - install required packages, patch `diffusers` import if needed.

2. **Data loading**
   - mount Google Drive,
   - load `captions_lidc_lora.csv`,
   - split into `train_df` and `val_df`,
   - create PyTorch datasets/dataloaders for:
     - text–image training,
     - mask-conditioned training.

3. **UNet fine-tuning**
   - load SD 1.5,
   - freeze VAE and text encoder,
   - unfreeze selected UNet blocks,
   - train up to 20k steps,
   - periodically save checkpoints and final `StableDiffusionPipeline`.

4. **ControlNet training**
   - load fine-tuned pipeline,
   - attach ControlNet initialized from `sd-controlnet-canny`,
   - train only ControlNet on tumor masks,
   - save `StableDiffusionControlNetPipeline` with FT UNet + ControlNet.

5. **Evaluation & visualizations**
   - compute PSNR/SSIM/FID for FT model,
   - compute PSNR/SSIM/FID for ControlNet+FT,
   - run qualitative tests (GT vs FT vs CN, shuffled masks, seeds),

To reuse the trained models outside Colab, copy directories:
- `sd15_ct_finetuned_unet_lidc*` – fine-tuned SD1.5 pipeline,
- `sd15_ct_finetuned_unet_lidc*_controlnet_mask` – pipeline with ControlNet,

and load them with `StableDiffusionPipeline.from_pretrained(...)` or
`StableDiffusionControlNetPipeline.from_pretrained(...)`.
