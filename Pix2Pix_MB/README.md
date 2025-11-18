# CT Lung Mask → Image GAN (U-Net + SSIM)

This project trains a **pix2pix (U-Net Generator + PatchGAN Discriminator)** model to generate CT lung slices based on masks (`lung_mask` / `combined_mask`).  
Additionally, **SSIM** is used in the loss function to improve preservation of vessel structures and overall image quality.

---

## Data Structure

Required directory structure:


dataset_lidc_masks/
  train/
    with_nodules/
      ..._slice_....png          # CT image (input to D, target for G)
      ..._combined_mask_....npy  # lung+nodule mask (input to G)
    clean/
      ..._slice_....png          # CT image without nodules
      ..._lung_mask_....npy      # lung mask (without nodules)
  test/
    with_nodules/
      ...
    clean/
      ...
	  
	  
Naming format:

- CT image:  
  `LIDC-IDRI-XXXX_slice_YYYY.png`
- lung mask:  
  `LIDC-IDRI-XXXX_lung_mask_YYYY.npy`
- lung + nodule mask:  
  `LIDC-IDRI-XXXX_combined_mask_YYYY.npy`


## Network Architecture

### Generator – U-Net

The generator is a 6-level **U-Net** that converts a 1-channel mask into a 1-channel CT image.

#### Encoder (downsampling)

- 6 blocks of `Conv2d(k=4, s=2)` → resolution reduction: 128 → 64 → 32 → … → 2
- Channels:  
  `1 → 64 → 128 → 256 → 512 → 512 → 512`
- Normalization: `InstanceNorm2d` from level 2  
- Activation: `LeakyReLU(0.2)`
- Bottleneck: size `2×2`

#### Decoder (upsampling)

- 5 blocks of `ConvTranspose2d(k=4, s=2)`
- Skip-connections (`concat`) with corresponding encoder layers
- Dropout (0.5) in the first 3 blocks
- Final layer: `Tanh` (output range `[-1, 1]`)

#### U-Net Advantages:

- **skip-connections preserve fine structures**, e.g., pulmonary vessels  
- enables reconstruction of stable details from early layers  
- handles medical images much better than a standard autoencoder

---

### Discriminator – PatchGAN

PatchGAN doesn't return a single number, but rather a **plausibility map** — each element of the map evaluates a local patch of the image.

Structure:

2 → 64 → 128 → 256 → 512 → 1

Conditions:

- Input is concatenation of `(mask, image)` → 2 channels
- Normalization: `InstanceNorm2d`
- Activation: `LeakyReLU(0.2)`
- Final map is smaller than 128×128

PatchGAN enforces:
- realistic CT textures,
- sharp edges,
- coherence of local structures.

---

## Loss Functions

### Generator minimizes:

L_G = L_GAN + L_L1 + L_SSIM

- **L_GAN** – BCEWithLogitsLoss(D(mask, fake), 1)
- **L1 Loss** – large coefficient (100) enforces pixel-wise agreement
- **SSIM Loss** – improves structure and local contrasts

### Discriminator minimizes:

L_D = 0.5 × (BCE(real, 1) + BCE(fake, 0))


---

## Hyperparameters

| Parameter       | Value |
|----------------|---------|
| IMG_SIZE        | 128     |
| BATCH_SIZE      | 64      |
| EPOCHS          | 200     |
| LR              | 0.0002  |
| BETAS           | (0.5, 0.999) |
| LAMBDA_L1       | 100     |
| LAMBDA_SSIM     | 5       |
| LR decay        | from epoch 100 |

---

## Results Logging

Every epoch logs:
- Loss_G
- Loss_D
- SSIM

Every 5 epochs:
- entry to `training_results_SSIM/training_log.txt`

Every 10 epochs:
- save grid of examples (`epoch_XX.png`)
- checkpoint: `checkpoint_epoch_XX.pth`
