# Lung CT Conditional DDPM (Mask + Class-Conditioned)

This implementation extends a basic lung CT DDPM with class conditioning.

The model:

- learns to predict diffusion noise on HU-windowed lung CT slices,
- is conditioned on:
  - a lung mask (anatomical constraint),
  - a binary class map (nodule present / absent),
- is trained with balanced batches of positive and negative slices,
- uses classifier-free guidance (CFG) at sampling time,
- is evaluated using PSNR, SSIM, MSE, FID, LPIPS.


During training:

1. A clean CT slice is windowed to a lung HU range, normalized to `[-1, 1]`.
2. A timestep `t` is sampled.
3. Gaussian noise is added to obtain `x_t` via the DDPM forward process.
4. The UNet receives:
   - lung mask,
   - class map (0 = no nodule, 1 = nodule),
   - noisy CT slice `x_t`,
   - and predicts the noise that was added.
5. The loss is MSE between predicted and true noise.

During sampling:

1. Start from pure Gaussian noise.
2. At each diffusion step, the model predicts noise in two modes:
   - conditional: with class map,
   - unconditional: with class map zeroed.
3. Combine them using **classifier-free guidance**:
   `ε = ε_uncond + CFG_SCALE * (ε_cond - ε_uncond)`.
4. The scheduler uses this noise estimate to move one step towards a clean CT slice.
5. After all steps, the final sample is a CT slice:
   - anatomically constrained by the lung mask,
   - configured as **nodule** or **no nodule** by the class map.


## Model Inputs and Outputs

### Input Tensor

The model input is a **3-channel tensor**:

1. **Lung mask** `(1, H, W)`  
   - binary mask of the lung region (resized to `IMG × IMG`),
   - constrains the spatial extent of generated content.

2. **Class map** `(1, H, W)`  
   - filled with 0.0 or 1.0 depending on the label (no-nodule / nodule),
   - provides global class conditioning in a spatial form.

3. **Noisy CT image `x_t`** `(1, H, W)`  
   - HU-windowed, normalized CT slice at diffusion timestep `t`.

After concatenation:

```
[B, 3, 128 128]  
```

### Output Tensor

The UNet outputs:

```
[B, 1, 128, 128]  
```

This output is the predicted noise that was added to the clean CT image to generate `x_t`.


## Architecture Overview

The model uses a **UNet2DModel** from `diffusers`:

```python
unet = UNet2DModel(
    sample_size        = IMG,
    in_channels        = 3,   # [lung_mask, class_map, x_t]
    out_channels       = 1,   # predicted noise
    block_out_channels = (32, 64, 96),
    down_block_types   = ("DownBlock2D",) * 3,
    up_block_types     = ("UpBlock2D",) * 3,
    layers_per_block   = 1,
)
sched = DDPMScheduler(num_train_timesteps=T_STEPS)
```

Key features:

- encoder–decoder UNet with 3 downsampling and upsampling levels,
- 3-channel input for mask conditioning and noisy image,
- 1-channel output for predicted noise,
- 1000-step DDPM noise schedule,
- EMA weights for improved sampling stability.

#Results:

<img width="287" height="578" alt="sample" src="https://github.com/user-attachments/assets/d8f03708-bd6e-4b61-bfc6-3f32fb411f4d" />

#Metrics:

<img width="600" height="362" alt="metrics" src="https://github.com/user-attachments/assets/6935f276-ceff-4545-a522-d66cfd531198" />

<img width="594" height="358" alt="Other metrics" src="https://github.com/user-attachments/assets/49d03e92-efc5-4399-ab58-958d0632d04a" />



