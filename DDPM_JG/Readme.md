# Lung CT Conditional DDPM

The model reverses the diffusion process by learning to predict the noise added to a CT slice at any timestep **t**.  
During sampling, this enables the following procedure:

1. Start from pure random noise  
2. Iteratively apply the model  
3. Gradually remove noise using the DDPM scheduler  
4. Obtain a CT slice consistent with the provided anatomical masks  

## Model Architecture

The model is based on the `UNet2DModel` from the `diffusers` library:

```python
unet = UNet2DModel(
    sample_size        = 128,
    in_channels        = 3,   # [lung mask, nodule mask, x_t]
    out_channels       = 1,   # predicted noise
    block_out_channels = (32, 64, 128),
    down_block_types   = ("DownBlock2D",)*3,
    up_block_types     = ("UpBlock2D",)*3,
    layers_per_block   = 1,
)
```

- The model input has **3 channels**:
  1. lung mask,  
  2. nodule mask,  
  3. noisy CT image `x_t`.  

- The output is a single channel representing the predicted noise for timestep **t**.

A `DDPMScheduler` with `num_train_timesteps = 1000` is used to perform both forward and reverse diffusion.

## Model Input

The model receives a **3-channel tensor** consisting of:

### **1. Lung mask**
A binary map *(H, W)* describing the lung region.  
It enforces the global anatomical structure of the generated CT slice.

### **2. Nodule mask**
A binary map *(H, W)* defining the location, size, and shape of the lesion.  
It allows the generator to place a nodule exactly where specified.

### **3. Noisy CT image `x_t`**
A CT slice corrupted with diffusion noise at timestep **t**.  
This input is essential for training the model to predict the added noise as required by DDPM.

After concatenation, the input tensor has shape:

```
[B, 3, 128, 128]
```

## Model Output

The model produces a single-channel tensor of shape:

```
[B, 1, 128, 128]
```

This output represents the **predicted noise** that was added to the clean CT image to obtain the noisy image `x_t` at timestep **t**.

In the DDPM framework, the UNet does **not generate CT images directly.  
Instead, it learns to reconstruct the noise component, allowing the DDPM scheduler to iteratively denoise the sample during the reverse diffusion process.

# Results:
<img width="522" height="550" alt="samples_ep046" src="https://github.com/user-attachments/assets/e6109c87-f222-406b-957f-4483adcd32b3" />

## Metrics:
| Region        | PSNR [dB] – mean | PSNR [dB] – std | SSIM – mean | SSIM – std |
|---------------|------------------|------------------|--------------|-------------|
| Whole image   | 15.1070          | 2.8078           | 0.07110      | 0.03889     |
