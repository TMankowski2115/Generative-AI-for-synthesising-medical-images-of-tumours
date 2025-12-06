# LIDC-IDRI Lung Nodule Diffusion (Vector Conditioned)

**Latent Diffusion Model (LDM)** for synthesizing lung CT scans with controllable nodules.
This model allows control over nodule **location, size, malignancy, and texture** via a continuous vector embedding mechanism.

## Results
<img width="256" height="256" alt="NOD1_x0.38_y0.69_sz0.05_mal0.75_tex0.75" src="https://github.com/user-attachments/assets/fc81ab40-27ea-430f-97ae-782c6da1fa5d" />
NOD1_x0.38_y0.69_sz0.05_mal0.75_tex0.75

<img width="256" height="256" alt="obraz" src="https://github.com/user-attachments/assets/5502dab1-7b02-495a-8950-8971baf1682b" />
NOD1_x0.81_y0.65_sz0.06_mal0.75_tex0.50


### Generation Quality (Diffusion Stage)
| Metric | Score |
| :--- | :--- |
| **FID (Fréchet Inception Distance)** | **12.778** |
### Reconstruction Fidelity (VAE Stage)
The Autoencoder plays a crucial role in preserving anatomical details.
| Metric | Score |
| :--- | :--- |
| **LPIPS (Similarity)** | **0.9787** |
| **MS-SSIM** | **0.9888** ± 0.004 |
| **MSE** | **3.01e-4** ± 1.67e-4 |

## 🏗️ Architecture

This project adapts the **MedFusion** framework to handle continuous vector.

### 1. Latent Space (VAE)
* **Backbone:** AutoencoderKL (trained on LIDC-IDRI slices).
* **Role:** Compresses 256x256 CT slices into a compact `32x32x8` latent representation.
* **Preprocessing:** Custom windowing `[-1000, 400] HU` normalized to `[-1, 1]`.

### 2. Denoising Model (UNet)
* **Architecture:** Standard UNet with Attention mechanisms (`use_attention='linear'`).
* **Input:** 8 latent channels (noisy image).
* **Conditioning:** **Global Vector Conditioning**.
    * **Input:** 6-dimensional vector:
      `[has_nodule, x, y, size, malignancy, texture]`
    * **Embedder:** Custom `ScalarEmbedder` (MLP) maps the vector to the `time_embedding` dimension (1024).
    * **Injection:** Added to the time embedding vector (Global Bias).

## Training Strategy (The Secret Sauce)

### Data Preparation
* **Dataset:** LIDC-IDRI (2D slices).
* **Balanced Sampling:** Both Training and Validation sets are balanced (60% nodules / 40% healthy) to prevent mode collapse and ensure the validation loss reflects the capability to generate pathologies.
* **Unfiltered Data:** The model uses slices containing multiple nodules (mapped to single vector targets) to maximize data volume and learn robust background anatomy.

### Stabilization Techniques
* **Attention Mechanism:** `use_attention='linear'` was enabled in the UNet to maintain global anatomical coherence (e.g., ensuring consistent lung boundaries) without the memory cost of full attention.

### Conditioning & Guidance
* **Classifier-Free Guidance (CFG):** Trained with `dropout=0.3`. This forces the model to learn both unconditional anatomy and conditional nodule generation.
* **Inference:** During sampling, a `guidance_scale=2.0` is used to mathematically enforce adherence to the input vector (size/position).
