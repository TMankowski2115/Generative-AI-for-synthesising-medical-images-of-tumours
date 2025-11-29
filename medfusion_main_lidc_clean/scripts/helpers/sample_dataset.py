import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
print("DBG ROOT_DIR:", ROOT_DIR)

import time
import math

import torch
import numpy as np
from PIL import Image

from medical_diffusion.models.pipelines import DiffusionPipeline


def chunks(lst, n):
    """Zwraca kolejne kawałki listy po n elementów."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def save_ct_sample(x, idx, out_dir, wl=-1000.0, wh=400.0):
    """
    x: tensor (1, H, W) w [-1, 1]
    Zapisuje:
      - PNG: w skali szarości (okno HU)
      - NPY: HU w oknie [wl, wh]
    """
    x = x.clamp(-1.0, 1.0)

    # [-1,1] -> [0,1]
    x01 = (x + 1.0) / 2.0  # (1,H,W)

    # ---------- PNG ----------
    img_uint8 = (x01 * 255.0).cpu().numpy().astype(np.uint8)  # (1,H,W)
    img_uint8 = img_uint8[0]  # (H,W)

    img = Image.fromarray(img_uint8, mode="L")
    img.save(out_dir / f"sample_{idx:05d}.png")

    # ---------- NPY w HU (z zachowaniem okna) ----------
    hu = x01.cpu().numpy() * (wh - wl) + wl  # (1,H,W) w HU
    np.save(out_dir / f"sample_{idx:05d}_hu.npy", hu.astype(np.float32))


if __name__ == "__main__":
    # ------------ Ustawienia ------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_ckpt = Path("runs/2025_11_19_160006_lidc_ct_diffusion/epoch=72-step=90000.ckpt")

    # ile próbek chcesz wygenerować
    n_samples = 64           # na start: 64; potem możesz dać np. 1000
    sample_batch = 16        # ile na raz generować (zależne od VRAM)
    steps_list = [100]   # różne liczby kroków sampler’a do porównania

    # latent shape – MUSI pasować do Twojego VAE:
    # VAE: emb_channels=8, downsample 256->32 -> (8,32,32)
    latent_shape = (8, 32, 32)

    # folder wyjściowy
    base_out = Path.cwd() / "synthetic_lidc_diffusion"
    base_out.mkdir(parents=True, exist_ok=True)

    # ------------ Wczytaj pipeline ------------
    print("Ładuję diffusion pipeline z:", diffusion_ckpt)
    pipeline = DiffusionPipeline.load_from_checkpoint(str(diffusion_ckpt))
    pipeline.to(device)
    pipeline.eval()

    # ponieważ używaliśmy pipeline bez warunkowania:
    condition = None
    un_cond = None
    guidance_scale = 1.0  # i tak CFG nic nie robi, bo nie ma cond_embeddera

    # ------------ Generowanie ------------
    for steps in steps_list:
        out_dir = base_out / f"steps_{steps}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Generuję {n_samples} próbek (steps={steps}) do {out_dir} ===")

        torch.manual_seed(0)
        counter = 0
        indices = list(range(n_samples))

        for chunk in chunks(indices, sample_batch):
            bsz = len(chunk)

            with torch.no_grad():
                # pipeline.sample:
                #  - batch_size
                #  - shape latentów (C,H,W) → tutaj (8,32,32)
                #  - guidance_scale, condition, un_cond, steps
                results = pipeline.sample(
                    bsz,
                    latent_shape,
                    guidance_scale=guidance_scale,
                    condition=condition,
                    un_cond=un_cond,
                    steps=steps,
                )

            # results: (B, C, H, W) w [-1,1], C=1 (po dekoderze VAE)
            if isinstance(results, torch.Tensor):
                imgs = results
            else:
                # na wszelki wypadek, gdyby pipeline zwrócił dict
                imgs = results[0]

            for b in range(imgs.size(0)):
                x = imgs[b]  # (C,H,W) = (1,H,W)
                save_ct_sample(x, counter, out_dir)
                counter += 1

        torch.cuda.empty_cache()
        time.sleep(2.0)

    print("\nGotowe. Zobacz folder:", base_out)
