import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print("DBG ROOT_DIR:", ROOT_DIR)

import logging
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim

from medical_diffusion.models.embedders.latent_embedders import VAE
from medical_diffusion.data.datasets.dataset_lidc_ct_2d import SimpleDataset2D

# ---------------- Settings ----------------
batch_size = 32          # możesz podbić / zmniejszyć w zależności od VRAM
max_samples = 1000       # None = wszystkie, np. 1000 na szybki test
fraction = 0.10          # np. 10% walidacji; możesz dać 1.0, jak chcesz całość

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# gdzie zapisywać logi
path_out = Path.cwd() / 'results' / 'LIDC_CT_VAE' / 'metrics'
path_out.mkdir(parents=True, exist_ok=True)

# ----------------- Logging -----------------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger("lidc_vae_eval")
logger.setLevel(logging.INFO)

# czyścimy poprzednie handlery, żeby nie powielać logów
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler(path_out / f'metrics_{current_time}.log', mode='w')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

logger.info(f"Device: {device}")

# --------------- Dataset / DataLoader ----------------
root_dir = Path("../dataset_lidc_2d_seg")

# użyjemy splitu 'val' do ewaluacji (bez guzków, jak w definicji SimpleDataset2D)
ds_real = SimpleDataset2D(root_dir=root_dir, split="test", fraction=fraction)

# prosty DataLoader; dataset zwraca dict: {"source": x, "label": y}
dm_real = DataLoader(
    ds_real,
    batch_size=batch_size,
    num_workers=0,          # na Windows często 0 jest najstabilniejsze
    shuffle=False,
    drop_last=False,
)

logger.info(f"Samples Real (po fraction): {len(ds_real)}")

# --------------- Load Model ------------------
# <- PODMIEŃ ŚCIEŻKĘ NA SWÓJ NAJLEPSZY CHECKPOINT
ckpt_path = Path("runs/2025_11_18_222802_lidc_ct_vae/epoch=45-step=57362.ckpt")
logger.info(f"Ładuję VAE z: {ckpt_path}")

model = VAE.load_from_checkpoint(str(ckpt_path))
model.to(device)
model.eval()

# ------------- Init Metrics ----------------------
# LPIPS wymaga 3 kanałów, więc powielimy grayscale CT (1->3)
calc_lpips = LPIPS().to(device)

mmssim_list = []
mse_list = []

n_processed = 0

# --------------- Start Calculation -----------------
for batch in tqdm(dm_real, desc="Eval VAE LIDC"):
    x_real = batch["source"].to(device)  # (B,1,H,W), w [-1,1]

    # obsługa max_samples
    if max_samples is not None:
        if n_processed >= max_samples:
            break
        if n_processed + x_real.size(0) > max_samples:
            x_real = x_real[: max_samples - n_processed]

    B = x_real.size(0)
    n_processed += B

    with torch.no_grad():
        # model(x) zwraca (recon, ..); bierzemy pierwsze wyjście
        x_fake = model(x_real)[0].clamp(-1, 1)

    # ---------- LPIPS (na 3 kanałach) ----------
    # LPIPS z torchmetrics zwykle zakłada 3 kanały; więc powtarzamy (1->3)
    x_real_3 = x_real.repeat(1, 3, 1, 1)
    x_fake_3 = x_fake.repeat(1, 3, 1, 1)
    calc_lpips.update(x_real_3, x_fake_3)  # zakłada zakres [-1,1]

    # ---------- MS-SSIM + MSE ----------
    # Konwertujemy do [0,1] dla SSIM/PSNR/MSE
    x_real_01 = (x_real + 1.0) / 2.0
    x_fake_01 = (x_fake + 1.0) / 2.0

    for i in range(B):
        img_real = x_real_01[i:i+1]  # (1,1,H,W)
        img_fake = x_fake_01[i:i+1]

        mmssim_val = mmssim(img_real, img_fake, normalize='relu')
        mmssim_list.append(mmssim_val)

        mse_val = torch.mean((img_real - img_fake) ** 2)
        mse_list.append(mse_val)

    if max_samples is not None and n_processed >= max_samples:
        break

logger.info(f"Przetworzono łącznie: {n_processed} próbek")

# -------------- Summary -------------------
mmssim_list = torch.stack(mmssim_list)
mse_list = torch.stack(mse_list)

lpips_score = 1.0 - calc_lpips.compute()

logger.info(f"LPIPS Score (1 - LPIPS): {lpips_score}")
logger.info(f"MS-SSIM: {torch.mean(mmssim_list)} ± {torch.std(mmssim_list)}")
logger.info(f"MSE: {torch.mean(mse_list)} ± {torch.std(mse_list)}")

print("===================================")
print("  WYNIKI VAE NA LIDC (NEGATYWY)")
print("===================================")
print(f"LPIPS (1 - LPIPS): {lpips_score.item():.4f}")
print(f"MS-SSIM: {mmssim_list.mean().item():.4f} ± {mmssim_list.std().item():.4f}")
print(f"MSE: {mse_list.mean().item():.6f} ± {mse_list.std().item():.6f}")
