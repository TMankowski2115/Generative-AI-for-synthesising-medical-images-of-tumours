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

# Metryki
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim

# Model i Dataset
from medical_diffusion.models.embedders.latent_embedders import VAE
# Importujemy Twój nowy dataset (Conditional)
from medical_diffusion.data.datasets.dataset_lidc_conditional import LIDCConditionalDataset

# ---------------- Settings ----------------
batch_size = 32
max_samples = None       # None = wszystkie
fraction = 1           # Ewaluacja na całym zbiorze testowym?
split = "test"           # 'val' lub 'test'

device = [2] if torch.cuda.is_available() else None

# Gdzie zapisywać logi
path_out = Path.cwd() / 'results' / 'LIDC_VAE_Conditional' / 'metrics'
path_out.mkdir(parents=True, exist_ok=True)

# ----------------- Logging -----------------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger("lidc_vae_eval")
logger.setLevel(logging.INFO)

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

# Używamy LIDCConditionalDataset
# Zwraca on {"source": [2, 256, 256], ...}
ds_real = LIDCConditionalDataset(root_dir=root_dir, split=split)

# Opcjonalne: ucięcie datasetu (fraction / max_samples handled later)
if fraction < 1.0:
    import random
    indices = random.sample(range(len(ds_real)), int(len(ds_real) * fraction))
    ds_real = torch.utils.data.Subset(ds_real, indices)

dm_real = DataLoader(
    ds_real,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False,
    drop_last=False,
)

logger.info(f"Samples Real (Eval set): {len(ds_real)}")

# --------------- Load Model ------------------
# !!! WAŻNE: Wpisz tutaj ścieżkę do SWOJEGO NOWEGO modelu !!!
ckpt_path = Path("/runs/2025_11_29_113618_lidc_vae_conditional_ready/loss=37652.2578.ckpt")

if not ckpt_path.exists():
    logger.warning(f"Checkpoint nie istnieje: {ckpt_path}. Sprawdź ścieżkę!")
else:
    logger.info(f"Ładuję VAE z: {ckpt_path}")

model = VAE.load_from_checkpoint(str(ckpt_path))
model.to(device)
model.eval()

# ------------- Init Metrics ----------------------
calc_lpips = LPIPS(net_type='alex').to(device) # net_type='alex' jest standardem, lżejszy

mmssim_list = []
mse_list = []

n_processed = 0

# --------------- Start Calculation -----------------
logger.info("Rozpoczynam pętlę ewaluacji...")

for batch in tqdm(dm_real, desc="Eval VAE LIDC"):
    # batch["source"] to [B, 2, H, W] -> (CT, Blob)
    x_input = batch["source"].to(device)
    
    # --- KLUCZOWY MOMENT ---
    # VAE oczekuje wejścia 1-kanałowego (samo CT).
    # Wycinamy kanał 0.
    x_real = x_input[:, 0:1, :, :] # -> [B, 1, H, W]
    
    # Obsługa max_samples
    if max_samples is not None:
        if n_processed >= max_samples:
            break
        if n_processed + x_real.size(0) > max_samples:
            x_real = x_real[: max_samples - n_processed]

    B = x_real.size(0)
    n_processed += B

    with torch.no_grad():
        # model(x) zwraca (recon, posterior, ...); bierzemy [0] = recon
        # Clampujemy wynik do [-1, 1], bo VAE czasem lekko przestrzeliwuje
        x_fake = model(x_real)[0].clamp(-1.0, 1.0)

    # ---------- LPIPS (na 3 kanałach) ----------
    # LPIPS wymaga inputu [-1, 1], więc jest OK.
    # Kopiujemy 1 kanał na 3 kanały RGB
    x_real_3 = x_real.repeat(1, 3, 1, 1)
    x_fake_3 = x_fake.repeat(1, 3, 1, 1)
    
    calc_lpips.update(x_real_3, x_fake_3)

    # ---------- MS-SSIM + MSE ----------
    # Te metryki wolą zakres [0, 1]
    x_real_01 = (x_real + 1.0) / 2.0
    x_fake_01 = (x_fake + 1.0) / 2.0

    # SSIM liczymy per sample, żeby uśrednić
    for i in range(B):
        img_real = x_real_01[i:i+1] # [1, 1, H, W]
        img_fake = x_fake_01[i:i+1]

        # normalize='relu' w mmssim czasem pomaga uniknąć NaN przy czarnych tłach
        try:
            val_ssim = mmssim(img_real, img_fake, data_range=1.0)
            mmssim_list.append(val_ssim)
        except Exception as e:
            # Zabezpieczenie na wypadek błędu MS-SSIM na bardzo małych/czarnych obrazkach
            pass

        val_mse = torch.mean((img_real - img_fake) ** 2)
        mse_list.append(val_mse)

    if max_samples is not None and n_processed >= max_samples:
        break

logger.info(f"Przetworzono łącznie: {n_processed} próbek")

# -------------- Summary -------------------
if len(mmssim_list) > 0:
    mmssim_list = torch.stack(mmssim_list)
    ms_ssim_mean = torch.mean(mmssim_list).item()
    ms_ssim_std = torch.std(mmssim_list).item()
else:
    ms_ssim_mean = 0.0; ms_ssim_std = 0.0

mse_list = torch.stack(mse_list)
mse_mean = torch.mean(mse_list).item()
mse_std = torch.std(mse_list).item()

# Compute LPIPS (zwraca dystans, więc 0 = identyczne)
# Czasem podaje się "LPIPS Score" jako 1 - distance (podobieństwo),
# ale standardem naukowym jest "LPIPS Distance" (im mniej tym lepiej).
# W Twoim poprzednim skrypcie liczyłeś (1 - LPIPS), czyli "Similarity".
lpips_dist = calc_lpips.compute().item()
lpips_similarity = 1.0 - lpips_dist

logger.info(f"LPIPS Distance (im mniej tym lepiej): {lpips_dist:.4f}")
logger.info(f"LPIPS Similarity (1 - dist): {lpips_similarity:.4f}")
logger.info(f"MS-SSIM: {ms_ssim_mean:.4f} ± {ms_ssim_std:.4f}")
logger.info(f"MSE: {mse_mean:.6f} ± {mse_std:.6f}")

print("===================================")
print("   WYNIKI VAE (Conditional Dataset)")
print("===================================")
print(f"Próbki: {n_processed}")
print(f"LPIPS Similarity (Higher is better): {lpips_similarity:.4f}")
print(f"LPIPS Distance   (Lower is better):  {lpips_dist:.4f}")
print(f"MS-SSIM          (Higher is better): {ms_ssim_mean:.4f} ± {ms_ssim_std:.4f}")
print(f"MSE              (Lower is better):  {mse_mean:.6f}")
print("===================================")