import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print("DBG ROOT_DIR:", ROOT_DIR)

import logging
from datetime import datetime
from tqdm import tqdm
import cv2

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torchmetrics.functional import structural_similarity_index_measure as ssim_tm
from torchmetrics.functional import peak_signal_noise_ratio as psnr_tm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from medical_diffusion.models.embedders.latent_embedders import VAE


# ======================= USTAWIENIA ===========================
# Ścieżki do NPY z HU:
#  - realne CT: np. ct_hu.npy z Twojego datasetu
#  - fake CT: sample_XXXXX_hu.npy wygenerowane przez dyfuzję
REAL_ROOT = Path("../dataset_lidc_2d_seg") / "slices"
FAKE_ROOT = Path("synthetic_lidc_diffusion") / "steps_100"  # <- podmień np. na steps_50 / steps_200

# Checkpoint Twojego VAE:
VAE_CKPT = Path("runs/2025_11_18_222802_lidc_ct_vae/epoch=45-step=57362.ckpt")

# Okno HU
WL, WH = -1000.0, 400.0

BATCH_SIZE = 32
MAX_SAMPLES = 1000  # None = wszystkie, albo np. 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Gdzie zapisać logi
OUT_DIR = Path.cwd() / "results" / "LIDC_CT_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================= LOGOWANIE ============================
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger("ct_npy_metrics")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler(OUT_DIR / f"metrics_{current_time}.log", mode="w")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info(f"Device: {DEVICE}")


# ======================= DATASET PAR NPY ======================
class RealFakeNPYDataset(Dataset):
    """
    Dataset zwracający pary (real_ct, fake_ct), oba jako HU [float32].
    Real:
      REAL_ROOT/**/ct_hu.npy
    Fake:
      FAKE_ROOT/*_hu.npy

    Łączy po indeksie:
      0..min(len(real), len(fake))-1
    """

    def __init__(self, real_root: Path, fake_root: Path, max_samples: int | None = None):
        super().__init__()
        # real: szukamy wszystkich ct_hu.npy w strukturze LIDC
        # bierzemy tylko slajsy NEGATYWNE
        real_files = sorted([
            p for p in real_root.rglob("ct_hu.npy")
            if p.parent.name.endswith("_neg")
        ])
        # fake: sample_XXXXX_hu.npy z generatora
        fake_files = sorted(fake_root.glob("*_hu.npy"))

        n = min(len(real_files), len(fake_files))
        if max_samples is not None:
            n = min(n, max_samples)

        self.real_files = real_files[:n]
        self.fake_files = fake_files[:n]

        logger.info(f"Real CT NPY: {len(self.real_files)}")
        logger.info(f"Fake CT NPY: {len(self.fake_files)}")

    def __len__(self):
        return len(self.real_files)

    def __getitem__(self, idx):
        real_path = self.real_files[idx]
        fake_path = self.fake_files[idx]

        real_hu = np.load(real_path).astype(np.float32)  # może być (H,W) albo (1,H,W)
        fake_hu = np.load(fake_path).astype(np.float32)  # często (1,H,W)

        # --- spłaszczenie nadmiarowych wymiarów ---
        real_hu = np.squeeze(real_hu)  # np. (1,512,512) -> (512,512)
        fake_hu = np.squeeze(fake_hu)  # np. (1,256,256) -> (256,256)

        # sanity-check: oczekujemy 2D
        if real_hu.ndim != 2 or fake_hu.ndim != 2:
            raise RuntimeError(
                f"Spodziewam się 2D po squeeze, ale mam "
                f"real {real_hu.shape}, fake {fake_hu.shape} dla idx={idx}"
            )

        # --- window HU -> [WL, WH] ---
        real_win = np.clip(real_hu, WL, WH)
        fake_win = np.clip(fake_hu, WL, WH)

        # --- [WL,WH] -> [0,1] ---
        real_01 = (real_win - WL) / (WH - WL)
        fake_01 = (fake_win - WL) / (WH - WL)

        # --- resize do wspólnego rozmiaru, np. 256x256 ---
        IMG_SIZE = 256
        real_01 = cv2.resize(real_01, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        fake_01 = cv2.resize(fake_01, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # --- do tensorów: (1,H,W) ---
        real_t = torch.from_numpy(real_01).unsqueeze(0).float()  # (1,256,256)
        fake_t = torch.from_numpy(fake_01).unsqueeze(0).float()  # (1,256,256)

        return real_t, fake_t


# ======================= FUNKCJA LATENTÓW =====================
def get_latent_from_vae(model: VAE, x: torch.Tensor) -> torch.Tensor:
    """
    Zwraca pojedynczy tensor latentu z VAE dla wejścia x (B,1,H,W) w [-1,1].

    Obsługuje typowe wyjścia Medfusion:
      - model(x) -> (recon, emb, mu, log_var)
      - emb może być tensorem albo listą/tuplą tensorów (deep supervision).
    """
    model.eval()
    with torch.no_grad():
        out = model(x)

    # Przypadek 1: forward zwraca dict z kluczem 'emb'
    if isinstance(out, dict):
        if "emb" in out:
            z = out["emb"]
        else:
            raise RuntimeError(f"VAE.forward() zwrócił dict bez klucza 'emb': {out.keys()}")
    # Przypadek 2: forward zwraca tuple/list (recon, emb, mu, logvar, ...)
    elif isinstance(out, (list, tuple)):
        if len(out) < 2:
            raise RuntimeError(f"VAE.forward() zwrócił zbyt krótki tuple/list: len={len(out)}")
        z = out[1]   # to zwykle 'emb' w Medfusion
    else:
        raise RuntimeError(
            f"Nieoczekiwany typ wyjścia z VAE.forward(): {type(out)}. "
            "Dopasuj get_latent_from_vae do swojego modelu."
        )

    # Jeśli emb jest listą/tuplą (np. kilka poziomów), bierzemy ostatni
    if isinstance(z, (list, tuple)):
        if len(z) == 0:
            raise RuntimeError("Lista latentów 'emb' jest pusta.")
        z = z[-1]   # np. najgłębszy poziom

    if not torch.is_tensor(z):
        raise RuntimeError(f"Oczekiwano tensora jako latent, ale dostaliśmy: {type(z)}")

    return z



# ======================= FRECHET DISTANCE ======================
def compute_frechet_distance(mu1, cov1, mu2, cov2):
    """
    Fréchet Distance między dwoma rozkładami Gaussa:
      N(mu1, cov1) i N(mu2, cov2)
    FD = ||mu1-mu2||^2 + Tr(cov1+cov2 - 2*sqrt(cov1*cov2))

    Zakładamy tensory torch na tym samym device.
    """
    diff = mu1 - mu2
    diff_norm_sq = torch.sum(diff * diff)

    # Produkt macierzy kowariancji
    cov_prod = cov1 @ cov2
    # Symetryzacja (na wszelki wypadek numeryczny)
    cov_prod = 0.5 * (cov_prod + cov_prod.T)

    # EIG decomp do sqrtm
    eigvals, eigvecs = torch.linalg.eigh(cov_prod)
    # ujemne eigenvalues ucinamy do zera
    eigvals = torch.clamp(eigvals, min=0.0)
    sqrt_cov_prod = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

    trace_cov1 = torch.trace(cov1)
    trace_cov2 = torch.trace(cov2)
    trace_sqrt = torch.trace(sqrt_cov_prod)

    fd = diff_norm_sq + trace_cov1 + trace_cov2 - 2.0 * trace_sqrt
    return fd


# ======================= GŁÓWNY KOD ============================
if __name__ == "__main__":
    # -------- Dataset / Loader ----------
    dataset = RealFakeNPYDataset(REAL_ROOT, FAKE_ROOT, max_samples=MAX_SAMPLES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    # -------- Metryki obrazowe ----------
    lpips_metric = LPIPS().to(DEVICE)

    ssim_vals = []
    psnr_vals = []

    # Zbierz latent features
    logger.info("Ładuję VAE do latentów...")
    vae = VAE.load_from_checkpoint(str(VAE_CKPT))
    vae.to(DEVICE)
    vae.eval()

    real_latents_list = []
    fake_latents_list = []

    logger.info("Start liczenia metryk obrazowych i latentów...")
    for real_batch, fake_batch in tqdm(loader, desc="Batches"):
        # real_batch, fake_batch: (B,1,H,W) w [0,1]
        real_01 = real_batch.to(DEVICE)
        fake_01 = fake_batch.to(DEVICE)

        # ---- SSIM/PSNR (w [0,1]) ----
        # torchmetrics funkcje przyjmują (B,C,H,W)
        ssim_val = ssim_tm(real_01, fake_01, data_range=1.0)
        psnr_val = psnr_tm(real_01, fake_01, data_range=1.0)

        ssim_vals.append(ssim_val.detach().cpu())
        psnr_vals.append(psnr_val.detach().cpu())

        # ---- LPIPS (w [-1,1], 3 kanały) ----
        real_m1p1 = real_01 * 2.0 - 1.0  # [0,1] -> [-1,1]
        fake_m1p1 = fake_01 * 2.0 - 1.0

        real_3 = real_m1p1.repeat(1, 3, 1, 1)
        fake_3 = fake_m1p1.repeat(1, 3, 1, 1)

        lpips_metric.update(real_3, fake_3)

        # ---- Latenty VAE (też w [-1,1]) ----
        with torch.no_grad():
            z_real = get_latent_from_vae(vae, real_m1p1)  # (B,C,h,w)
            z_fake = get_latent_from_vae(vae, fake_m1p1)

        # zamiast flattenowania C*H*W → global average pooling po (H,W)
        # z_real, z_fake: (B, C, h, w)
        z_real_gap = z_real.mean(dim=(2, 3))  # (B, C)
        z_fake_gap = z_fake.mean(dim=(2, 3))  # (B, C)

        z_real_flat = z_real_gap.detach().cpu()  # (B, C)
        z_fake_flat = z_fake_gap.detach().cpu()  # (B, C)

        real_latents_list.append(z_real_flat)
        fake_latents_list.append(z_fake_flat)


    # --------- Obrazowe: podsumowanie ----------
    ssim_vals = torch.stack(ssim_vals)  # (num_batches,)
    psnr_vals = torch.stack(psnr_vals)

    ssim_mean = ssim_vals.mean().item()
    ssim_std = ssim_vals.std().item()
    psnr_mean = psnr_vals.mean().item()
    psnr_std = psnr_vals.std().item()

    lpips_val = lpips_metric.compute().item()  # im mniejsze, tym lepiej

    logger.info(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    logger.info(f"PSNR: {psnr_mean:.2f} dB ± {psnr_std:.2f}")
    logger.info(f"LPIPS: {lpips_val:.4f} (mniej = lepiej)")

    # --------- Latenty: Fréchet Distance -------
    real_latents = torch.cat(real_latents_list, dim=0)  # (N, D)
    fake_latents = torch.cat(fake_latents_list, dim=0)  # (N, D)

    logger.info(f"Latent real shape: {real_latents.shape}")
    logger.info(f"Latent fake shape: {fake_latents.shape}")

    # mean/cov
    mu_real = real_latents.mean(dim=0)
    mu_fake = fake_latents.mean(dim=0)

    # rowvar=False => kowariancja po cechach
    cov_real = torch.from_numpy(np.cov(real_latents.numpy(), rowvar=False)).float()
    cov_fake = torch.from_numpy(np.cov(fake_latents.numpy(), rowvar=False)).float()

    fd_val = compute_frechet_distance(mu_real, cov_real, mu_fake, cov_fake).item()

    logger.info(f"Fréchet Distance (latent VAE): {fd_val:.4f}")

    # --------- Wydruk na koniec ----------
    print("========================================")
    print("  METRYKI CT (NPY → NPY, obraz + latent)")
    print("========================================")
    print(f"SSIM:  {ssim_mean:.4f} ± {ssim_std:.4f}")
    print(f"PSNR:  {psnr_mean:.2f} dB ± {psnr_std:.2f}")
    print(f"LPIPS: {lpips_val:.4f} (mniej = lepiej)")
    print(f"FD (latent VAE): {fd_val:.4f}")
