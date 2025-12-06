import sys
from pathlib import Path

# Ustalanie ścieżki do roota
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print("DBG ROOT_DIR:", ROOT_DIR)

import logging
from datetime import datetime
from tqdm import tqdm
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Importy Metryk ---
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from medical_diffusion.metrics.torchmetrics_pr_recall import ImprovedPrecessionRecall

# ---------------- Settings ----------------
batch_size = 32
max_samples = None  # Ogranicznik, żeby FID nie liczył się wieki (dla LIDC mamy tysiące slajsów)

# Katalog z REALNYMI CT (Ground Truth)
LIDC_ROOT = ROOT_DIR / "../dataset_lidc_2d_seg"
REAL_SLICES_ROOT = LIDC_ROOT / "slices"

# Katalog z WYGENEROWANYMI obrazkami
FAKE_ROOT = Path.cwd() / "synthetic_only_results_4"  # <-- Tu celujemy w folder z wynikami generatora

# Gdzie zapisywać wyniki
path_out = Path.cwd() / "results" / "LIDC_Diffusion_Vector" / "metrics"
path_out.mkdir(parents=True, exist_ok=True)

# ----------------- Logging ----------------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger("lidc_eval")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler(path_out / f"metrics_{current_time}.log", mode="w")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


# -------------- Helpers ---------------------
def pil2torch_rgb_uint8(pil_img):
    """
    Konwersja PIL -> Tensor [3, H, W] uint8.
    FID wymaga 3 kanałów (RGB), więc powielamy grayscale.
    """
    # convert("RGB") automatycznie replikuje kanał L na R, G, B
    arr = np.array(pil_img.convert("RGB"))  
    return torch.as_tensor(arr).permute(2, 0, 1)


# ---------------- Datasety ----------------
class LIDCRealCTDataset(Dataset):
    """
    Czyta realne CT z datasetu.
    TERAZ: Bierze WSZYSTKIE slajsy (z guzkami i bez),
    aby reprezentowały pełną dystrybucję danych prawdziwych.
    """

    def __init__(self, slices_root: Path, max_samples: int | None = None):
        super().__init__()
        self.files = []
        
        # Rekurencyjne szukanie ct.png we wszystkich podkatalogach
        # slices/<pid>/<suid>/<slice>/ct.png
        # Używamy rglob dla szybkości
        logger.info(f"Skanowanie katalogu Real: {slices_root}")
        all_cts = sorted(list(slices_root.rglob("ct.png")))
        
        # Filtrujemy tylko te katalogi, które mają labels.json (dla pewności spójności)
        # (Można to pominąć dla szybkości, jeśli ufasz strukturze folderów)
        for ct_path in all_cts:
            if (ct_path.parent / "labels.json").exists():
                self.files.append(ct_path)
                
            if max_samples is not None and len(self.files) >= max_samples:
                break

        logger.info(f"[Real] Załadowano {len(self.files)} prawdziwych slajsów CT.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image
        path = self.files[idx]
        img = Image.open(path)
        x = pil2torch_rgb_uint8(img)  # (3,H,W) uint8
        return x


class FakeCTDataset(Dataset):
    """
    Czyta wygenerowane PNG z folderu wyników.
    """

    def __init__(self, fake_root: Path, max_samples: int | None = None):
        super().__init__()
        self.files = sorted(list(fake_root.glob("*.png")))
        
        if max_samples is not None:
            self.files = self.files[:max_samples]
            
        if len(self.files) == 0:
            logger.warning(f"UWAGA: Nie znaleziono żadnych plików PNG w {fake_root}")
            
        logger.info(f"[Fake] Załadowano {len(self.files)} syntetycznych slajsów.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image
        path = self.files[idx]
        img = Image.open(path)
        x = pil2torch_rgb_uint8(img)  # (3,H,W) uint8
        return x


# ==============================================================================
# GŁÓWNA LOGIKA
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # 1. Inicjalizacja Datasetów
    if not FAKE_ROOT.exists():
        logger.error(f"Katalog z fake'ami nie istnieje: {FAKE_ROOT}")
        sys.exit(1)

    ds_real = LIDCRealCTDataset(REAL_SLICES_ROOT, max_samples=max_samples)
    ds_fake = FakeCTDataset(FAKE_ROOT, max_samples=max_samples)

    if len(ds_fake) == 0:
        print("Brak obrazów wygenerowanych do oceny!")
        sys.exit(1)

    # Shuffle=True dla Real jest ważne przy PSNR/SSIM, żeby nie porównywać
    # ciągle tego samego pacjenta (jeśli dane są posortowane) z losowymi generacjami.
    dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)
    dm_fake = DataLoader(ds_fake, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False)

    # 2. Inicjalizacja Metryk
    # FID (Feature extractor: Inception v3)
    calc_fid = FID(feature=2048).to(device)
    
    # Precision / Recall (Manifold)
    calc_pr = ImprovedPrecessionRecall(splits_real=1, splits_fake=1).to(device)

    # PSNR / SSIM (Pixel-level stats)
    calc_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    calc_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # ---------------------------------------------------------
    # FAZA 1: Dystrybucja (FID, Precision, Recall)
    # ---------------------------------------------------------
    # Tutaj liczymy statystyki całego zbioru, nie parujemy obrazów.
    
    logger.info(">>> Obliczanie statystyk REAL...")
    for real_batch in tqdm(dm_real, desc="FID Real"):
        imgs = real_batch.to(device) # [B, 3, H, W] uint8
        calc_fid.update(imgs, real=True)
        calc_pr.update(imgs, real=True)

    logger.info(">>> Obliczanie statystyk FAKE...")
    for fake_batch in tqdm(dm_fake, desc="FID Fake"):
        imgs = fake_batch.to(device)
        calc_fid.update(imgs, real=False)
        calc_pr.update(imgs, real=False)

    # ---------------------------------------------------------
    # FAZA 2: Podobieństwo pikselowe (PSNR, SSIM)
    # ---------------------------------------------------------
    # Porównujemy losowe batche Real z losowymi batchami Fake.
    # To daje informację o ogólnej jakości sygnału (szum, kontrast).
    
    psnr_scores = []
    ssim_scores = []
    
    logger.info(">>> Obliczanie PSNR/SSIM (Random Pairing)...")
    
    # Iterujemy po mniejszym zbiorze (zazwyczaj fake jest mniejszy)
    min_len = min(len(dm_real), len(dm_fake))
    
    # Resetujemy loadery, żeby iterować od nowa
    iter_real = iter(dm_real)
    iter_fake = iter(dm_fake)
    
    # Liczba kroków
    steps = min(len(dm_real), len(dm_fake))

    for _ in tqdm(range(steps), desc="PSNR/SSIM"):
        try:
            real_batch = next(iter_real).to(device).float() / 255.0
            fake_batch = next(iter_fake).to(device).float() / 255.0
        except StopIteration:
            break

        # Wyrównanie wielkości batcha (ostatni batch może być mniejszy)
        b_size = min(real_batch.shape[0], fake_batch.shape[0])
        real_batch = real_batch[:b_size]
        fake_batch = fake_batch[:b_size]

        # Wyrównanie rozdzielczości (jeśli Real 512 != Fake 256)
        if real_batch.shape[-1] != fake_batch.shape[-1]:
            real_batch = F.interpolate(
                real_batch, 
                size=fake_batch.shape[-2:], 
                mode='bilinear', 
                align_corners=False, 
                antialias=True
            )

        # Update metryk
        batch_psnr = calc_psnr(fake_batch, real_batch)
        batch_ssim = calc_ssim(fake_batch, real_batch)
        
        psnr_scores.append(batch_psnr)
        ssim_scores.append(batch_ssim)

    # ---------------------------------------------------------
    # WYNIKI
    # ---------------------------------------------------------
    fid_score = calc_fid.compute()
    # precision, recall = calc_pr.compute()
    
    final_psnr = torch.stack(psnr_scores).mean() if psnr_scores else torch.tensor(0.0)
    final_ssim = torch.stack(ssim_scores).mean() if ssim_scores else torch.tensor(0.0)

    # Console Output
    print("\n" + "="*40)
    print(f" WYNIKI EWALUACJI (Mixed Nodules/Healthy)")
    print("="*40)
    print(f"Ilość próbek Real: {len(ds_real)}")
    print(f"Ilość próbek Fake: {len(ds_fake)}")
    print("-" * 40)
    print(f"FID (niższy = lepszy):     {fid_score.item():.4f}")
    # print(f"Precision (jakość):        {precision.item():.4f}")
    # print(f"Recall (różnorodność):     {recall.item():.4f}")
    print("-" * 40)
    print(f"PSNR (dB):                 {final_psnr.item():.4f}")
    print(f"SSIM:                      {final_ssim.item():.4f}")
    print("="*40)

    # JSON Save
    results = {
        "timestamp": current_time,
        "n_real": len(ds_real),
        "n_fake": len(ds_fake),
        "fid": fid_score.item(),
        # "precision": precision.item(),
        # "recall": recall.item(),
        "psnr": final_psnr.item(),
        "ssim": final_ssim.item()
    }

    out_file = path_out / f"eval_results_{current_time}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Wyniki zapisano do: {out_file}")