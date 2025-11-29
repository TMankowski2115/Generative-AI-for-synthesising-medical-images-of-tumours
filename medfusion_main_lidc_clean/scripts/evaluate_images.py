import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print("DBG ROOT_DIR:", ROOT_DIR)

import logging
from datetime import datetime
from tqdm import tqdm
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance as FID
from medical_diffusion.metrics.torchmetrics_pr_recall import ImprovedPrecessionRecall


# ---------------- Settings --------------
batch_size = 64
max_samples = 1000   # None = wszystkie, albo np. 1000 na szybki test

# katalog z realnymi CT (Twoje LIDC 2D)
LIDC_ROOT = Path("../dataset_lidc_2d_seg")      # główny katalog
REAL_SLICES_ROOT = LIDC_ROOT / "slices"      # tam są pacjenci/serie/z_xxxx_.../ct.png

# katalog z FAKE obrazkami z dyfuzji (PNG z naszego sample skryptu)
FAKE_ROOT = Path("synthetic_lidc_diffusion") / "steps_100"  # <- PODMIEŃ np. na steps_50 / steps_200

# gdzie zapisywać logi/metryki
path_out = Path.cwd() / "results" / "LIDC_Diffusion" / "metrics"
path_out.mkdir(parents=True, exist_ok=True)


# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger("lidc_diffusion_fid")
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
    PIL.Image -> tensor uint8 (3, H, W).
    FID / Inception lubią 3 kanały, więc wymuszamy RGB.
    """
    arr = np.array(pil_img.convert("RGB"))  # (H, W, 3)
    return torch.as_tensor(arr).permute(2, 0, 1)  # (3, H, W)


# ---------------- Datasety ----------------
class LIDCRealCTDataset(Dataset):
    """
    Czyta realne CT z LIDC 2D:
      dataset_lidc_2d_seg/slices/<pid>/<suid>/<z_xxxx_...>/
        ct.png
        labels.json (has_nodule: bool)

    Bierzemy TYLKO slajsy z has_nodule = False,
    bo na takich trenowałeś dyfuzję.
    """

    def __init__(self, slices_root: Path, max_samples: int | None = None):
        super().__init__()
        self.files = []

        for pid_dir in slices_root.iterdir():
            if not pid_dir.is_dir():
                continue
            for suid_dir in pid_dir.iterdir():
                if not suid_dir.is_dir():
                    continue
                for slice_dir in suid_dir.iterdir():
                    if not slice_dir.is_dir():
                        continue
                    labels_path = slice_dir / "labels.json"
                    ct_png_path = slice_dir / "ct.png"
                    if not (labels_path.exists() and ct_png_path.exists()):
                        continue
                    try:
                        with labels_path.open("r", encoding="utf-8") as f:
                            labels = json.load(f)
                    except Exception:
                        continue

                    # Bierzemy tylko negatywy (has_nodule=false)
                    if labels.get("has_nodule", True):
                        continue

                    self.files.append(ct_png_path)

        self.files = sorted(self.files)
        if max_samples is not None:
            self.files = self.files[:max_samples]

        logger.info(f"[Real] LIDC negatywnych slajsów (do FID): {len(self.files)}")

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
    Czyta FAKE CT z katalogu:
      synthetic_lidc_diffusion/steps_XXX/*.png
    """

    def __init__(self, fake_root: Path, max_samples: int | None = None):
        super().__init__()
        self.files = sorted(list(fake_root.glob("*.png")))
        if max_samples is not None:
            self.files = self.files[:max_samples]
        logger.info(f"[Fake] syntetycznych slajsów (do FID): {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image
        path = self.files[idx]
        img = Image.open(path)
        x = pil2torch_rgb_uint8(img)  # (3,H,W) uint8
        return x


# ------------- Inicjalizacja datasetów / loaderów -------------
ds_real = LIDCRealCTDataset(REAL_SLICES_ROOT, max_samples=max_samples)
ds_fake = FakeCTDataset(FAKE_ROOT, max_samples=max_samples)

dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)
dm_fake = DataLoader(ds_fake, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)

logger.info(f"Samples Real: {len(ds_real)}")
logger.info(f"Samples Fake: {len(ds_fake)}")

# ------------- Init Metrics ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

calc_fid = FID().to(device)  # wymaga uint8 lub float w [0,1] / [0,255]
calc_pr = ImprovedPrecessionRecall(splits_real=1, splits_fake=1).to(device)


# --------------- Liczenie metryk -----------------
for real_batch in tqdm(dm_real, desc="Real batches"):
    imgs_real_batch = real_batch.to(device)  # (B,3,H,W) uint8

    # ----- FID -----
    calc_fid.update(imgs_real_batch, real=True)

    # ----- Precision/Recall -----
    calc_pr.update(imgs_real_batch, real=True)


for fake_batch in tqdm(dm_fake, desc="Fake batches"):
    imgs_fake_batch = fake_batch.to(device)

    # ----- FID -----
    calc_fid.update(imgs_fake_batch, real=False)

    # ----- Precision/Recall -----
    calc_pr.update(imgs_fake_batch, real=False)


# -------------- Podsumowanie -------------------
fid = calc_fid.compute()
precision, recall = calc_pr.compute()

logger.info(f"FID Score: {fid}")
logger.info(f"Precision: {precision}, Recall: {recall}")

print("===================================")
print("  WYNIKI DYFUZJI NA LIDC (NEGATYWY)")
print("===================================")
print(f"FID: {fid.item():.4f}")
print(f"Precision: {precision.item():.4f}")
print(f"Recall: {recall.item():.4f}")
