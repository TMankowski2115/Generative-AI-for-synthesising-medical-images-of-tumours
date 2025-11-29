from pathlib import Path
import json

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

WL, WH = -1000.0, 400.0          # okno HU
IMG_SIZE = 256                   # 256x256

class SimpleDataset2D(Dataset):
    """
    LIDC 2D CT slices (tylko slajsy BEZ guzka).
    Czyta strukturę:
      dataset_lidc_2d_seg/
        slices/<patient_id>/<series_uid>/<z_xxxx_...>/
          ct_hu.npy
          lung_mask.png
          labels.json   # has_nodule=false

    Zwraca:
      x: tensor (1, 256, 256) w zakresie [-1, 1]
      y: label = 0 (dummy, bo trenujemy bez warunkowania)
    """
    def __init__(self, root_dir, split="train", fraction=1.0):
        """
        root_dir: ścieżka do 'dataset_lidc_2d_seg'
        split: 'train' / 'val' / 'test'
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.fraction = fraction

        # wczytaj listę pacjentów dla działania Medfusion "per split"
        split_file = self.root_dir / "splits" / f"{split}.txt"
        with split_file.open("r", encoding="utf-8") as f:
            allowed_pids = {line.strip() for line in f if line.strip()}

        slices_root = self.root_dir / "slices"
        self.samples = []

        # przejście po slices/<pid>/<suid>/<z_xxxx_*>
        for pid_dir in slices_root.iterdir():
            if not pid_dir.is_dir():
                continue
            if pid_dir.name not in allowed_pids:
                continue

            for suid_dir in pid_dir.iterdir():
                if not suid_dir.is_dir():
                    continue

                for slice_dir in suid_dir.iterdir():
                    if not slice_dir.is_dir():
                        continue

                    labels_path = slice_dir / "labels.json"
                    ct_hu_path = slice_dir / "ct_hu.npy"
                    lung_mask_path = slice_dir / "lung_mask.png"

                    if not (labels_path.exists() and ct_hu_path.exists() and lung_mask_path.exists()):
                        continue

                    # sprawdź, czy to slajs BEZ guzka (has_nodule=false)
                    with labels_path.open("r", encoding="utf-8") as f:
                        labels = json.load(f)
                    if labels.get("has_nodule", True):   # True → ma guzek → omijamy
                        continue

                    # w tym miejscu *wiemy*, że to negatyw
                    self.samples.append({
                        "ct_hu": ct_hu_path,
                        "lung_mask": lung_mask_path,
                    })

        print(f"[LIDC-{split}] znaleziono {len(self.samples)} slajsów (przed frakcją).")

        if self.fraction < 1.0:
            import random
            random.seed(42)
            k = int(len(self.samples) * self.fraction)
            self.samples = random.sample(self.samples, k)
            print(f"[LIDC-{split}] po zastosowaniu fraction={self.fraction}: {len(self.samples)} slajsów.")


        print(f"[LIDC-{split}] negatywnych slajsów bez guzka: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # --- wczytaj HU ---
        ct_hu = np.load(item["ct_hu"]).astype(np.float32)   # [H,W] w HU

        # --- wczytaj maskę płuc (0,1,2) ---
        lung_mask = cv2.imread(str(item["lung_mask"]), cv2.IMREAD_UNCHANGED)
        if lung_mask is None:
            raise RuntimeError(f"Nie mogę wczytać lung_mask: {item['lung_mask']}")
        lung_mask = lung_mask.astype(np.int32)

        # binarna maska płuc
        lung_bin = lung_mask > 0

        # nałóż maskę na HU – tło ustaw na powietrze (WL)
        ct_hu_masked = ct_hu.copy()
        ct_hu_masked[~lung_bin] = WL

        # okno HU [-1000, 400]
        ct_win = np.clip(ct_hu_masked, WL, WH)

        # normalizacja do [0,1]
        ct_01 = (ct_win - WL) / (WH - WL)

        # skala do [-1,1] (tak zwykle pracuje VAE / diffusion)
        ct_norm = ct_01 * 2.0 - 1.0

        # resize do 256x256
        ct_resized = cv2.resize(ct_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # tensor (1, H, W)
        x = torch.from_numpy(ct_resized).unsqueeze(0).float()

        # dummy label (0) – Medfusion i tak go nie użyje,
        # bo będziemy mieć cond_embedder = None
        y = torch.tensor(0, dtype=torch.long)

        return {
            "source": x,   # to model bierze jako wejście
            "label": y,    # może zignorować, ale klucz istnieje
        }
