# lidc_diffusion_dataset.py

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


# =============================== HU NORMALIZACJA ===============================

def hu_window_to_unit(arr_hu: np.ndarray, wl=-1000, wh=400):
    """Windowing HU → [0,1] float32."""
    a = np.clip(arr_hu, wl, wh)
    return (a - wl) / (wh - wl + 1e-8)


def normalize_ct_unit_to_minus1_1(x):
    """[0,1] → [-1,1]."""
    return x * 2.0 - 1.0


# =============================== MASK LOADER ===================================

def load_mask_nearest(path):
    """Load mask PNG preserving values {0, 1, 2} and return float32."""
    import cv2
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    
    # ZMIANA: Zwracamy surowe wartości (0, 1, 2), a nie bool (arr > 0)
    return arr.astype(np.float32)


# =============================== DATASET =======================================

class LIDCDiffusionDataset(Dataset):
    """
    Zwraca:
        ct             -> [1,H,W],   [-1,1]  (z HU -> window)
        lung_mask      -> [1,H,W],   0/1
        nodule_mask    -> [1,H,W],   0/1
        cond_vector    -> [5]
        prompt         -> string
        path           -> str
        has_nodule     -> tensor 0/1
    """

    def __init__(self, slices_root, split_txt, image_size=256):
        self.slices_root = slices_root
        self.image_size = image_size

        with open(split_txt, "r") as f:
            self.pids = [x.strip() for x in f if x.strip()]

        self.samples = []
        self.has_nodule_flags = []

        for pid in self.pids:
            pid_dir = os.path.join(slices_root, pid)
            if not os.path.isdir(pid_dir):
                print(f"[WARN] brak katalogu pacjenta: {pid}")
                continue

            for suid in os.listdir(pid_dir):
                suid_dir = os.path.join(pid_dir, suid)
                if not os.path.isdir(suid_dir):
                    continue

                for sub in os.listdir(suid_dir):
                    full = os.path.join(suid_dir, sub)
                    if not os.path.isdir(full):
                        continue

                    ct_hu_path = os.path.join(full, "ct_hu.npy")
                    labels_path = os.path.join(full, "labels.json")

                    if not (os.path.exists(ct_hu_path) and os.path.exists(labels_path)):
                        continue

                    self.samples.append(full)

                    with open(labels_path, "r") as f:
                        labels = json.load(f)

                    self.has_nodule_flags.append(bool(labels.get("has_nodule", False)))

        print(f"[INFO] Załadowano {len(self.samples)} slice'ów HU (bezstratnie).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        root = self.samples[idx]

        # ------------------------------------------
        # 1) LOAD HU (bezstratnie!)
        # ------------------------------------------
        ct_hu = np.load(os.path.join(root, "ct_hu.npy")).astype(np.float32)

        # → [0,1]
        ct_unit = hu_window_to_unit(ct_hu, wl=-1000, wh=400).astype(np.float32)

        # → torch [1,H,W]
        ct = torch.from_numpy(ct_unit)[None, :, :]  # [1,H,W]

        # resize bilinear
        ct = F.interpolate(ct[None], (self.image_size, self.image_size),
                           mode="bilinear", align_corners=False)[0]

        # → [-1,1]
        ct = normalize_ct_unit_to_minus1_1(ct)

        # ------------------------------------------
        # 2) LOAD MASKS (nearest)
        # ------------------------------------------
        lung_np = load_mask_nearest(os.path.join(root, "lung_mask.png"))
        lung = torch.from_numpy(lung_np)[None, :, :]
        lung = F.interpolate(lung[None], (self.image_size, self.image_size),
                             mode="nearest")[0]

        nod_path = os.path.join(root, "nodule_mask.png")
        if os.path.exists(nod_path):
            nod_np = load_mask_nearest(nod_path)
        else:
            nod_np = np.zeros_like(lung_np, dtype=np.float32)

        nod = torch.from_numpy(nod_np)[None, :, :]
        nod = F.interpolate(nod[None], (self.image_size, self.image_size),
                            mode="nearest")[0]

        labels_path = os.path.join(root, "labels.json")
        with open(labels_path, "r") as f:
            labels = json.load(f)

        H, W = labels["image_size"]
        has_nodule = bool(labels.get("has_nodule", False))

        if has_nodule:
            xmin, ymin, xmax, ymax = labels["nodule_bbox_xyxy_px"]
            cx = (xmin + xmax) / 2.0 / W
            cy = (ymin + ymax) / 2.0 / H
            radius = labels["diameter_mm"] / 30.0

            # --- NOWOŚĆ: Odległość od opłucnej ---
            # Pobieramy wartość, domyślnie 0.0
            dist_mm = labels.get("distance_to_pleura_mm")
            if dist_mm is None:
                dist_mm = 0.0
            
            # Normalizacja: zakładamy, że >30mm to już "daleko". 
            # Zakres [0, 1].
            dist_norm = np.clip(dist_mm, 0.0, 30.0) / 30.0

            side = labels["side"]
            if side == "left":
                side_vec = [1.0, 0.0]
            elif side == "right":
                side_vec = [0.0, 1.0]
            else:
                side_vec = [0.0, 0.0]
        else:
            # Dla negatywów (tło) wszystko zerujemy
            cx = cy = radius = dist_norm = 0.0
            side_vec = [0.0, 0.0]

        # ZMIANA: Wektor ma teraz 6 elementów: [cx, cy, r, side_L, side_R, dist]
        cond_vector = torch.tensor([cx, cy, radius] + side_vec + [dist_norm], dtype=torch.float32)

        # ------------------------------------------
        # 4) LOAD PROMPT
        # ------------------------------------------
        prompt_path = os.path.join(root, "prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                prompt = f.read().strip()
        else:
            prompt = ""

        return {
            "ct": ct,                      # [1,H,W] [-1,1]
            "lung_mask": lung,             # [1,H,W] 0/1
            "nodule_mask": nod,            # [1,H,W] 0/1
            "cond_vector": cond_vector,    # [5]
            "prompt": prompt,
            "path": root,
            "has_nodule": torch.tensor(int(has_nodule), dtype=torch.int64)
        }
