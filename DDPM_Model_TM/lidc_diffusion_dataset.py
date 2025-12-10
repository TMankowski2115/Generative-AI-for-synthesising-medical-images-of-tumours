# lidc_diffusion_dataset.py

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


# =============================== MASK LOADER ===================================

def load_mask_nearest(path):
    """Wczytuje maskę zachowując wartości 0, 1, 2."""
    import cv2
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    # Zwracamy surowe wartości (0, 1, 2), a nie bool (arr > 0)
    return arr.astype(np.float32)


# =============================== HU TRANSFORM ==================================

def process_ct_to_normalized(ct_hu, lung_mask_np, target_size):
    """
    Zmienia HU na zakres [-1, 1] zoptymalizowany pod płuca.
    Tło (poza maską) jest sztywno ustawiane na -1.0 (czarne).
    """
    # 1. Okno "Lung Optimized"
    # wl (low) = -1000 (powietrze)
    # wh (high) = +100  (wszystko powyżej, np. kości, będzie idealnie białe)
    wl, wh = -1000, 100 
    
    ct_unit = np.clip(ct_hu, wl, wh)
    ct_unit = (ct_unit - wl) / (wh - wl + 1e-8) # [0, 1]
    
    # 2. Normalizacja do [-1, 1]
    ct_norm = ct_unit * 2.0 - 1.0 
    
    # 3. TWARDE MASKOWANIE TŁA (Naprawa "szarego tła")
    # Wszystko co nie jest płucem (mask == 0), staje się idealnie czarne (-1.0)
    bg_mask = (lung_mask_np == 0)
    ct_norm[bg_mask] = -1.0
    
    # 4. Konwersja na tensor i resize
    t_ct = torch.from_numpy(ct_norm)[None, :, :] # [1, H, W]
    t_ct = F.interpolate(t_ct[None], (target_size, target_size), mode="bilinear", align_corners=False)[0]
    
    return t_ct


# =============================== DATASET =======================================

class LIDCDiffusionDataset(Dataset):
    """
    Zwraca:
        ct             -> [1,H,W],   [-1,1]  (zoptymalizowany kontrast, czarne tło)
        lung_mask      -> [1,H,W],   0/1
        nodule_mask    -> [1,H,W],   0/1
        cond_vector    -> [6]        [cx, cy, radius, side_L, side_R, dist_pleura]
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
                # print(f"[WARN] brak katalogu pacjenta: {pid}") # Opcjonalne wyciszenie
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
        # 1) LOAD MASKS (Najpierw maska, bo potrzebna do tła)
        # ------------------------------------------
        lung_np = load_mask_nearest(os.path.join(root, "lung_mask.png"))
        
        # Resize maski płuc
        lung = torch.from_numpy(lung_np)[None, :, :]
        lung = F.interpolate(lung[None], (self.image_size, self.image_size),
                             mode="nearest")[0]

        # ------------------------------------------
        # 2) LOAD HU & PROCESS (Z nowym oknem i maskowaniem)
        # ------------------------------------------
        ct_hu = np.load(os.path.join(root, "ct_hu.npy")).astype(np.float32)
        ct = process_ct_to_normalized(ct_hu, lung_np, self.image_size)

        # ------------------------------------------
        # 3) LOAD NODULE MASK
        # ------------------------------------------
        nod_path = os.path.join(root, "nodule_mask.png")
        if os.path.exists(nod_path):
            nod_np = load_mask_nearest(nod_path)
        else:
            nod_np = np.zeros_like(lung_np, dtype=np.float32)

        nod = torch.from_numpy(nod_np)[None, :, :]
        nod = F.interpolate(nod[None], (self.image_size, self.image_size),
                            mode="nearest")[0]

        # ------------------------------------------
        # 4) CONDITIONS (Labels)
        # ------------------------------------------
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

            # Odległość od opłucnej
            dist_mm = labels.get("distance_to_pleura_mm")
            if dist_mm is None:
                dist_mm = 0.0
            
            # Normalizacja: zakładamy, że >30mm to już "daleko"
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

        # Wektor ma 6 elementów: [cx, cy, r, side_L, side_R, dist]
        cond_vector = torch.tensor([cx, cy, radius] + side_vec + [dist_norm], dtype=torch.float32)

        # ------------------------------------------
        # 5) LOAD PROMPT
        # ------------------------------------------
        prompt_path = os.path.join(root, "prompt.txt")
        prompt = ""
        # Prompt ignorujemy w treningu numerycznym, ale zostawiamy do kompatybilności
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                prompt = f.read().strip()

        return {
            "ct": ct,                      # [1,H,W] [-1,1], tło=-1.0
            "lung_mask": lung,             # [1,H,W] 0/1/2
            "nodule_mask": nod,            # [1,H,W] 0/1
            "cond_vector": cond_vector,    # [6]
            "prompt": prompt,
            "path": root,
            "has_nodule": torch.tensor(int(has_nodule), dtype=torch.int64)
        }
