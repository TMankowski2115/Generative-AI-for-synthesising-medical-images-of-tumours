from pathlib import Path
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# ───────── KONFIGURACJA ──────────────────────────────────────────────────────
WL, WH = -1000.0, 400.0          # Okno płucne (HU)
IMG_SIZE = 256                   # Rozmiar wejściowy modelu
NEGATIVE_RATIO = 0.4             # Cel: 40% datasetu to negatywy (zdrowe płuca)

# Parametry normalizacji cech LIDC (zakresy 1..5)
LIDC_MAX_VAL = 5.0
DEFAULT_VAL = 3.0 

class LIDCVectorDataset(Dataset):
    """
    Dataset dla MedFusion ze sterowaniem wektorowym (Global Conditioning).
    
    Cechy:
    - UNFILTERED: Akceptuje slajsy z wieloma guzkami (zwiększenie wolumenu danych).
    - BALANCED VAL: Balansuje też zbiór walidacyjny, żeby loss był miarodajny.
    - VECTOR: [has_nod, x, y, size, mal, tex] (wszystko w zakresie 0-1).
    
    Zwraca:
      source: Tensor [1, 256, 256] -> Czyste CT (w zakresie -1..1)
      target: Tensor [6] -> Wektor cech
    """
    def __init__(self, root_dir, split="train"):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        
        # 1. Wczytaj listę pacjentów dla splitu
        split_file = self.root_dir / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Brak pliku splitu: {split_file}")
            
        with split_file.open("r", encoding="utf-8") as f:
            allowed_pids = {line.strip() for line in f if line.strip()}

        slices_root = self.root_dir / "slices"
        
        pos_samples = []
        neg_samples = []
        
        skipped_missing_data = 0

        # 2. Skanowanie folderów
        print(f"[Init] Skanowanie {split}...")
        for pid_dir in slices_root.iterdir():
            if not pid_dir.is_dir() or pid_dir.name not in allowed_pids:
                continue

            for suid_dir in pid_dir.iterdir():
                if not suid_dir.is_dir(): continue

                for slice_dir in suid_dir.iterdir():
                    if not slice_dir.is_dir(): continue
                    
                    f_labels = slice_dir / "labels.json"
                    f_ct = slice_dir / "ct_hu.npy"
                    f_lung = slice_dir / "lung_mask.png"
                    
                    if not (f_labels.exists() and f_ct.exists() and f_lung.exists()):
                        continue
                        
                    # Sprawdź metadane
                    try:
                        with open(f_labels, 'r') as f:
                            meta = json.load(f)
                            has_nodule = meta.get("has_nodule", False)
                            
                            if has_nodule:
                                # Sprawdź czy mamy kompletne cechy (nie chcemy uczyć na None)
                                chars = meta.get("lidc_characteristics_mean", {})
                                val_mal = chars.get("malignancy")
                                val_tex = chars.get("texture")
                                
                                # Jeśli brakuje oceny radiologa, pomijamy tę próbkę
                                if val_mal is None or val_tex is None:
                                    skipped_missing_data += 1
                                    continue
                                
                                # UWAGA: Nie filtrujemy już nodule_count > 1.
                                # Każdy folder nod-XX to osobna, poprawna próbka.

                    except Exception:
                        continue
                    
                    sample = {
                        "ct": f_ct,
                        "lung": f_lung,
                        "labels": f_labels,
                        "has_nodule": has_nodule
                    }
                    
                    if has_nodule:
                        f_nod = slice_dir / "nodule_mask.png"
                        if f_nod.exists():
                            sample["nodule_mask"] = f_nod
                            pos_samples.append(sample)
                    else:
                        neg_samples.append(sample)

        # 3. Balansowanie (Dla Train ORAZ Val/Test)
        # Dzięki temu val_loss będzie sprawiedliwie oceniał generowanie guzków (trudniejsze zadanie),
        # a nie tylko tła.
        n_pos = len(pos_samples)
        if n_pos > 0:
            # Obliczamy ile negatywów dobrać (tak żeby negatywy stanowiły NEGATIVE_RATIO, np. 40%)
            # Wzór: target_neg / (n_pos + target_neg) = 0.4
            target_neg = int((NEGATIVE_RATIO * n_pos) / (1.0 - NEGATIVE_RATIO))
            
            # Nie możemy wziąć więcej niż mamy
            target_neg = min(target_neg, len(neg_samples))
            
            # Ustawiamy ziarno losowości zależne od splitu
            # Train: losowo (42), Val/Test: stałe ziarno (1337) żeby walidacja była powtarzalna
            rng_seed = 42 if split == 'train' else 1337
            rng = random.Random(rng_seed)
            
            neg_samples_balanced = rng.sample(neg_samples, target_neg)
        else:
            neg_samples_balanced = neg_samples # Fallback
        
        self.samples = pos_samples + neg_samples_balanced
        
        # Mieszamy tylko zbiór treningowy. Walidacyjny może być posortowany.
        if split == "train":
            random.shuffle(self.samples)

        print(f"[Dataset-{split}] Total: {len(self.samples)} (Pos: {len(pos_samples)}, Neg: {len(neg_samples_balanced)})")
        print(f"   -> Pominięto (brak cech w XML): {skipped_missing_data}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # ── 1. Wczytanie obrazu ──
        ct_hu = np.load(item["ct"]).astype(np.float32)
        lung_mask = cv2.imread(str(item["lung"]), cv2.IMREAD_UNCHANGED)
        
        # Pobierz oryginalne wymiary (ważne do normalizacji współrzędnych)
        orig_h, orig_w = ct_hu.shape

        # ── 2. Przygotowanie wektora cech (Target) i Maski ──
        if item["has_nodule"]:
            nod_mask = cv2.imread(str(item["nodule_mask"]), cv2.IMREAD_UNCHANGED)
            final_mask = (lung_mask > 0) | (nod_mask > 0)
            
            with open(item["labels"], 'r') as f:
                meta = json.load(f)
            
            # --- Współrzędne i Rozmiar ---
            bbox = meta.get("nodule_bbox_xyxy_px", [0, 0, 0, 0])
            
            # Środek (Center)
            cx_px = (bbox[0] + bbox[2]) / 2.0
            cy_px = (bbox[1] + bbox[3]) / 2.0
            
            # Normalizacja do [0, 1]
            norm_x = np.clip(cx_px / orig_w, 0.0, 1.0)
            norm_y = np.clip(cy_px / orig_h, 0.0, 1.0)
            
            # Rozmiar (maksymalny wymiar bboxa)
            size_px = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            # Normalizujemy względem szerokości obrazu (np. guzek 50px na 512px -> ~0.1)
            norm_size = np.clip(size_px / orig_w, 0.0, 1.0)
            
            # --- Cechy LIDC (Texture, Malignancy) ---
            chars = meta.get("lidc_characteristics_mean", {})
            
            # Pobieramy wartości (filtr w __init__ gwarantuje, że nie są None, ale dajemy fallback)
            vm = chars.get("malignancy", DEFAULT_VAL)
            vt = chars.get("texture", DEFAULT_VAL)
            
            def norm_char(val):
                # Skala 1..5 -> 0..1
                return (val - 1.0) / (LIDC_MAX_VAL - 1.0)

            val_mal = norm_char(vm)
            val_tex = norm_char(vt)
            
            # Wektor: [1, x, y, size, mal, tex]
            target_vec = [1.0, norm_x, norm_y, norm_size, val_mal, val_tex]
            
        else:
            # NEGATYW
            final_mask = (lung_mask > 0)
            
            # Wektor: [0, 0, 0, 0, 0, 0]
            # Same zera to jasny sygnał dla modelu: "To jest puste płuco, ignoruj pozycję"
            target_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # ── 3. Preprocessing CT (Standard) ──
        ct_hu_masked = ct_hu.copy()
        ct_hu_masked[~final_mask] = WL # Wycięcie tła
        
        # Okno i normalizacja
        ct_win = np.clip(ct_hu_masked, WL, WH)
        ct_01 = (ct_win - WL) / (WH - WL)
        ct_norm = ct_01 * 2.0 - 1.0  # [-1, 1]
        
        # Resize
        ct_resized = cv2.resize(ct_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # ── 4. Tworzenie Tensorów ──
        # Source: [1, 256, 256] -> Czyste CT (kompatybilne z VAE)
        x = torch.from_numpy(ct_resized).unsqueeze(0).float()
        x = torch.clamp(x, -1.0, 1.0) # Zabezpieczenie przed artefaktami resize
        
        # Target: [6] -> Wektor
        y = torch.tensor(target_vec, dtype=torch.float32)

        return {
            "source": x,
            "target": y
        }