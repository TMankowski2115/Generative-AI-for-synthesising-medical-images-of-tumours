from pathlib import Path
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# ───────── KONFIGURACJA ──────────────────────────────────────────────────────
WL, WH = -1000.0, 400.0          # Okno HU (Lung Window)
IMG_SIZE = 256                   # Rozmiar wejściowy modelu
NEGATIVE_RATIO = 0.4             # Cel: 40% datasetu to negatywy

# Maksymalne wartości w skali LIDC do normalizacji (klucze z labels.json)
# Używamy tego, by sprowadzić cechy do zakresu 0..1
LIDC_MAX_VALS = {
    "subtlety": 5.0,
    "calcification": 6.0,
    "texture": 5.0,
    "malignancy": 5.0
}
# Domyślna wartość (środek skali), gdyby brakowało oceny w XML
DEFAULT_LIDC_VAL = 3.0 

class LIDCConditionalDataset(Dataset):
    """
    Dataset dla MedFusion (LDM) z warunkowaniem (Conditioning).
    
    Wejście modelu (source):
      Tensor [2, 256, 256]
       - Kanał 0: Obraz CT (wycięty do płuc/guzka), zakres [-1, 1]
       - Kanał 1: Gaussian Heatmap (lokalizacja guzka), zakres [0, 1]
       
    Warunek (attributes):
      Tensor [5]
       - [has_nodule (0/1), subtlety, calcification, texture, malignancy]
       - Cechy znormalizowane do [0, 1]. Dla negatywów reszta to -1.
    """
    def __init__(self, root_dir, split="train"):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        
        # 1. Wczytaj listę pacjentów dla danego splitu
        split_file = self.root_dir / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Brak pliku splitu: {split_file}")
            
        with split_file.open("r", encoding="utf-8") as f:
            allowed_pids = {line.strip() for line in f if line.strip()}

        slices_root = self.root_dir / "slices"
        
        pos_samples = []
        neg_samples = []

        # 2. Przeskanuj katalogi i podziel na POS / NEG
        print(f"[Init] Skanowanie katalogów dla {split}...")
        for pid_dir in slices_root.iterdir():
            if not pid_dir.is_dir() or pid_dir.name not in allowed_pids:
                continue

            for suid_dir in pid_dir.iterdir():
                if not suid_dir.is_dir(): continue

                for slice_dir in suid_dir.iterdir():
                    if not slice_dir.is_dir(): continue
                    
                    # Wymagane pliki
                    f_labels = slice_dir / "labels.json"
                    f_ct = slice_dir / "ct_hu.npy"
                    f_lung = slice_dir / "lung_mask.png"
                    
                    if not (f_labels.exists() and f_ct.exists() and f_lung.exists()):
                        continue
                        
                    # Sprawdź czy to pozytyw czy negatyw
                    # (Twój skrypt nazywa foldery np. "z_0142_nod-01" lub "z_0087_neg")
                    try:
                        with open(f_labels, 'r') as f:
                            meta = json.load(f)
                            has_nodule = meta.get("has_nodule", False)
                    except Exception:
                        continue
                    
                    sample = {
                        "path": slice_dir,
                        "ct": f_ct,
                        "lung": f_lung,
                        "labels": f_labels,
                        "has_nodule": has_nodule
                    }
                    
                    # Dla pozytywów musi być maska guzka
                    if has_nodule:
                        f_nod = slice_dir / "nodule_mask.png"
                        if f_nod.exists():
                            sample["nodule_mask"] = f_nod
                            pos_samples.append(sample)
                    else:
                        neg_samples.append(sample)

        # 3. Balansowanie datasetu (Negative Downsampling)
        n_pos = len(pos_samples)
        if n_pos > 0:
            # target_neg / (n_pos + target_neg) = 0.4  =>  target_neg = (0.4 * n_pos) / 0.6
            target_neg = int((NEGATIVE_RATIO * n_pos) / (1.0 - NEGATIVE_RATIO))
            target_neg = min(target_neg, len(neg_samples)) # Nie możemy wziąć więcej niż mamy
            
            random.seed(42)
            neg_samples_balanced = random.sample(neg_samples, target_neg)
        else:
            neg_samples_balanced = neg_samples # Fallback jeśli brak pozytywów (np. test set)

        self.samples = pos_samples + neg_samples_balanced
        random.shuffle(self.samples)

        print(f"[Dataset-{split}] Total: {len(self.samples)}")
        print(f"   - Positives: {len(pos_samples)}")
        print(f"   - Negatives: {len(neg_samples_balanced)} (Ratio: {len(neg_samples_balanced)/len(self.samples):.2f})")

    def __len__(self):
        return len(self.samples)

    def _generate_blob(self, img_size, bbox, sigma_scale=0.25):
        """Generuje mapę Gaussa [H, W] na podstawie bbox [minx, miny, maxx, maxy]."""
        H, W = img_size
        heatmap = torch.zeros((H, W), dtype=torch.float32)

        if bbox is None:
            return heatmap

        min_x, min_y, max_x, max_y = bbox
        # Zabezpieczenie przed pustym bboxem
        if max_x <= min_x or max_y <= min_y:
            return heatmap
            
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        bw = max_x - min_x
        bh = max_y - min_y
        
        # Sigma zależna od rozmiaru guzka (żeby plamka pasowała wielkością)
        # Skalujemy w dół (datasetLIDC ma 512x512, my chcemy 256x256 -> ratio)
        # Ale bbox jest w pikselach oryginału? Sprawdźmy jsona.
        # W jsonie masz "nodule_bbox_xyxy_px" względem oryginalnego obrazu 512x512.
        # Tutaj robimy resize do 256, więc musimy przeskalować parametry Gaussa.
        
        # Obliczymy Gaussa na siatce 256x256, więc skalujemy współrzędne:
        scale_x = W / 512.0 # Zakładam, że LIDC ma 512
        scale_y = H / 512.0 # (Można to pobrać z image_size w jsonie dla pewności)
        
        cx *= scale_x
        cy *= scale_y
        bw *= scale_x
        bh *= scale_y
        
        sigma_x = bw * sigma_scale
        sigma_y = bh * sigma_scale
        
        # Grid
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        gaussian = torch.exp(
            -(((x_grid - cx)**2) / (2 * sigma_x**2 + 1e-1) + 
              ((y_grid - cy)**2) / (2 * sigma_y**2 + 1e-1))
        )
        
        if gaussian.max() > 0:
            gaussian /= gaussian.max()
            
        return gaussian

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Wczytanie obrazu i masek
        ct_hu = np.load(item["ct"]).astype(np.float32)
        lung_mask = cv2.imread(str(item["lung"]), cv2.IMREAD_UNCHANGED) # 0,1,2
        
        # Ustalenie oryginalnego rozmiaru do skalowania bboxa
        orig_h, orig_w = ct_hu.shape
        
        if item["has_nodule"]:
            nod_mask = cv2.imread(str(item["nodule_mask"]), cv2.IMREAD_UNCHANGED) # 0,1
            # Wczytaj metadane
            with open(item["labels"], 'r') as f:
                meta = json.load(f)
                
            # --- ATTRYBUTY ---
            # Pobierz cechy LIDC i znormalizuj
            chars = meta.get("lidc_characteristics_mean", {})
            
            def norm(key):
                val = chars.get(key)
                if val is None: val = DEFAULT_LIDC_VAL
                max_v = LIDC_MAX_VALS.get(key, 5.0)
                return (val - 1.0) / (max_v - 1.0) # Zakładamy skalę startującą od 1
            
            # Wektor: [1, subtlety, calc, texture, malignancy]
            attr_vec = [
                1.0, 
                norm("subtlety"),
                norm("calcification"),
                norm("texture"),
                norm("malignancy")
            ]
            
            # --- BLOB ---
            bbox = meta.get("nodule_bbox_xyxy_px", None) # [x1, y1, x2, y2]
            # Przeskaluj bboxa jeśli obraz nie jest 512 (zabezpieczenie)
            scale_w = IMG_SIZE / orig_w
            scale_h = IMG_SIZE / orig_h
            
            # Generuj heatmapę od razu w docelowym rozmiarze 256x256
            # (bbox trzeba przeskalować w funkcji _generate_blob, ale tu podajemy raw)
            # Uwaga: moja funkcja _generate_blob wyżej zakładała sztywno 512 -> 256. 
            # Zróbmy to porządnie: podajmy przeskalowane współrzędne do funkcji.
            
            bbox_rescaled = [
                bbox[0] * scale_w, bbox[1] * scale_h,
                bbox[2] * scale_w, bbox[3] * scale_h
            ]
            blob_map = self._generate_blob((IMG_SIZE, IMG_SIZE), bbox_rescaled)
            
            # --- MASKA FINALNA ---
            # Suma logiczna (Lung OR Nodule)
            final_mask = (lung_mask > 0) | (nod_mask > 0)
            
        else:
            # NEGATYW
            attr_vec = [0.0, -1.0, -1.0, -1.0, -1.0] # 0 = no nodule
            blob_map = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.float32)
            final_mask = (lung_mask > 0)

        # 2. Preprocessing CT
        # Wycięcie tła
        ct_hu_masked = ct_hu.copy()
        ct_hu_masked[~final_mask] = WL # Ustaw tło na powietrze
        
        # Okno i normalizacja
        ct_win = np.clip(ct_hu_masked, WL, WH)
        ct_01 = (ct_win - WL) / (WH - WL)     # [0, 1]
        ct_norm = ct_01 * 2.0 - 1.0           # [-1, 1]
        
        # Resize CT do 256x256
        ct_resized = cv2.resize(ct_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # 3. Złożenie tensora wejściowego [2, 256, 256]
        # Kanał 0: CT [-1, 1]
        # Kanał 1: Blob [0, 1] (normalizujemy go osobno, bo to maska prawdopodobieństwa)
        
        img_tensor = torch.from_numpy(ct_resized).unsqueeze(0).float() # [1, 256, 256]
        blob_tensor = blob_map.unsqueeze(0).float()                    # [1, 256, 256]
        
        # Konkatenacja
        x = torch.cat([img_tensor, blob_tensor], dim=0) # -> [2, 256, 256]
        
        # Wektor atrybutów
        attributes = torch.tensor(attr_vec, dtype=torch.float32)

        return {
            "source": x,          # Input do UNeta
            "attributes": attributes, # Input do Condition Embeddera
            "label": torch.tensor(0) # Dummy
        }