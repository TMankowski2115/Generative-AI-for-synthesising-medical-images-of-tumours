import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Ustalanie ścieżki do root projektu
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Importy z biblioteki medical_diffusion
from medical_diffusion.models.pipelines import DiffusionPipeline

# ==============================================================================
# 1. KONFIGURACJA WEKTORÓW (TUTAJ WPISUJESZ SWOJE DANE)
# ==============================================================================
# Format: [has_nodule, x, y, size, malignancy, texture]
# x, y, size: 0.0 - 1.0 (ułamek obrazu)
# mal, tex: 0.0 - 1.0 (znormalizowane cechy)

VECTORS = [
    # --- Seria: Zmiana wielkości (w tym samym miejscu) ---
    [1.0, 0.3, 0.3, 0.03, 0.8, 0.5], # Mały guzek
    [1.0, 0.3, 0.3, 0.05, 0.8, 0.5], # Średni guzek
    [1.0, 0.3, 0.3, 0.09, 0.8, 0.5], # Duży guzek

    # --- Seria: Zmiana położenia (ten sam rozmiar) ---
    [1.0, 0.3, 0.3, 0.05, 0.9, 1.0], # Lewy górny
    [1.0, 0.8, 0.3, 0.05, 0.9, 1.0], # Prawy górny
    [1.0, 0.5, 0.8, 0.05, 0.9, 1.0], # Dół środek

    # --- Seria: Cechy (złośliwość / tekstura) ---
    [1.0, 0.6, 0.4, 0.07, 0.0, 0.0], # Łagodny, GGO (rozmyty)
    [1.0, 0.6, 0.4, 0.07, 1.0, 1.0], # Złośliwy, Solid (lity, ostry)

    # --- Zdrowe płuco (dla porównania) ---
    [0.0, 0.5, 0.5, 0.00, 0.0, 0.0],
]

# ==============================================================================
# 2. DEFINICJE KLAS (Niezbędne do wczytania modelu)
# ==============================================================================

class ScalarEmbedder(nn.Module):
    def __init__(self, num_scalars=6, emb_dim=1024):
        super().__init__()
        self.emb_dim = emb_dim  
        self.net = nn.Sequential(
            nn.Linear(num_scalars, 256),
            nn.SiLU(),
            nn.Linear(256, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    def forward(self, x):
        if x.ndim == 1: x = x.unsqueeze(0)
        return self.net(x)

class VectorDiffusionPipeline(DiffusionPipeline):
    pass

# ==============================================================================
# 3. FUNKCJA ZAPISU
# ==============================================================================

def save_image(fake_ct, vec, idx, out_dir):
    # Rozpakowanie wektora do nazwy pliku
    has_nod = int(vec[0])
    x = vec[1]
    y = vec[2]
    size = vec[3]
    mal = vec[4]
    tex = vec[5]
    
    filename = f"gen_{idx:03d}_NOD{has_nod}_x{x:.2f}_y{y:.2f}_sz{size:.2f}_mal{mal:.2f}_tex{tex:.2f}.png"

    # [-1, 1] -> [0, 255]
    fake_np = ((fake_ct.clamp(-1, 1) + 1) / 2.0).cpu().numpy().squeeze()
    fake_u8 = (fake_np * 255).astype(np.uint8)
    
    img = Image.fromarray(fake_u8, mode="L")
    img.save(out_dir / filename)
    print(f"Zapisano: {filename}")

# ==============================================================================
# 4. GŁÓWNA LOGIKA
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- USTAWIENIA ---
    ckpt_path = ROOT_DIR / "runs/2025_12_01_163938_lidc_vector_diffusion/epoch=59-step=53640.ckpt"
    out_dir = Path.cwd() / "custom_vectors_results3"
    
    steps = 50
    guidance_scale = 2.0  # Siła podążania za wektorem

    # ------------------

    if not ckpt_path.exists():
        print(f"BŁĄD: Brak pliku {ckpt_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Ładuję model...")
    pipeline = VectorDiffusionPipeline.load_from_checkpoint(str(ckpt_path))
    pipeline.to(device)
    pipeline.eval()
    
    # Flagi bezpieczeństwa (dla latent diffusion)
    pipeline.do_input_centering = False
    pipeline.clip_x0 = False

    print(f"Start generowania {len(VECTORS)} obrazów z listy...")

    # Konwersja listy pythonowej na tensor
    # Kształt: [N, 6]
    cond_tensor = torch.tensor(VECTORS, dtype=torch.float32, device=device)
    
    # Warunek pusty dla CFG (tylko jeśli guidance > 1.0)
    un_cond = None
    if guidance_scale > 1.0:
        un_cond = torch.zeros_like(cond_tensor)

    # Kształt latentów VAE (8 kanałów, 32x32)
    latent_shape = (8, 32, 32)

    with torch.no_grad():
        
        imgs = pipeline.sample(
            num_samples=len(VECTORS),
            img_size=latent_shape,
            condition=cond_tensor,
            un_cond=un_cond,
            guidance_scale=guidance_scale,
            steps=steps
        )

        # Zapis wyników
        for i in range(len(VECTORS)):
            save_image(imgs[i], VECTORS[i], i, out_dir)

    print(f"\nGotowe! Wyniki w: {out_dir}")