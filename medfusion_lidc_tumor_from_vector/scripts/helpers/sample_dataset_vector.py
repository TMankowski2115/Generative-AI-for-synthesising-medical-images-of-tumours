import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

# Ustalanie ścieżki do root projektu
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Importy z biblioteki medical_diffusion
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.embedders.latent_embedders import VAE
from medical_diffusion.models.pipelines import DiffusionPipeline

# Import Datasetu
from medical_diffusion.data.datasets.dataset_lidc_vector import LIDCVectorDataset

# ==============================================================================
# 1. RE-DEFINICJA KLAS (Dla load_from_checkpoint)
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
# 2. FUNKCJA ZAPISU (TYLKO FAKE)
# ==============================================================================

def save_generated_image(fake_ct, vec, idx, out_dir):
    """
    Zapisuje TYLKO wygenerowany obraz.
    Nazwa pliku zawiera parametry, które model otrzymał.
    """
    # Wektor: [has_nod, x, y, size, mal, tex]
    has_nod = int(vec[0])
    x = vec[1]
    y = vec[2]
    size = vec[3]
    mal = vec[4]
    tex = vec[5]
    
    # Budowanie nazwy pliku
    filename = f"{idx:04d}_NOD{has_nod}_x{x:.2f}_y{y:.2f}_sz{size:.2f}_mal{mal:.2f}_tex{tex:.2f}.png"

    # Normalizacja [-1, 1] -> [0, 1]
    # squeeze() usuwa wymiar kanału (1, H, W) -> (H, W)
    fake_np = ((fake_ct.clamp(-1, 1) + 1) / 2.0).cpu().numpy().squeeze()

    # Konwersja do uint8 (0-255)
    fake_u8 = (fake_np * 255).astype(np.uint8)

    # Tworzenie obrazka (Grayscale)
    img = Image.fromarray(fake_u8, mode="L")
    img.save(out_dir / filename)

# ==============================================================================
# 3. GŁÓWNA PĘTLA
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- KONFIGURACJA ---
    # Wklej tu ścieżkę do swojego wytrenowanego modelu
    ckpt_path = ROOT_DIR / "runs/2025_12_01_163938_lidc_vector_diffusion/epoch=59-step=53640.ckpt"
    
    num_images_to_generate = 5000  # Ile obrazków łącznie
    batch_size = 16               # Szybciej niż 1
    steps = 50                    # Liczba kroków samplowania
    guidance_scale = 2.0          # Siła warunkowania (CFG)

    out_dir = Path.cwd() / "synthetic_only_results_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ładowanie Modelu
    if not ckpt_path.exists():
        print(f"BŁĄD: Brak pliku {ckpt_path}")
        sys.exit(1)
        
    print(f"Ładuję model...")
    pipeline = VectorDiffusionPipeline.load_from_checkpoint(str(ckpt_path))
    pipeline.to(device)
    pipeline.eval()
    
    # Fix flag (na wypadek gdyby checkpoint ich nie zapisał)
    pipeline.do_input_centering = False
    pipeline.clip_x0 = False

    # 2. Ładowanie Datasetu Testowego (jako źródła wektorów)
    # Nadal używamy datasetu, żeby brać z niego sensowne współrzędne guzków,
    # a nie losować ich w ciemno.
    print(f"Ładuję zbiór testowy...")
    root_dir_data = ROOT_DIR / "../dataset_lidc_2d_seg"
    ds_test = LIDCVectorDataset(root_dir=root_dir_data, split="test")
    
    loader = DataLoader(
        ds_test, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )

    print(f"Start generowania {num_images_to_generate} obrazów...")
    
    generated_count = 0
    pbar = tqdm(total=num_images_to_generate)

    with torch.no_grad():
        for batch in loader:
            if generated_count >= num_images_to_generate:
                break
            
            # Pobieramy TYLKO wektory (target)
            vectors = batch['target'].to(device) # [B, 6]
            
            current_batch_size = vectors.shape[0]
            latent_shape = (8, 32, 32) # Kształt latentów

            # CFG Setup
            un_cond = None
            if guidance_scale > 1.0:
                un_cond = torch.zeros_like(vectors)

            # Generowanie
            fake_ct_batch = pipeline.sample(
                num_samples=current_batch_size,
                img_size=latent_shape,
                condition=vectors,
                un_cond=un_cond,
                guidance_scale=guidance_scale,
                steps=steps
            )

            # Zapisywanie batcha
            for i in range(current_batch_size):
                if generated_count >= num_images_to_generate: break
                
                save_generated_image(
                    fake_ct_batch[i], 
                    vectors[i], 
                    generated_count, 
                    out_dir
                )
                generated_count += 1
                pbar.update(1)

    pbar.close()
    print(f"\nGotowe! Czyste, wygenerowane obrazy są w: {out_dir}")