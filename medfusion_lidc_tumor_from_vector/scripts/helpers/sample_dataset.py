import sys
from pathlib import Path

# Ustalanie ścieżki do roota projektu
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

# --- Importy biblioteki medical_diffusion ---
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.embedders.latent_embedders import VAE
from medical_diffusion.data.datasets.dataset_lidc_2d_features import LIDCConditionalDataset

# --- Import klas modelu z Twojego skryptu treningowego ---
try:
    from scripts.train_diffusion_blob import ScalarEmbedder, ConditionalDiffusionPipeline
except ImportError:
    print("\n[BŁĄD IMPORTU] Nie znaleziono klas w 'scripts.train_diffusion'.")
    print("Upewnij się, że plik 'train_diffusion.py' istnieje w folderze 'scripts'")
    sys.exit(1)

# ==============================================================================
# 1. KLASA POMOCNICZA (Wrapper estymatora)
# ==============================================================================

class ModelWrapper(nn.Module):
    """
    Owijamy noise_estimator w klasę nn.Module.
    Dzięki temu PyTorch pozwoli na przypisanie go do pipeline.noise_estimator.
    """
    def __init__(self, original_estimator, blob_tensor):
        super().__init__()
        self.original_estimator = original_estimator
        self.blob_tensor = blob_tensor

    # ZMIANA: argument nazywa się 'condition', a nie 'cond'
    def forward(self, x, t, condition=None, **kwargs):
        # x: [B, 8, 32, 32] (latents ze schedulera)
        # self.blob_tensor: [B, 1, 32, 32] (mapa guzka)
        
        # 1. Doklejamy Bloba do latentów -> [B, 9, 32, 32]
        unet_input = torch.cat([x, self.blob_tensor], dim=1)
        
        # 2. Wywołujemy oryginalny UNet
        # ZMIANA: Przekazujemy 'condition=condition' (tak jak oczekuje UNet w medfusion)
        return self.original_estimator(unet_input, t, condition=condition, **kwargs)

# ==============================================================================
# 2. FUNKCJE ZAPISU
# ==============================================================================

def save_sample_grid(ct_fake, ct_real, blob, idx, out_dir, attributes=None):
    def denorm(x):
        # [-1, 1] -> [0, 1]
        return ((x.clamp(-1, 1) + 1) / 2.0).cpu().numpy()

    # 1. Pobieramy dane i usuwamy wymiar kanału (squeeze)
    # ct_fake[0] ma kształt [1, 256, 256] -> po squeeze [256, 256]
    fake_np = denorm(ct_fake[0]).squeeze()
    real_np = denorm(ct_real[0]).squeeze()
    blob_np = blob[0].cpu().numpy().squeeze() # Blob też wymaga squeeze

    # 2. Konwersja do uint8 (0-255)
    fake_u8 = (fake_np * 255).astype(np.uint8)
    real_u8 = (real_np * 255).astype(np.uint8)
    blob_u8 = (blob_np * 255).astype(np.uint8)

    # 3. Łączymy horyzontalnie: Heatmapa | Oryginał | AI Generator
    # Teraz łączymy macierze 2D, więc wynik też będzie 2D
    combined = np.hstack([blob_u8, real_u8, fake_u8])
    
    # 4. Tworzymy obraz PIL (teraz zadziała, bo combined jest 2D)
    img = Image.fromarray(combined, mode="L")
    
    # Budowanie nazwy pliku z atrybutów
    suffix = ""
    if attributes is not None:
        # attributes: [has_nodule, subt, calc, text, malig]
        is_nodule = attributes[0] > 0.5
        malig = attributes[4]
        suffix = f"_nod{int(is_nodule)}_mal{malig:.2f}"

    filename = f"sample_{idx:04d}{suffix}.png"
    img.save(out_dir / filename)

# ==============================================================================
# 3. GŁÓWNA PĘTLA
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- KONFIGURACJA ---
    ckpt_path = ROOT_DIR / "runs/2025_11_30_205717_lidc_conditional_diffusion/epoch=2-step=5364.ckpt"
    
    n_samples = 5
    steps = 500
    guidance_scale = 1.0 # >1.0 dla CFG
    
    out_dir = Path.cwd() / "synthetic_conditional_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Start Generowania Warunkowego ---")
    print(f"Model: {ckpt_path}")
    
    if not ckpt_path.exists():
        print(f"BŁĄD: Nie znaleziono pliku checkpointu: {ckpt_path}")
        sys.exit(1)

    # 1. Ładowanie modelu
    pipeline = ConditionalDiffusionPipeline.load_from_checkpoint(str(ckpt_path))
    pipeline.to(device)
    pipeline.eval()

    # 2. Ładowanie danych testowych (źródło blobów)
    root_dir_data = ROOT_DIR / "../dataset_lidc_2d_seg" 
    ds_test = LIDCConditionalDataset(root_dir=root_dir_data, split="test")
    loader = DataLoader(ds_test, batch_size=1, shuffle=True, num_workers=4)
    
    print(f"Rozpoczynam generowanie {n_samples} próbek...")
    generated_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if generated_count >= n_samples:
                break

            x_input = batch['source'].to(device)
            attrs = batch['attributes'].to(device)
            
            real_ct = x_input[:, 0:1, :, :]
            blob_orig = x_input[:, 1:2, :, :]

            # Skalowanie bloba do rozmiaru latentów (32x32)
            latent_h, latent_w = 32, 32
            blob_small = F.interpolate(blob_orig, size=(latent_h, latent_w), mode='bilinear')
            
            # --- ZAPAMIĘTANIE ORYGINALNEGO ESTYMATORA ---
            original_estimator = pipeline.noise_estimator
            
            # --- PODMIANA NA WRAPPER ---
            # Dzięki klasie ModelWrapper(nn.Module) PyTorch nie zgłosi błędu
            pipeline.noise_estimator = ModelWrapper(original_estimator, blob_small)
            
            try:
                # Obsługa CFG (Guidance)
                un_cond = None
                if guidance_scale > 1.0:
                    un_cond = torch.zeros_like(attrs)

                # --- SAMPLING ---
                # Używamy nazw argumentów zgodnych z definicją DiffusionPipeline:
                # def sample(self, num_samples, img_size, condition=None, **kwargs):
                
                fake_ct = pipeline.sample(
                    num_samples=1,                  # Zamiast batch_size
                    img_size=(8, latent_h, latent_w), # Zamiast latent_shape
                    condition=attrs,
                    un_cond=un_cond,              # Przekazane przez **kwargs do forward
                    guidance_scale=guidance_scale, # Przekazane przez **kwargs do forward
                    steps=steps                   # Przekazane przez **kwargs do denoise
                )

            finally:
                # --- PRZYWRACANIE ORYGINAŁU ---
                # Kluczowe, by w kolejnej iteracji nie owijać wrappera w wrapper
                pipeline.noise_estimator = original_estimator
            
            # Wyciągnięcie wyniku
            if isinstance(fake_ct, dict): fake_ct = fake_ct['sample']
            if isinstance(fake_ct, list): fake_ct = fake_ct[0]
            
            # Zapis
            save_sample_grid(fake_ct, real_ct, blob_orig, generated_count, out_dir, attrs[0])
            
            generated_count += 1
            print(f"[{generated_count}/{n_samples}] Wygenerowano...")

    print(f"\nGotowe! Wyniki zapisane w: {out_dir}")