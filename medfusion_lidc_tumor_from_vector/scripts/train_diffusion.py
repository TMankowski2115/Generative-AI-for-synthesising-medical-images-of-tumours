import sys
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torchvision  # <--- WAŻNE: Potrzebne do zapisu podglądu
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Ustalanie ścieżki do root projektu
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print("DBG ROOT_DIR:", ROOT_DIR)

# Importy z biblioteki medical_diffusion
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import TimeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE

# Import datasetu
from medical_diffusion.data.datasets.dataset_lidc_vector import LIDCVectorDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# ==============================================================================
# 1. SCALAR EMBEDDER (Z poprawką self.emb_dim)
# ==============================================================================
class ScalarEmbedder(nn.Module):
    def __init__(self, num_scalars=6, emb_dim=1024):
        """
        num_scalars: 6 [has_nodule, x, y, size, malignancy, texture]
        """
        super().__init__()
        # !!! FIX: Zapisujemy emb_dim, bo UNet tego wymaga !!!
        self.emb_dim = emb_dim  
        
        self.net = nn.Sequential(
            nn.Linear(num_scalars, 256),
            nn.SiLU(),
            nn.Linear(256, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)

# ==============================================================================
# 2. CUSTOM PIPELINE Z PODGLĄDEM (Bezpieczna wersja)
# ==============================================================================
class VectorDiffusionPipeline(DiffusionPipeline):
    def on_validation_epoch_end(self):
        """
        Generuje podgląd po każdej epoce walidacji i zapisuje PNG na dysku.
        """
        if self.latent_embedder is None or self.noise_scheduler is None:
            return

        # 1. Warunki testowe: [has_nod, x, y, size, mal, tex]
        conditions = [
            # 1. Zdrowe płuco (Referencja)
            # POPRAWKA: Pozycja też 0.0, zgodnie z datasetem!
            [0.0, 0.00, 0.00, 0.00, 0.0, 0.0], 
            
            # 2. Mały, łagodny guzek (GGO) - Prawy Górny Płat
            [1.0, 0.75, 0.25, 0.06, 0.1, 0.2], 

            # 3. Średni guzek, neutralny - Lewy środek
            [1.0, 0.25, 0.50, 0.08, 0.5, 0.5], 

            # 4. Duży, bardzo złośliwy guz (Solid) - Prawy Dolny
            [1.0, 0.70, 0.70, 0.12, 0.9, 0.9], 
        ]
        
        cond_tensor = torch.tensor(conditions, device=self.device, dtype=torch.float32)
        latent_shape = (8, 32, 32) 
        
        # 2. Generowanie
        with torch.no_grad():
            imgs = self.sample(
                num_samples=len(conditions),
                img_size=latent_shape,
                condition=cond_tensor,
                guidance_scale=2.0, 
                steps=50            
            )
        
        # 3. Zapis PNG (omijamy logger medfusion)
        # Latenty z VAE dekodowane są do [-1, 1], więc normalizujemy do [0, 1] dla PNG
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0
        
        grid = torchvision.utils.make_grid(imgs, nrow=4, padding=2)
        
        # Folder zapisu
        try:
            log_dir = Path(self.logger.log_dir)
        except Exception:
            log_dir = Path("vis_samples_fallback")
            
        save_path = log_dir / "vis_samples"
        save_path.mkdir(parents=True, exist_ok=True)
        
        torchvision.utils.save_image(grid, save_path / f"epoch_{self.current_epoch:03d}.png")
        print(f"[Epoch {self.current_epoch}] Zapisano podgląd: {save_path}")

# ==============================================================================
# 3. GŁÓWNA KONFIGURACJA
# ==============================================================================
if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = ROOT_DIR / 'runs' / f"{current_time}_lidc_vector_diffusion"
    path_run_dir.mkdir(parents=True, exist_ok=True)

    gpus = [3] if torch.cuda.is_available() else None # Zmień indeks GPU jeśli trzeba (np. [3])
    print(f"Używam GPU: {gpus}")

    # --- 1. Dataset ---
    root_dir = ROOT_DIR / "../dataset_lidc_2d_seg" 

    ds_train = LIDCVectorDataset(root_dir=root_dir, split="train")
    ds_val   = LIDCVectorDataset(root_dir=root_dir, split="val")

    dm = SimpleDataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=32, # 32
        pin_memory=True,
        num_workers=8,
    )

    # --- 2. VAE ---
    # Podaj poprawną ścieżkę do checkpointu VAE
    latent_embedder_checkpoint = ROOT_DIR / "runs/2025_11_29_113618_lidc_vae_conditional_ready/loss=37652.2578.ckpt" 
    
    if not Path(latent_embedder_checkpoint).exists():
        print(f"UWAGA: Nie znaleziono VAE: {latent_embedder_checkpoint}")

    # --- 3. UNet ---
    base_ch = 256
    time_emb_dim = base_ch * 4 

    noise_estimator = UNet
    noise_estimator_kwargs = {
        "in_ch": 8, 
        "out_ch": 8, 
        "spatial_dims": 2,
        "hid_chs": [base_ch, base_ch, 512, 1024],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "time_embedder": TimeEmbbeding,
        "time_embedder_kwargs": {"emb_dim": time_emb_dim},
        
        # Wektor Warunkowy
        "cond_embedder": ScalarEmbedder,
        "cond_embedder_kwargs": {
            "num_scalars": 6, 
            "emb_dim": time_emb_dim 
        },
        "deep_supervision": False,
        "use_res_block": True,
        "use_attention": "linear", #linear
    }

    # --- 4. Pipeline (Vector) ---
    pipeline = VectorDiffusionPipeline(
        noise_estimator=noise_estimator,
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=GaussianNoiseScheduler,
        noise_scheduler_kwargs={
            "timesteps": 1000,
            "beta_start": 0.002, 
            "beta_end": 0.02, 
            "schedule_strategy": "scaled_linear"
        },
        latent_embedder=VAE,
        latent_embedder_checkpoint=str(latent_embedder_checkpoint),
        
        estimator_objective="x_T",
        classifier_free_guidance_dropout=0.3, # 0.3 / 0.5
        
        do_input_centering=False,     # Wyłączamy, bo latenty VAE nie są w [0,1]
        clip_x0=False,                # Wyłączamy, bo latenty VAE wychodzą poza [-1,1]
        sample_every_n_steps=10_000_000 # Duża liczba, żeby wyłączyć domyślny sampling
    )

    # --- 5. Trainer ---
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor="val/loss",
        save_last=True,
        save_top_k=2,
        mode="min",
    )

    trainer = Trainer(
        accelerator='gpu',
        devices=gpus,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=100,
        min_epochs=50,
        max_epochs=500,
    )

    print("Start treningu Vector Diffusion...")
    trainer.fit(pipeline, datamodule=dm)
    print("Najlepszy checkpoint:", checkpointing.best_model_path)