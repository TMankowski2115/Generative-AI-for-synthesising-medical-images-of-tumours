# train_latent_embedder_2d_lidc.py
import sys
from pathlib import Path

# Ustalanie ścieżki do roota projektu
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print("DBG ROOT_DIR:", ROOT_DIR)

from datetime import datetime
import torch
from torch.utils.data import Dataset # Potrzebne do Wrappera
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# IMPORTUJEMY TWÓJ NOWY DATASET
# Zakładam, że zapisałeś go w medical_diffusion/data/datasets/dataset_lidc_conditional.py
# Jeśli plik leży luźno obok skryptu, zmień import na: from dataset_lidc_conditional import LIDCConditionalDataset
from medical_diffusion.data.datasets.dataset_lidc_2d_features import LIDCConditionalDataset
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.embedders.latent_embedders import VAE

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# -----------------------------------------------------------------------------
# WRAPPER ADAPTUJĄCY (Kluczowy element!)
# -----------------------------------------------------------------------------
class VAEAdapterDataset(Dataset):
    """
    Bierze dataset 2-kanałowy [CT, Blob] i zwraca tylko 1 kanał [CT].
    Dzięki temu VAE uczy się kompresować obraz, ignorując mapę lokalizacji (blob),
    która nie wymaga kompresji i będzie podana osobno do modelu dyfuzyjnego.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # data['source'] ma kształt [2, 256, 256] -> (CT, Blob)
        
        # Wycinamy tylko kanał 0 (CT) zachowując wymiar kanału [1, 256, 256]
        # VAE nie musi widzieć bloba, żeby nauczyć się rekonstruować obraz.
        ct_only = data['source'][0:1, :, :] 
        
        # Nadpisujemy source
        data['source'] = ct_only
        
        # Atrybuty i label zostawiamy (VAE ich nie używa, ale nie przeszkadzają)
        return data

# -----------------------------------------------------------------------------
# GŁÓWNY SKRYPT
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Ustawienia
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # Zmieniamy nazwę folderu, żeby wiedzieć że to model warunkowy (chociaż VAE jest ten sam)
    path_run_dir = Path.cwd() / 'runs' / f"{current_time}_lidc_vae_conditional_ready"
    path_run_dir.mkdir(parents=True, exist_ok=True)

    gpus = [2] if torch.cuda.is_available() else None # Zwykle [0] jeśli masz 1 GPU
    print(f"Używam GPU: {gpus}")

    # 2. Dataset LIDC (Nowy, z guzkami i blobami)
    # Wskazujemy na katalog dataset_lidc_2d_seg
    root_dir = Path("../dataset_lidc_2d_seg") 

    # Ładujemy pełny dataset (z guzkami i bez)
    # fraction=1.0 zalecane dla VAE, żeby nauczył się różnorodnych tekstur
    ds_train_full = LIDCConditionalDataset(root_dir=root_dir, split="train") 
    ds_val_full   = LIDCConditionalDataset(root_dir=root_dir, split="val")

    # 3. Owijamy w Adapter (To VAE widzi tylko CT)
    ds_train = VAEAdapterDataset(ds_train_full)
    ds_val   = VAEAdapterDataset(ds_val_full)

    # 4. DataModule
    dm = SimpleDataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=32,   # Możesz zwiększyć do 32/64 jeśli masz dużo VRAM (VAE jest lekki)
        pin_memory=True,
        num_workers=8,
    )

    # 5. Model VAE
    # WAŻNE: in_channels=1, bo Adapter podaje tylko CT.
    model = VAE(
        in_channels=1,     # Tylko CT
        out_channels=1,    # Rekonstrukcja tylko CT
        emb_channels=8,    # Wymiar latent space (zostawiamy 8)
        spatial_dims=2,
        hid_chs=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        deep_supervision=1,
        use_attention='none',
        loss=torch.nn.MSELoss, 
        embedding_loss_weight=1e-6,
    )

    print("Cuda available:", torch.cuda.is_available())

    # 6. Callbacki
    to_monitor = "val/L1"
    min_max = "min"
    
    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0001,
        patience=15,
        mode=min_max,
        verbose=True
    )

    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        save_last=True,
        save_top_k=2,
        mode=min_max,
        # filename='vae-{epoch:02d}-{val/loss:.4f}' # Czytelna nazwa pliku
    )

    # 7. Trainer
    trainer = Trainer(
        accelerator='gpu' if gpus is not None else 'cpu',
        devices=gpus if gpus is not None else None,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        
        # Parametry treningu
        limit_val_batches=1, # Sprawdzamy na całym walidacyjnym
        min_epochs=20,
        max_epochs=200,        # VAE zazwyczaj zbiega dość szybko
        num_sanity_val_steps=2,
        # precision=16,        # Możesz odkomentować dla szybszego treningu (Mixed Precision)
    )

    # 8. Start
    print("Rozpoczynam trening VAE...")
    trainer.fit(model, datamodule=dm)

    # Zapis
    print("Najlepszy checkpoint:", checkpointing.best_model_path)