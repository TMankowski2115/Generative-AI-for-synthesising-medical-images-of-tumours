# train_latent_embedder_2d_lidc.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print("DBG ROOT_DIR:", ROOT_DIR)

from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from medical_diffusion.data.datasets.dataset_lidc_ct_2d import SimpleDataset2D
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.embedders.latent_embedders import VAE

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":

    # --------------- Ustawienia ogólne --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / f"{current_time}_lidc_ct_vae"
    path_run_dir.mkdir(parents=True, exist_ok=True)

    gpus = [1] if torch.cuda.is_available() else None
    print(f"Używam GPU: {gpus}")

    

    # --------------- Dataset LIDC (train + val, tylko bez guzków) ---------------
    root_dir = Path("../dataset_lidc_2d_seg")

    ds_train = SimpleDataset2D(root_dir=root_dir, split="train", fraction=0.3)  # has_nodule = false
    ds_val   = SimpleDataset2D(root_dir=root_dir, split="val", fraction=0.3)    # has_nodule = false

    dm = SimpleDataModule(
        ds_train=ds_train,
        ds_val=ds_val,         # <–– ważne: przekazujemy walidację
        batch_size=16,
        pin_memory=True,
        num_workers=4,
    )

    # --------------- Model VAE (1 kanał, 256x256) ----------------
    model = VAE(
        in_channels=1,
        out_channels=1,
        emb_channels=8,
        spatial_dims=2,
        hid_chs=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        deep_supervision=1,
        use_attention='none',
        loss=torch.nn.MSELoss,       # wtedy nazwa metryki zwykle "train/MSE" / "val/MSE"
        embedding_loss_weight=1e-6,
    )

    print("Cuda available:", torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)

    # -------------- Monitorowanie i callbacki ---------------------
    # Po pierwszym odpaleniu zerknij w logi Lightninga:
    # jeśli zobaczysz np. "val/L1", zmień to_monitor na "val/L1".
    to_monitor = "val/L1"
    min_max = "min"
    save_and_sample_every = 50

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,
        patience=20,      # ile epok bez poprawy zanim stop
        mode=min_max,
    )

    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        every_n_train_steps=None,     # przy walidacji lepiej zapisywać po epokach
        save_last=True,
        save_top_k=3,
        mode=min_max,
    )

    # -------------- Trainer z walidacją ---------------------------
    trainer = Trainer(
        accelerator='gpu' if gpus is not None else 'cpu',
        devices=gpus if gpus is not None else None,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,

        check_val_every_n_epoch=1,   # walidacja co epokę
        log_every_n_steps=save_and_sample_every,

        # limit_train_batches=100,   # NA START: tylko 100 batchy na epokę
        # limit_val_batches=10,      # np. 10 batchy walidacyjnych

        limit_val_batches=1.0,       # użyj całego ds_val (ułamek, np. 0.25, też działa)
        min_epochs=50,
        max_epochs=500,

        # min_epochs=1,
        # max_epochs=10,
        num_sanity_val_steps=2,
        # precision=16,
    )

    # ------------------- Start treningu ---------------------------
    trainer.fit(model, datamodule=dm)

    # ------------- Zapis ścieżki do najlepszego modelu ------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    print("Najlepszy checkpoint:", checkpointing.best_model_path)
