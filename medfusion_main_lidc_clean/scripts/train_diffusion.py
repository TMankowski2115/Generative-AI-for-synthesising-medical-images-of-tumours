import sys
from pathlib import Path



from datetime import datetime

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print("DBG ROOT_DIR:", ROOT_DIR)

from medical_diffusion.data.datasets.dataset_lidc_ct_2d import SimpleDataset2D
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import TimeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    # --------------- Ogólne ustawienia --------------------
    ROOT_DIR = Path(__file__).resolve().parent.parent
    print("DBG ROOT_DIR:", ROOT_DIR)

    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = ROOT_DIR / 'runs' / f"{current_time}_lidc_ct_diffusion"
    path_run_dir.mkdir(parents=True, exist_ok=True)

    gpus = [1] if torch.cuda.is_available() else None
    print(f"Używam GPU: {gpus}")

    # --------------- Dataset LIDC (tylko negatywy) -------------------
    root_dir = ROOT_DIR / "../dataset_lidc_2d_seg"

    # Na start polecam fraction=0.1; jak będzie OK, zmień na 1.0
    ds_train = SimpleDataset2D(
        root_dir=root_dir,
        split="train",
        fraction=0.30,
    )

    dm = SimpleDataModule(
        ds_train=ds_train,
        batch_size=16,      # dostosuj do VRAM (RTX 3070 → 16 powinno być OK)
        pin_memory=True,
        num_workers=16,
    )

    # --------------- Latent embedder = Twój VAE ----------------------
    latent_embedder = VAE
    latent_embedder_checkpoint = ROOT_DIR / "runs" / "2025_11_18_222802_lidc_ct_vae" / "epoch=45-step=57362.ckpt"
    print("Ładuję latent embedder z:", latent_embedder_checkpoint)

    # --------------- Noise estimator (UNet w latentach) --------------
    # VAE ma emb_channels=8 → latent ma 8 kanałów, więc UNet in_ch=8, out_ch=8
    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {
        "emb_dim": 1024,   # jak w przykładach Medfusion (4 * 256)
    }

    noise_estimator = UNet
    noise_estimator_kwargs = {
        "in_ch": 8,
        "out_ch": 8,
        "spatial_dims": 2,
        "hid_chs": [256, 256, 512, 1024],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "time_embedder": time_embedder,
        "time_embedder_kwargs": time_embedder_kwargs,
        "cond_embedder": None,          # BRAK warunkowania
        "cond_embedder_kwargs": None,
        "deep_supervision": False,
        "use_res_block": True,
        "use_attention": "none",
    }

    # --------------- Noise scheduler (DDPM) ---------------------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        "timesteps": 1000,
        "beta_start": 0.002,
        "beta_end": 0.02,
        "schedule_strategy": "scaled_linear",
    }

    # --------------- Pipeline dyfuzyjny -------------------------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator,
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint=str(latent_embedder_checkpoint),

        estimator_objective="x_T",       # zostawiam jak w oryginalnym przykładzie Medfusion
        estimate_variance=False,
        use_self_conditioning=False,
        use_ema=False,

        # Brak warunkowania → CFG dropout = 0
        classifier_free_guidance_dropout=0.0,

        # CT i tak masz już w [-1,1] po VAE, więc:
        do_input_centering=False,
        clip_x0=False,

        sample_every_n_steps=1000,       # co ile kroków ma próbować generować sample
    )

    # --------------- Trening / Trainer --------------------------------
    to_monitor = "train/loss"
    min_max = "min"
    save_and_sample_every = 500

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,
        patience=30,
        mode=min_max,
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=3,
        mode=min_max,
    )

    trainer = Trainer(
        accelerator='gpu' if gpus is not None else 'cpu',
        devices=gpus if gpus is not None else None,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],  # możesz dodać early_stopping jak włączysz walidację
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every,
        auto_lr_find=False,

        # Na start polecam ograniczyć:
        limit_train_batches=1.0,   # np. 1000 batchy na epokę (albo 1.0 dla całości)
        limit_val_batches=0,        # brak walidacji na razie

        min_epochs=50,
        max_epochs=500,
        num_sanity_val_steps=2,
        # precision=16,             # możesz włączyć mixed precision, jak chcesz szybciej
    )

    # ---------------- Start treningu ---------------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Zapis najlepszego checkpointu --------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    print("Najlepszy checkpoint diffusion:", checkpointing.best_model_path)
