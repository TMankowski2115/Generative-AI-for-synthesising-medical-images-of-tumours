# train_diffusion.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import math
import heapq
import glob
import re

from lidc_diffusion_dataset import LIDCDiffusionDataset
from lidc_controlnet_model import LIDCControlNetUNet
from sampler_utils import make_oversampling_sampler
from diffusion_utils import DiffusionSchedule

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.utils import make_grid, save_image


# =====================================================================
# CHECKPOINT MANAGER (TOP 3 + LAST)
# =====================================================================
class CheckpointManager:
    """Zapisuje tylko N najlepszych modeli (FID) oraz zawsze model 'last'."""
    def __init__(self, save_dir, max_to_keep=3, mode='min'):
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        self.mode = mode
        self.top_k = [] 

    def save(self, model_state, optimizer_state, ema_state, epoch, score):
        # 1. Zapisz Top K (z nazwą epoki i FID)
        ckpt_path = os.path.join(self.save_dir, f"model_epoch_{epoch:03d}_fid{score:.2f}.pt")
        
        torch.save({
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "ema_state": ema_state,
            "score": score
        }, ckpt_path)
        
        # Zarządzanie listą Top K
        entry = (score, epoch, ckpt_path)
        self.top_k.append(entry)
        
        reverse = True if self.mode == 'max' else False
        self.top_k.sort(key=lambda x: x[0], reverse=reverse)
        
        if len(self.top_k) > self.max_to_keep:
            worst = self.top_k.pop() 
            if os.path.exists(worst[2]):
                os.remove(worst[2])
                print(f"[Checkpoints] Usunięto słabszy model: {worst[2]}")

        print(f"[Checkpoints] Top {self.max_to_keep}: {[f'Ep{x[1]}:{x[0]:.2f}' for x in self.top_k]}")

    def save_last(self, model_state, optimizer_state, ema_state, epoch, score):
        # 2. Zapisz Safety Save (model_last.pt) - ZAWSZE nadpisuje
        last_path = os.path.join(self.save_dir, "model_last.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "ema_state": ema_state,
            "score": score
        }, last_path)
        print(f"[Safety] Zaktualizowano model_last.pt (Epoka {epoch})")


# =====================================================================
# EMA CLASS
# =====================================================================
class EMA:
    def __init__(self, model, beta=0.9999):
        self.beta = beta
        self.step = 0
        self.ema_state = {
            k: v.clone().detach().to(v.device)
            for k, v in model.state_dict().items()
        }

    def update(self, model):
        self.step += 1
        with torch.no_grad():
            model_state = model.state_dict()
            for k, v in self.ema_state.items():
                if k in model_state:
                    current_val = model_state[k].to(v.device)
                    v.mul_(self.beta).add_(current_val, alpha=1 - self.beta)

    def copy_to(self, model):
        current_state = model.state_dict()
        new_state = {
            k: v if k in self.ema_state else current_state[k]
            for k, v in self.ema_state.items()
        }
        model.load_state_dict(new_state)


# =====================================================================
# UTILS
# =====================================================================

def next_run_dir(base_dir="lidc_diffusion_ckpts_lungs_only/runs"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    new_idx = 1 if not existing else max(int(x.split("_")[1]) for x in existing) + 1
    run_dir = os.path.join(base_dir, f"run_{new_idx:03d}")
    os.makedirs(run_dir)
    return run_dir


def ct_to_rgb01(ct):
    x = (ct + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0).repeat(1, 3, 1, 1)


# =====================================================================
# DDPM SAMPLING
# =====================================================================

@torch.no_grad()
def ddpm_sample(model, schedule, lung_mask, nodule_mask, cond_vec,
                device, num_steps=None, cfg_scale=3.0):

    model.eval()
    B, _, H, W = lung_mask.shape

    if num_steps is None:
        timesteps = torch.arange(schedule.T - 1, -1, -1, device=device)
    else:
        timesteps = torch.linspace(
            schedule.T - 1, 0, steps=num_steps, device=device, dtype=torch.long
        )

    x = torch.randn(B, 1, H, W, device=device)

    lung_zero = torch.zeros_like(lung_mask)
    nod_zero = torch.zeros_like(nodule_mask)
    cond_zero = torch.zeros_like(cond_vec)

    for ti in timesteps:
        t = torch.full((B,), int(ti.item()), device=device, dtype=torch.long)

        eps_cond = model(x, t, lung_mask, nodule_mask, cond_vec)
        # Unconditional generation (wszystko wyzerowane)
        eps_uncond = model(x, t, lung_zero, nod_zero, cond_zero)
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        beta = schedule.betas[t][:, None, None, None]
        alpha = schedule.alphas[t][:, None, None, None]
        alpha_bar = schedule.alphas_cumprod[t][:, None, None, None]

        if ti > 0:
            alpha_bar_prev = schedule.alphas_cumprod[t - 1][:, None, None, None]
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar)

        sigma_t = torch.sqrt(beta * (1 - alpha_bar_prev) / (1 - alpha_bar))

        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * eps
        )

        noise = sigma_t * torch.randn_like(x) if ti > 0 else 0.0
        x = mean + noise

    return x


# =====================================================================
# EVALUATION (Validation Loss + Metrics)
# =====================================================================

@torch.no_grad()
def quick_epoch_eval(model, schedule, eval_loader, device, epoch):
    model.eval()
    
    # Metryki generatywne
    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception = InceptionScore(normalize=True).to(device)

    psnr_vals, ssim_vals = [], []
    
    # Dodano: Akumulacja Validation Loss
    val_loss_accum = 0.0
    val_batches = 0
    
    mse = nn.MSELoss()
    max_batches = 50 

    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break

        ct = batch["ct"].to(device)
        lung_raw = batch["lung_mask"].to(device)
        lung_left = (lung_raw == 1).float()
        lung_right = (lung_raw == 2).float()
        lung_two_ch = torch.cat([lung_left, lung_right], dim=1)
        
        nodule = batch["nodule_mask"].to(device)
        cond_vec = batch["cond_vector"].to(device)

        B = ct.size(0)
        t = torch.randint(0, schedule.T, (B,), device=device)

        x_t, noise = schedule.q_sample(ct, t)
        
        # 1. Obliczenie Validation Loss (MSE) - Forward Pass
        #    Podajemy pełne warunki (bez dropoutu) dla sprawdzenia wierności modelu
        noise_pred = model(x_t, t, lung_two_ch, nodule, cond_vec)
        loss_val = mse(noise_pred, noise)
        val_loss_accum += loss_val.item()
        val_batches += 1

        # 2. Rekonstrukcja x0 do metryk wizualnych (FID, PSNR itp.)
        alpha_bar_t = schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1.5, 1.5)

        psnr_vals.append(peak_signal_noise_ratio(x0_pred, ct, data_range=2.0))
        ssim_vals.append(structural_similarity_index_measure(x0_pred, ct, data_range=2.0))

        real_rgb = ct_to_rgb01(ct)
        fake_rgb = ct_to_rgb01(x0_pred)

        fid.update(real_rgb, real=True)
        fid.update(fake_rgb, real=False)
        inception.update(fake_rgb)

    # Obliczenia końcowe
    avg_val_loss = val_loss_accum / max(1, val_batches)
    
    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    psnr_mean = torch.stack(psnr_vals).mean().item()
    ssim_mean = torch.stack(ssim_vals).mean().item()

    print(
        f"[Eval @ epoch {epoch}] "
        f"Val_Loss={avg_val_loss:.5f} | "
        f"FID={fid_score:.2f} | IS={is_mean:.2f} | "
        f"PSNR={psnr_mean:.2f} | SSIM={ssim_mean:.3f}"
    )

    return {
        "Val_Loss": avg_val_loss,
        "FID": fid_score,
        "IS_mean": is_mean.item(),
        "IS_std": is_std.item(),
        "PSNR": psnr_mean,
        "SSIM": ssim_mean
    }


# =====================================================================
# TRENING
# =====================================================================

def train():

    # --- KONFIGURACJA ---
    SLICES_ROOT = r"/home/s189030/raid/PB/dataset_lidc_2d_seg/slices"
    TRAIN_SPLIT = r"/home/s189030/raid/PB/dataset_lidc_2d_seg/splits/train.txt"

    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 16
    EPOCHS = 150
    LR = 5e-5 
    T_STEPS = 1000
    POS_FACTOR = 2  
    
    # --- DROPOUT CONFIG ---
    CFG_DROP = 0.2         # Prawdopodobieństwo wyrzucenia WSZYSTKIEGO (trening bezwarunkowy)
    NODULE_DROP = 0.5      # NOWOŚĆ: Prawdopodobieństwo wyrzucenia SAMEJ MASKI GUZKA (uczenie z wektora)
    
    SAVE_DIR = "lidc_diffusion_ckpts_large" 
    SAMPLE_DIR = "lidc_samples_large"
    
    DEVICE_INDEX = 3
    if torch.cuda.is_available():
        if torch.cuda.device_count() > DEVICE_INDEX:
            DEVICE = torch.device(f"cuda:{DEVICE_INDEX}")
        else:
            print(f"[WARN] Karta cuda:{DEVICE_INDEX} niedostępna. Używam cuda:0.")
            DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    
    print("Using:", DEVICE)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    run_dir = next_run_dir(os.path.join(SAVE_DIR, "runs"))
    print(f"[Info] Run directory: {run_dir}")

    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            # Dodałem "Val_Loss" do nagłówka
            csv_writer.writerow(["Epoch", "Avg_Train_Loss", "Val_Loss", "FID", "IS_mean", "IS_std", "PSNR", "SSIM"])
    print(f"[Info] Logging metrics to: {csv_path}")

    # Dataset
    dataset = LIDCDiffusionDataset(SLICES_ROOT, TRAIN_SPLIT, IMAGE_SIZE)
    sampler = make_oversampling_sampler(dataset, pos_factor=POS_FACTOR)

    train_loader = DataLoader(
        dataset, BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True
    )
    eval_loader = DataLoader(
        dataset, EVAL_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    # --- MODEL ---
    print("[INIT] Tworzenie modelu Large (ch=128 + Attention)...")
    model = LIDCControlNetUNet(
        base_channels=128, 
        emb_dim=256,
        cond_dim=6
    ).to(DEVICE)

    ckpt_manager = CheckpointManager(SAVE_DIR, max_to_keep=3, mode='min')
    ema = EMA(model, beta=0.9999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    schedule = DiffusionSchedule(T=T_STEPS, device=DEVICE)

    # --- INTELIGENTNE WZNAWIANIE ---
    current_epoch = 1
    resume_path = None
    last_ckpt_path = os.path.join(SAVE_DIR, "model_last.pt")
    
    if os.path.exists(last_ckpt_path):
        print(f"[RESUME] Wykryto 'model_last.pt'.")
        resume_path = last_ckpt_path
    else:
        pattern = os.path.join(SAVE_DIR, "model_epoch_*.pt")
        ckpts = glob.glob(pattern)
        if ckpts:
            def get_epoch_from_filename(fname):
                match = re.search(r"model_epoch_(\d+)_", os.path.basename(fname))
                return int(match.group(1)) if match else 0
            latest_ckpt = max(ckpts, key=get_epoch_from_filename)
            print(f"[RESUME] Nie znaleziono 'model_last.pt', wczytuję najnowszy: {latest_ckpt}")
            resume_path = latest_ckpt
        else:
            print("[INIT] Brak checkpointów. Start od Epoki 1.")

    if resume_path:
        try:
            ckpt = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "ema_state" in ckpt:
                ema.ema_state = ckpt["ema_state"]
            current_epoch = ckpt["epoch"] + 1
            print(f"[RESUME] Sukces! Wznawiam trening od Epoki {current_epoch}.")
        except Exception as e:
            print(f"[ERROR] Błąd podczas wczytywania checkpointu: {e}")
            current_epoch = 1

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(current_epoch, EPOCHS + 1):

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        epoch_loss = 0.0

        for batch in pbar:
            ct = batch["ct"].to(DEVICE) 
            lung_raw = batch["lung_mask"].to(DEVICE)
            lung_left = (lung_raw == 1).float()
            lung_right = (lung_raw == 2).float()
            lung_two_ch = torch.cat([lung_left, lung_right], dim=1)
            
            nodule = batch["nodule_mask"].to(DEVICE)
            cond_vec = batch["cond_vector"].to(DEVICE) 

            B = ct.size(0)
            t = torch.randint(0, schedule.T, (B,), device=DEVICE)
            x_t, noise = schedule.q_sample(ct, t)

            # --- NOWY SYSTEM DROPOUTU ---
            # 1. Losujemy czy robimy pełny CFG Drop (ukrywa wszystko)
            # Shape: [B, 1, 1, 1]
            cfg_drop_mask = (torch.rand(B, 1, 1, 1, device=DEVICE) < CFG_DROP).float()
            cfg_drop_vec = cfg_drop_mask[:, 0, 0, 0].unsqueeze(1)

            # 2. Losujemy czy robimy SPECYFICZNY Nodule Mask Drop
            # (Ukrywa tylko maskę guzka, zmuszając model do patrzenia na wektor)
            nodule_only_drop_mask = (torch.rand(B, 1, 1, 1, device=DEVICE) < NODULE_DROP).float()

            # Maska guzka jest zerowana, jeśli:
            # - Wypadł CFG drop (wszystko zerujemy)
            # - LUB wypadł Nodule Drop (tylko guzek zerujemy)
            final_nodule_drop = torch.max(cfg_drop_mask, nodule_only_drop_mask)

            # Aplikacja dropoutów
            lung_in = lung_two_ch * (1 - cfg_drop_mask)   # Płuca znikają tylko przy CFG
            cond_in = cond_vec * (1 - cfg_drop_vec)       # Wektor znika tylko przy CFG
            nodule_in = nodule * (1 - final_nodule_drop)  # Guzek znika przy CFG LUB NoduleDrop

            # Forward
            noise_pred = model(x_t, t, lung_in, nodule_in, cond_in)

            # Loss: MSE + SSIM
            alpha_bar = schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            
            ssim_loss = 1 - structural_similarity_index_measure(
                x0_pred.detach(), ct, data_range=2.0
            )

            loss = mse(noise_pred, noise) + 0.1 * ssim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ema.update(model)
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch}] avg_train_loss={avg_loss:.5f}")
        
        # --- EVAL ---
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        
        # Tutaj obliczany jest teraz też Validation Loss
        metrics = quick_epoch_eval(model, schedule, eval_loader, DEVICE, epoch)
        
        with open(csv_path, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch, 
                f"{avg_loss:.5f}", 
                f"{metrics['Val_Loss']:.5f}",  # Zapis Val Loss
                f"{metrics['FID']:.4f}", 
                f"{metrics['IS_mean']:.4f}", 
                f"{metrics['IS_std']:.4f}", 
                f"{metrics['PSNR']:.4f}", 
                f"{metrics['SSIM']:.4f}"
            ])

        # --- SAMPLE ---
        with torch.no_grad():
            sample = [dataset[i] for i in range(8)]
            ct = torch.stack([s["ct"] for s in sample]).to(DEVICE)
            lung_raw = torch.stack([s["lung_mask"] for s in sample]).to(DEVICE)
            lung_left = (lung_raw == 1).float()
            lung_right = (lung_raw == 2).float()
            lung_two_ch = torch.cat([lung_left, lung_right], dim=1)
            
            nod = torch.stack([s["nodule_mask"] for s in sample]).to(DEVICE)
            cond = torch.stack([s["cond_vector"] for s in sample]).to(DEVICE)

            # Sampling standardowy (z pełnymi maskami, żeby zobaczyć czy model umie)
            gen = ddpm_sample(model, schedule, lung_two_ch, nod, cond, DEVICE)
            
            # Opcjonalnie: można dodać sampling z ukrytym guzkiem (nod=zeros), 
            # by sprawdzić czy wektor działa, ale na razie zostawiam standard.
            
            ct_rgb = ct_to_rgb01(ct)
            gen_rgb = ct_to_rgb01(gen)
            grid = make_grid(torch.cat([ct_rgb, gen_rgb], dim=0), nrow=8)
            save_image(grid, f"{SAMPLE_DIR}/epoch_{epoch:03d}.png")

        # --- SAVE CHECKPOINTS ---
        # 1. Top 3 (dla FID)
        ckpt_manager.save(
            model.state_dict(), optimizer.state_dict(), ema.ema_state, epoch, metrics['FID']
        )
        # 2. Safety Save (zawsze ostatni)
        ckpt_manager.save_last(
            model.state_dict(), optimizer.state_dict(), ema.ema_state, epoch, metrics['FID']
        )
        
        model.load_state_dict(original_weights)

    print("Training finished.")

if __name__ == "__main__":
    train()
