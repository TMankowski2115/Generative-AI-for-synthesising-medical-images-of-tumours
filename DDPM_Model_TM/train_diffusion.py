# train_diffusion.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lidc_diffusion_dataset import LIDCDiffusionDataset
from lidc_controlnet_model import LIDCControlNetUNet
from sampler_utils import make_oversampling_sampler
from diffusion_utils import DiffusionSchedule

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter


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
    B, _, H, W = lung_mask.shape  # lung_mask: [B,2,H,W]

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
# QUICK EVAL
# =====================================================================

@torch.no_grad()
def quick_epoch_eval(model, schedule, eval_loader, device, writer, epoch):

    model.eval()
    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception = InceptionScore(normalize=True).to(device)

    psnr_vals, ssim_vals = [], []
    max_batches = 20

    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break

        ct = batch["ct"].to(device)             # [B,1,H,W]

        # lung_mask: [B,1,H,W] z wartościami 0/1/2
        lung_raw = batch["lung_mask"].to(device)   # [B,1,H,W]

        lung_left = (lung_raw == 1).float()        # [B,1,H,W]
        lung_right = (lung_raw == 2).float()       # [B,1,H,W]
        lung_two_ch = torch.cat([lung_left, lung_right], dim=1)  # [B,2,H,W]

        # CT tylko w płucach
        lung_union = (lung_left + lung_right).clamp(0, 1)        # [B,1,H,W]
        ct_masked = ct * lung_union                              # [B,1,H,W]

        nodule = batch["nodule_mask"].to(device)   # [B,1,H,W]
        cond_vec = batch["cond_vector"].to(device) # [B,5]

        B = ct.size(0)
        t = torch.randint(0, schedule.T, (B,), device=device)

        x_t, noise = schedule.q_sample(ct_masked, t)
        noise_pred = model(x_t, t, lung_two_ch, nodule, cond_vec)

        alpha_bar_t = schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        psnr_vals.append(peak_signal_noise_ratio(x0_pred, ct_masked, data_range=2.0))
        ssim_vals.append(structural_similarity_index_measure(x0_pred, ct_masked, data_range=2.0))

        real_rgb = ct_to_rgb01(ct_masked)
        fake_rgb = ct_to_rgb01(x0_pred)

        fid.update(real_rgb, real=True)
        fid.update(fake_rgb, real=False)
        inception.update(fake_rgb)

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    psnr_mean = torch.stack(psnr_vals).mean().item()
    ssim_mean = torch.stack(ssim_vals).mean().item()

    print(
        f"[Eval quick @ epoch {epoch}] "
        f"FID={fid_score:.2f} | IS={is_mean:.2f}±{is_std:.2f} | "
        f"PSNR={psnr_mean:.2f} | SSIM={ssim_mean:.3f}"
    )

    writer.add_scalar("eval_quick/FID", fid_score, epoch)
    writer.add_scalar("eval_quick/IS_mean", is_mean, epoch)
    writer.add_scalar("eval_quick/IS_std", is_std, epoch)
    writer.add_scalar("eval_quick/PSNR", psnr_mean, epoch)
    writer.add_scalar("eval_quick/SSIM", ssim_mean, epoch)


# =====================================================================
# TRENING
# =====================================================================

def train():

    SLICES_ROOT = "dataset_lidc_2d_seg/slices"
    TRAIN_SPLIT = "dataset_lidc_2d_seg/splits/train.txt"

    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 1e-4
    T_STEPS = 1000
    POS_FACTOR = 5
    CFG_DROP = 0.4

    SAVE_DIR = "lidc_diffusion_ckpts_lungs_only"
    SAMPLE_DIR = "lidc_sample_output_lungs_only"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", DEVICE)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    run_dir = next_run_dir(os.path.join(SAVE_DIR, "runs"))
    writer = SummaryWriter(run_dir)
    print(f"[TensorBoard] logging into: {run_dir}")

    # Dataset
    dataset = LIDCDiffusionDataset(SLICES_ROOT, TRAIN_SPLIT, IMAGE_SIZE)
    sampler = make_oversampling_sampler(dataset, pos_factor=POS_FACTOR)

    train_loader = DataLoader(
        dataset, BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True
    )
    eval_loader = DataLoader(
        dataset, 4, shuffle=True, num_workers=2, pin_memory=True
    )

    # Model
    model = LIDCControlNetUNet(
        base_channels=64,
        emb_dim=256,
        cond_dim=5
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    schedule = DiffusionSchedule(T=T_STEPS, device=DEVICE)

    global_step = 0

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(1, EPOCHS + 1):

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        epoch_loss = 0.0

        for batch in pbar:

            ct = batch["ct"].to(DEVICE)  # [B,1,H,W]

            # === MASKI PŁUC ===
            lung_raw = batch["lung_mask"].to(DEVICE)   # [B,1,H,W]

            lung_left = (lung_raw == 1).float()        # [B,1,H,W]
            lung_right = (lung_raw == 2).float()       # [B,1,H,W]
            lung_two_ch = torch.cat([lung_left, lung_right], dim=1)  # [B,2,H,W]

            # CT tylko w płucach
            lung_union = (lung_left + lung_right).clamp(0, 1)  # [B,1,H,W]
            ct_in = ct * lung_union                            # [B,1,H,W]

            # nodule mask (jeśli jest)
            nodule = batch["nodule_mask"].to(DEVICE)   # [B,1,H,W]

            cond_vec = batch["cond_vector"].to(DEVICE) # [B,5]

            B = ct.size(0)
            t = torch.randint(0, schedule.T, (B,), device=DEVICE)

            x_t, noise = schedule.q_sample(ct_in, t)

            # classifier-free dropout  (JEDNA MASKA [B,1,1,1])
            drop = (torch.rand(B, 1, 1, 1, device=DEVICE) < CFG_DROP).float()
            drop_vec = drop[:, 0, 0, 0].unsqueeze(1)  # [B,1]

            lung_in = lung_two_ch * (1 - drop)        # [B,2,H,W]
            nodule_in = nodule * (1 - drop)           # [B,1,H,W]
            cond_in = cond_vec * (1 - drop_vec)       # [B,5]

            # MODEL FORWARD
            noise_pred = model(x_t, t, lung_in, nodule_in, cond_in)

            # SSIM loss
            alpha_bar = schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            ssim_loss = 1 - structural_similarity_index_measure(
                x0_pred.detach(), ct_in, data_range=2.0
            )

            loss = mse(noise_pred, noise) + 0.1 * ssim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.5f}")
        writer.add_scalar("train/avg_epoch_loss", avg_loss, epoch)

        # EVAL
        quick_epoch_eval(model, schedule, eval_loader, DEVICE, writer, epoch)

        # SAMPLE GRID
        with torch.no_grad():
            sample = [dataset[i] for i in range(8)]

            ct = torch.stack([s["ct"] for s in sample]).to(DEVICE)          # [B,1,H,W]
            lung_raw = torch.stack([s["lung_mask"] for s in sample]).to(DEVICE)  # [B,1,H,W]

            lung_left = (lung_raw == 1).float()
            lung_right = (lung_raw == 2).float()
            lung_two_ch = torch.cat([lung_left, lung_right], dim=1)

            lung_union = (lung_left + lung_right).clamp(0, 1)
            ct_in = ct * lung_union

            nod = torch.stack([s["nodule_mask"] for s in sample]).to(DEVICE)  # [B,1,H,W]
            cond = torch.stack([s["cond_vector"] for s in sample]).to(DEVICE) # [B,5]

            gen = ddpm_sample(model, schedule, lung_two_ch, nod, cond, DEVICE)

            ct_rgb = ct_to_rgb01(ct_in)
            gen_rgb = ct_to_rgb01(gen)

            grid = make_grid(torch.cat([ct_rgb, gen_rgb], dim=0), nrow=8)
            save_image(grid, f"{SAMPLE_DIR}/epoch_{epoch:03d}.png")
            writer.add_image("samples/epoch", grid, epoch)

        ckpt_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"[Checkpoint saved → {ckpt_path}]")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    train()
