import torch
import numpy as np
import cv2
import os
from torchvision.utils import save_image

from lidc_diffusion_dataset import LIDCDiffusionDataset
from lidc_controlnet_model import LIDCControlNetUNet
from diffusion_utils import DiffusionSchedule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ct_to_rgb01(ct):
    """
    ct: [B,1,H,W] w [-1,1]
    """
    x = (ct + 1.0) / 2.0
    return x.clamp(0, 1).repeat(1, 3, 1, 1)


@torch.no_grad()
def ddpm_sample(model, schedule, lung_mask_2ch, nodule_mask, cond_vec,
                device, num_steps=None, cfg_scale=3.0):
    """
    model: LIDCControlNetUNet
    lung_mask_2ch: [B,2,H,W] (left,right)
    nodule_mask:   [B,1,H,W]
    cond_vec:      [B,5]
    """
    model.eval()
    B, _, H, W = lung_mask_2ch.shape

    if num_steps is None:
        timesteps = torch.arange(schedule.T - 1, -1, -1, device=device)
    else:
        timesteps = torch.linspace(schedule.T - 1, 0, steps=num_steps,
                                   device=device, dtype=torch.long)

    x_t = torch.randn(B, 1, H, W, device=device)

    lung_zero = torch.zeros_like(lung_mask_2ch)
    nod_zero = torch.zeros_like(nodule_mask)
    cond_zero = torch.zeros_like(cond_vec)

    for t_idx in timesteps:
        t = torch.full((B,), int(t_idx.item()), device=device, dtype=torch.long)

        eps_cond = model(x_t, t, lung_mask_2ch, nodule_mask, cond_vec)
        eps_uncond = model(x_t, t, lung_zero, nod_zero, cond_zero)

        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        beta_t = schedule.betas[t[0]]
        alpha_t = schedule.alphas[t[0]]
        alpha_bar_t = schedule.alphas_cumprod[t[0]]

        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)

        x_t = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps
        ) + torch.sqrt(beta_t) * noise

    return x_t


def draw_nodule_mask_from_lung_labels(lung_raw, side="left", diameter_mm=10):
    """
    lung_raw: [1,H,W] lub [H,W] — wartości 0,1,2 (0=bg, 1=left, 2=right)
    side: 'left' / 'right'
    Zwraca:
        nod_mask [1,H,W]  (0/1),
        cond_vec [5],
        (cx,cy,radius_px)
    """
    if lung_raw.ndim == 3:
        # [1,H,W] -> [H,W]
        lung_raw = lung_raw[0]

    H, W = lung_raw.shape
    radius_px = int((diameter_mm / 30.0) * 40)  # heurystyka z treningu

    lung_np = lung_raw.cpu().numpy()

    if side == "left":
        roi = (lung_np == 1)
    else:
        roi = (lung_np == 2)

    ys, xs = np.where(roi)
    if len(xs) == 0:
        raise ValueError("empty lung mask for selected side")

    i = np.random.choice(len(xs))
    cy, cx = ys[i], xs[i]

    nod_mask = np.zeros((H, W), dtype=np.float32)
    cv2.circle(nod_mask, (cx, cy), radius_px, 1.0, -1)

    cond_vec = torch.tensor([
        cx / W,
        cy / H,
        diameter_mm / 30.0,
        1.0 if side == "left" else 0.0,
        0.0 if side == "left" else 1.0
    ], dtype=torch.float32)

    return torch.tensor(nod_mask, dtype=torch.float32).unsqueeze(0), cond_vec, (cx, cy, radius_px)


def draw_bbox_on_tensor(image_tensor, cx, cy, r):
    """
    Rysuje czerwony bounding box na tensorze [1,3,H,W].
    Zwraca tensor [1,3,H,W].
    """
    img = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255)
    img = img.astype(np.uint8).copy()

    xmin = int(max(0, cx - r))
    xmax = int(min(img.shape[1] - 1, cx + r))
    ymin = int(max(0, cy - r))
    ymax = int(min(img.shape[0] - 1, cy + r))

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    img = torch.from_numpy(img.astype(np.float32) / 255.0)
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img


def choose_side_from_lung(lung_raw):
    """
    lung_raw: [1,H,W] lub [H,W] z wartościami 0/1/2
    Wybiera sensowną stronę (left/right) w zależności od tego,
    gdzie faktycznie jest płuco.
    """
    if lung_raw.ndim == 3:
        lung_raw = lung_raw[0]

    left_count = (lung_raw == 1).sum().item()
    right_count = (lung_raw == 2).sum().item()

    if left_count > 50 and right_count > 50:
        return np.random.choice(["left", "right"])
    elif left_count > 50:
        return "left"
    elif right_count > 50:
        return "right"
    else:
        raise ValueError("both lungs empty in this slice")

def pick_slice_with_side(dataset, side):
    """Zwraca indeks slice'a, który zawiera wybrane płuco (left/right)."""

    attempts = 0
    max_attempts = 5000

    while attempts < max_attempts:
        idx = np.random.randint(0, len(dataset))
        batch = dataset[idx]

        lung = batch["lung_mask"]  # (H,W), wartości 0/1/2
        # sprawdzamy czy jest dane płuco
        if side == "left" and (lung == 1).sum() > 50:
            return idx
        if side == "right" and (lung == 2).sum() > 50:
            return idx

        attempts += 1

    raise RuntimeError(f"Could not find slice with {side} lung after {max_attempts} attempts.")


def main():

    dataset = LIDCDiffusionDataset(
        r"C:\Users\elias\programs\cGAN model\dataset_lidc_2d_seg\slices",
        r"C:\Users\elias\programs\cGAN model\dataset_lidc_2d_seg\splits\test.txt",
        image_size=256
    )

    # negatywne sample (bez guzka)
    neg_indices = [i for i, f in enumerate(dataset.has_nodule_flags) if not f]

    # losujemy do skutku slice, w którym jest jakieś płuco
    while True:
        idx = np.random.choice(neg_indices)
        batch = dataset[idx]
        lung_raw = batch["lung_mask"]  # zakładamy [H,W] lub [1,H,W]

        try:
            side ="left" #choose_side_from_lung(lung_raw)
            break
        except ValueError:
            continue  # spróbuj inny slice

   

    # CT: [1,H,W] -> [1,1,H,W]
    ct = batch["ct"].unsqueeze(0).to(DEVICE)

    # lung_raw -> [1,H,W]
    if lung_raw.ndim == 2:
        lung_raw = lung_raw.unsqueeze(0)
    lung_raw = lung_raw.to(DEVICE)  # [1,H,W]

    # 2-kanałowa maska płuc [B,2,H,W]
    lung_left = (lung_raw == 1).float().unsqueeze(1)   # [1,1,H,W]
    lung_right = (lung_raw == 2).float().unsqueeze(1)  # [1,1,H,W]
    lung_2ch = torch.cat([lung_left, lung_right], dim=1)  # [1,2,H,W]

    # --- parametry guzka ---
    diameter = 4  # mm
    print(f"Wybrany indeks: {idx}, strona: {side}, diameter: {diameter} mm")
    # nodule mask z rzeczywistego labelu płuca (1 lub 2)
    nod_mask, cond_vec, (cx, cy, r) = draw_nodule_mask_from_lung_labels(
        lung_raw, side=side, diameter_mm=diameter
    )
    nodule = nod_mask.unsqueeze(0).to(DEVICE)      # [1,1,H,W]
    cond_vec = cond_vec.unsqueeze(0).to(DEVICE)    # [1,5]

    # ----- model + schedule -----
    model = LIDCControlNetUNet(base_channels=64, emb_dim=256, cond_dim=5).to(DEVICE)
    ckpt = torch.load(
        r"C:\Users\elias\programs\cGAN model\lidc_diffusion_ckpts_lungs_only\model_epoch_010.pt",
        map_location=DEVICE
    )
    model.load_state_dict(ckpt["model_state"])

    schedule = DiffusionSchedule(T=1000, device=DEVICE)

    # ----- generacja -----
    gen = ddpm_sample(model, schedule, lung_2ch, nodule, cond_vec, DEVICE)

    real_rgb = ct_to_rgb01(ct).cpu()
    gen_rgb = ct_to_rgb01(gen).cpu()

    gen_bbox = draw_bbox_on_tensor(gen_rgb, cx, cy, r)

    os.makedirs("results", exist_ok=True)
    save_image(real_rgb, "results/base_ct.png")
    save_image(gen_rgb, "results/generated_ct.png")
    save_image(gen_bbox, "results/generated_ct_bbox.png")

    print("Saved:")
    print(" - results/base_ct.png")
    print(" - results/generated_ct.png")
    print(" - results/generated_ct_bbox.png")


if __name__ == "__main__":
    main()
