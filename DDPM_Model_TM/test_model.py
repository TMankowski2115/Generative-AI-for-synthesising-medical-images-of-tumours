import torch
from torch.utils.data import DataLoader

from lidc_diffusion_dataset import LIDCDiffusionDataset
from lidc_controlnet_model import LIDCControlNetUNet


def main():
    ds = LIDCDiffusionDataset(
        slices_root="dataset_lidc_2d_seg/slices",
        split_txt="dataset_lidc_2d_seg/splits/train.txt",
        image_size=256
    )

    loader = DataLoader(ds, batch_size=2, shuffle=True)

    batch = next(iter(loader))

    ct = batch["ct"]               # [B,1,256,256], w [-1,1]
    lung = batch["lung_mask"]      # [B,1,256,256]
    nod = batch["nodule_mask"]     # [B,1,256,256]
    cond = batch["cond_vector"]    # [B,5]

    B = ct.size(0)
    timesteps = torch.randint(0, 1000, (B,), dtype=torch.long)  # przykładowe t

    model = LIDCControlNetUNet(base_channels=64, emb_dim=256, cond_dim=5)

    with torch.no_grad():
        noise_pred = model(ct, timesteps, lung, nod, cond)

    print("\n===== MODEL DEBUG =====")
    print("Input ct shape:", ct.shape)
    print("Lung mask shape:", lung.shape)
    print("Nodule mask shape:", nod.shape)
    print("Cond vector shape:", cond.shape)
    print("Timesteps shape:", timesteps.shape)
    print("Output noise_pred shape:", noise_pred.shape)
    print("Output range min/max:", noise_pred.min().item(), noise_pred.max().item())


if __name__ == "__main__":
    main()
