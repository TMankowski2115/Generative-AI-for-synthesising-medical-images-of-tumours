import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lidc_diffusion_dataset import LIDCDiffusionDataset

ds = LIDCDiffusionDataset(
    "dataset_lidc_2d_seg/slices",
    "dataset_lidc_2d_seg/splits/train.txt",
    image_size=256
)

batch = next(iter(DataLoader(ds, batch_size=1, shuffle=True)))

ct = batch["ct"][0,0].cpu().numpy()
lung = batch["lung_mask"][0,0].cpu().numpy()
nod = batch["nodule_mask"][0,0].cpu().numpy()
path = batch["path"][0]
cond = batch["cond_vector"][0]

print("\n===== DEBUG INFO =====")
print("Path:", path)
print("CT range:", ct.min(), ct.max())
print("lung unique:", set(lung.flatten()))
print("nodule unique:", set(nod.flatten()))
print("cond vector:", cond)

# CT z powrotem do [0,1] żeby dało się wyświetlić
ct_disp = (ct + 1) / 2.0

overlay = ct_disp.copy()
overlay[nod > 0] = 1.0   # zaznacz guzek na biało

plt.figure(figsize=(12,3))
plt.subplot(1,4,1); plt.imshow(ct_disp, cmap='gray'); plt.title("CT")
plt.subplot(1,4,2); plt.imshow(lung, cmap='gray'); plt.title("Lung mask")
plt.subplot(1,4,3); plt.imshow(nod, cmap='gray'); plt.title("Nodule mask")
plt.subplot(1,4,4); plt.imshow(overlay, cmap='gray'); plt.title("Overlay")

plt.show()
