# CT Lung Mask → Image GAN (U-Net + SSIM)

Ten projekt trenuje model typu **pix2pix (U-Net Generator + PatchGAN Discriminator)** do generowania slajsów CT płuc na podstawie masek (`lung_mask` / `combined_mask`).  
Dodatkowo do funkcji straty używany jest **SSIM**, aby poprawić zachowanie struktury naczyń i ogólną jakość obrazu.

---

## Struktura danych

Wymagana struktura katalogów:
```text
dataset_lidc_masks/
  train/
    with_nodules/
      ..._slice_....png          # obraz CT (wejście do D, target dla G)
      ..._combined_mask_....npy  # maska płuc+guzków (wejście do G)
    clean/
      ..._slice_....png          # obraz CT bez guzków
      ..._lung_mask_....npy      # maska płuc (bez guzków)
  test/
    with_nodules/
      ...
    clean/
      ...
	  
Format nazw:

- obraz CT:  
  `LIDC-IDRI-XXXX_slice_YYYY.png`
- maska płuc:  
  `LIDC-IDRI-XXXX_lung_mask_YYYY.npy`
- maska płuc + guzki:  
  `LIDC-IDRI-XXXX_combined_mask_YYYY.npy`


## Architektura sieci

### Generator – U-Net

Generator to 6-poziomowy **U-Net**, który konwertuje 1-kanałową maskę na 1-kanałowy obraz CT.

#### Enkoder (downsampling)

- 6 bloków `Conv2d(k=4, s=2)` → redukcja rozdzielczości: 128 → 64 → 32 → … → 2
- Kanały:  
  `1 → 64 → 128 → 256 → 512 → 512 → 512`
- Normalizacja: `InstanceNorm2d` od poziomu 2  
- Aktywacja: `LeakyReLU(0.2)`
- Bottleneck: rozmiar `2×2`

#### Dekoder (upsampling)

- 5 bloków `ConvTranspose2d(k=4, s=2)`
- Skip-connections (`concat`) z odpowiadającymi warstwami enkodera
- Dropout (0.5) w pierwszych 3 blokach
- Ostatnia warstwa: `Tanh` (wyjście w zakresie `[-1, 1]`)

#### Zalety U-Net:

- **skip-connections zachowują drobne struktury**, np. naczynia płucne  
- umożliwia odtwarzanie stabilnych detali z wczesnych warstw  
- dużo lepiej radzi sobie z medycznymi obrazami niż zwykły autoencoder

---

### Dyskryminator – PatchGAN

PatchGAN nie zwraca jednej liczby, tylko **mapę wiarygodności** — każdy element mapy ocenia lokalny patch obrazu.

Struktura:

2 → 64 → 128 → 256 → 512 → 1

Warunki:

- Wejściem jest konkatenacja `(mask, image)` → 2 kanały
- Normalizacja: `InstanceNorm2d`
- Aktywacja: `LeakyReLU(0.2)`
- Ostateczna mapa jest mniejsza niż 128×128

PatchGAN wymusza:
- realistyczne tekstury CT,
- ostrość krawędzi,
- spójność lokalnych struktur.

---

## Funkcje straty

### Generator minimalizuje:

L_G = L_GAN + L_L1 + L_SSIM

- **L_GAN** – BCEWithLogitsLoss(D(mask, fake), 1)
- **L1 Loss** – duży współczynnik (100) wymusza zgodność pikselową
- **SSIM Loss** – poprawia strukturę i kontrasty lokalne

### Dyskryminator minimalizuje:

L_D = 0.5 * ( BCE(real, 1) + BCE(fake, 0) )


---

## Hiperparametry

| Parametr        | Wartość |
|----------------|---------|
| IMG_SIZE        | 128     |
| BATCH_SIZE      | 64      |
| EPOCHS          | 200     |
| LR              | 0.0002  |
| BETAS           | (0.5, 0.999) |
| LAMBDA_L1       | 100     |
| LAMBDA_SSIM     | 5       |
| LR decay        | od 100 epoki |

---

## Logowanie wyników

Co epokę wypisywane są:
- Loss_G
- Loss_D
- SSIM

Co 5 epok:
- wpis do `training_results_SSIM/training_log.txt`

Co 10 epok:
- zapis siatki przykładów (`epoch_XX.png`)
- checkpoint: `checkpoint_epoch_XX.pth`

---
