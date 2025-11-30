# Trening i Ewaluacja VAE + Modelu Dyfuzyjnego na LIDC

## 1. Trening VAE

Model VAE trenowano wyłącznie na **negatywnych** (bezguzkowych) slajsach z datasetu **LIDC**.

### Wyniki rekonstrukcji VAE  
Ewaluacja na 1000 losowych przykładach:

| Metryka       | Wynik |
|---------------|--------|
| **LPIPS (1 − LPIPS)** | **0.9787** |
| **MS-SSIM** | **0.9888 ± 0.0041** |
| **MSE** | **0.000301 ± 0.000167** |

Rekonstrukcje są niemal perfekcyjne — model dobrze nauczył się rozkładu zdrowych CT.

---

## 2. Trening modelu dyfuzyjnego

Model dyfuzyjny również trenowano na tych samych **negatywnych** slajsach LIDC.

Po treningu wygenerowano syntetyczne obrazy CT.

### Przykładowe wygenerowane obrazy

<img width="1034" height="260" alt="sample_103" src="https://github.com/user-attachments/assets/6aa015b6-95c9-4ee8-b827-8ce9e5230b48" />

---

## 3. Ewaluacja jakości obrazów dyfuzyjnych

Ocena została przeprowadzona względem **1000 realnych negatywnych slajsów LIDC**.

Użyte metryki:

- **FID** (Fréchet Inception Distance)  
- **Improved Precision/Recall** (implementacja Gustava Skibbe)  
- **PSNR / SSIM** (porównanie parami)

---

## 3.1. Metryki generatywne (FID / Precision / Recall)

| Metryka | Wynik |
|---------|--------|
| **FID** | **48.03** |
| **Precision** | **0.0938** |
| **Recall** | **0.4500** |

---

## 3.2. Metryki PSNR / SSIM (porównania per-sample)

| Metryka | Wynik |
|---------|--------|
| **PSNR** | **23.58 dB** |
| **SSIM** | **0.741** |

---
