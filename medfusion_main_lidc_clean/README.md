Trening i Ewaluacja VAE + Dyfuzji na LIDC

1. Trening VAE

Model trenowano na negatywnych slajsach LIDC.
Wyniki ewaluacji rekonstrukcji na 1000 losowych przykładach:

Wyniki rekonstrukcji VAE
LPIPS (1 − LPIPS):   0.9787
MS-SSIM:             0.9888 ± 0.0041
MSE:                 0.000301 ± 0.000167


2. Trening modelu dyfuzyjnego

Model dyfuzyjny był trenowany na tych samych negatywnych przykładach.
Następnie wygenerowano syntetyczne slajsy CT.

Przykładowe wygenerowane obrazy
<img width="1034" height="260" alt="sample_103" src="https://github.com/user-attachments/assets/6aa015b6-95c9-4ee8-b827-8ce9e5230b48" />


(miejsce na obrazki)

3. Ewaluacja jakości obrazów dyfuzyjnych

Ocena odbyła się w porównaniu do 1000 realnych slajsów LIDC (negatywnych).
Użyto:

FID (Fréchet Inception Distance)

Improved Precision/Recall (Gustav Skibbe implementation)

PSNR / SSIM (porównanie obrazów parami – batchowa analiza)

3.1. Metryki generatywne (FID / PR)
FID:        48.03
Precision:   0.0938
Recall:      0.4500

3.2. PSNR / SSIM (porównanie jakości per-sample)
PSNR:  23.58 dB
SSIM:   0.741


5. Podsumowanie
Rekonstrukcje VAE

VAE nauczył się perfekcyjnie rekonstrukcji zdrowych slajsów — jakość niemal identyczna z oryginałem.

Model dyfuzyjny

Pierwszy trening działa, generuje sensowne CT, ale wyniki metryk wskazują:

FID do poprawienia,

Precision jest niskie → zdarzają się artefakty,

Recall jest już całkiem dobry (~45%) → model reprodukuje sporą część rozkładu.
