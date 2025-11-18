# LIDC-IDRI → 2D lung slices dataset

Skrypt buduje 2D-owy dataset slajsów płuc z guzkiem i bez guzka na podstawie oryginalnego LIDC-IDRI (DICOM + XML).

## Co robi skrypt?

- Szuka anotacji LIDC (`*.xml`) w katalogu `LIDC-IDRI/`.
- Dla każdej serii CT:
  - Wczytuje volumetrie CT (TYLKO modality = CT, z min. `MIN_SLICES`).
  - Segmentuje płuca modelem **lungmask R231** → `lung_mask.png` (0=bg, 1=left, 2=right).
  - (Stub) segmentuje płaty płuc → `lobes_mask.png` (0=bg, 1..5 = płaty).
  - Parsuje anotacje guzków (poligony 2D) z XML + uśrednia cechy LIDC po radiologach.
  - Filtruje slajsy:
    - min. pokrycie płuc `MIN_LUNG_RATIO`,
    - obecne oba płuca (1 i 2).
  - Tworzy osobny sample dla:
    - slajsu bez guzka (`*_neg`),
    - każdego guzka na slajsie (`*_nod-XX`).

## Struktura wyjściowa

Tworzy się katalog:

```text
dataset_lidc_2d_seg/
  slices/<patient_id>/<series_uid>/
    z_0142_nod-01/
      ct.png          # CT w oknie [-1000, 400], [0,1], zamaskowane do płuc ∪ guzek
      ct_hu.npy       # surowy slajs CT w HU (float32)
      lung_mask.png   # 0=bg, 1=left, 2=right
      lobes_mask.png  # 0=bg, 1..5 płaty (stub)
      nodule_mask.png # 0/1 – tylko ten KONKRETNY guzek
      labels.json     # metadane guzka/slajsu (średnie cechy LIDC, położenie, wymiar itd.)
      meta.json       # info techniczne (okno HU, semantyka masek, typ HU)
      prompt.txt      # krótki opis guzka po polsku
    z_0087_neg/
      ct.png
      ct_hu.npy
      lung_mask.png
      lobes_mask.png
      labels.json     # has_nodule=false
      meta.json
  splits/
    train.txt         # listy pacjentów wg splitu (po patient_id)
    val.txt
    test.txt
  patients.csv        # patient_id, split
