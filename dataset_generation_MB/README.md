# LIDC-IDRI → 2D Lung Slice Dataset

This script builds a 2D slice-level lung dataset with and without nodules based on the original LIDC-IDRI dataset (DICOM + XML annotations).

## What the script does

- Scans all LIDC annotation files (`*.xml`) inside `LIDC-IDRI/`.
- For each CT series:
  - Loads the CT volume (ONLY modality = CT, with at least `MIN_SLICES` slices).
  - Segments lungs using **lungmask R231** → `lung_mask.png` (0=bg, 1=left, 2=right).
  - Segments lung lobes using a placeholder U-Net (stub) → `lobes_mask.png` (0=bg, 1..5 lobes).
  - Parses nodule annotations (2D polygon contours) from XML and averages all radiologist characteristics.
  - Filters slices based on:
    - minimum lung coverage: `MIN_LUNG_RATIO`,
    - both lungs present (labels 1 and 2 must appear).
  - Creates separate samples for:
    - slices without any nodule (`*_neg`),
    - each individual nodule present on a slice (`*_nod-XX`).

## Output structure

The script produces:

```text
dataset_lidc_2d_seg/
  slices/<patient_id>/<series_uid>/
    z_0142_nod-01/
      ct.png          # CT windowed to [-1000,400], normalized to [0,1], masked to lungs ∪ nodule
      ct_hu.npy       # raw HU slice, float32
      lung_mask.png   # 0=bg, 1=left lung, 2=right lung
      lobes_mask.png  # 0=bg, 1..5 lung lobes (stub)
      nodule_mask.png # 0/1 – only this specific nodule
      labels.json     # metadata (nodule geometry, LIDC averaged ratings, location, spacing, etc.)
      meta.json       # technical info (HU window, mask semantics, HU dtype)
      prompt.txt      # short natural-language clinical-style description (Polish)
    z_0087_neg/
      ct.png
      ct_hu.npy
      lung_mask.png
      lobes_mask.png
      labels.json     # has_nodule=false
      meta.json

  splits/
    train.txt         # train patient IDs
    val.txt
    test.txt

  patients.csv        # mapping: patient_id → split
