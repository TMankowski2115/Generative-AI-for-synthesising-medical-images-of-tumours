# -*- coding: utf-8 -*-
"""
LIDC-IDRI → dataset 2D (slice-level) w układzie:
slices/<patient_id>/<series_uid>/
  z_0142_nod-01/
    ct.png                 # CT po oknie [-1000,400], znormalizowany do [0,1], zamaskowany do płuc ∪ guzek
    ct_hu.npy              # surowy slajs CT w HU (float32) – bez utraty informacji
    lung_mask.png          # 0=bg, 1=left, 2=right (z lungmask R231) – slajsy bez 1 i 2 są odrzucane
    lobes_mask.png         # 0=bg, 1=LLL, 2=LUL, 3=RLL, 4=RML, 5=RUL (z U-net LTR-Lobes – stub)
    nodule_mask.png        # 0/1 - tylko ten KONKRETNY guzek (dla pozytywów)
    labels.json
    meta.json
    prompt.txt
  z_0087_neg/
    ct.png
    ct_hu.npy
    lung_mask.png
    lobes_mask.png
    labels.json            # has_nodule=false
    meta.json

Dodatkowo: splits/train.txt, splits/val.txt, splits/test.txt oraz patients.csv
"""

from pathlib import Path
import os, re, xml.etree.ElementTree as ET
import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer
from skimage.draw import polygon
from collections import defaultdict
import cv2
import json
from sklearn.model_selection import train_test_split

# ───────── KONFIG ─────────────────────────────────────────────────────────────
LIDC_ROOT = Path("LIDC-IDRI")                 # katalog z danymi LIDC
OUT_DIR = Path("dataset_lidc_2d_seg")             # katalog wyjściowy
NS = {'lidc': 'http://www.nih.gov'}           # namespace XML LIDC

# Splity (po PACJENCIE!)
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# Filtry / parametry techniczne
MIN_SLICES = 100
WINDOW = (-1000, 400)                         # okno HU → [0,1]
MIN_NODULE_PIXELS = 10                        # odrzucaj bardzo malutkie kontury 2D
SLICE_NAME_PAD = 4                            # z_XXXX
# Klucze INT (nie stringi)
LUNG_SEMANTICS = {0: "bg", 1: "left", 2: "right"}
LOBES_SEMANTICS = {
    0: "bg", 1: "left_lower", 2: "left_upper",
    3: "right_lower", 4: "right_middle", 5: "right_upper"
}
MIN_LUNG_RATIO = 0.15                         # minimalne pokrycie płuc w slajsie (0..1)

# Wszystkie cechy LIDC, które agregujemy średnią z ocen radiologów
LIDC_CHAR_KEYS = [
    "subtlety",
    "internalStructure",
    "calcification",   # skala 1..6
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy"
]

# ───────── MODELE ─────────────────────────────────────────────────────────────
inferer_lung = LMInferer(modelname="R231")    # 0=bg, zwykle 1/2 – ale odrzucamy slajsy bez obu płuc (1 i 2)

# ───────── UTIL ───────────────────────────────────────────────────────────────
def masked_ct01(ct01: np.ndarray, lung_slice: np.ndarray, nodule_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Zostawia tylko CT w obszarze płuc (1/2) oraz – jeśli podane – guzka (1).
    Reszta = 0. Wejście/wyjście: [0,1].
    """
    lung_bin = (lung_slice > 0).astype(np.uint8)         # 0/1
    if nodule_mask is not None:
        nod_bin = (nodule_mask > 0).astype(np.uint8)     # 0/1
        union = np.clip(lung_bin + nod_bin, 0, 1)        # 0/1
    else:
        union = lung_bin
    return ct01 * union.astype(np.float32)

def _find_any(root, *tags):
    want = {t.lower() for t in tags}
    for el in root.iter():
        if el.tag.split('}')[-1].lower() in want and el.text and el.text.strip():
            return el.text.strip()
    return None

def get_pid_uid(xml_path: Path):
    """Wydobądź PatientID i SeriesInstanceUID z XML (fallback regex)."""
    try:
        root = ET.parse(xml_path).getroot()
        pid = _find_any(root, "PatientID", "PatientId")
        suid = _find_any(root, "SeriesInstanceUID", "SeriesInstanceUid")
    except ET.ParseError:
        pid = suid = None
    if not pid:
        m = re.search(r"LIDC-IDRI-\d{4}", xml_path.as_posix())
        pid = m.group(0) if m else None
    if not suid:
        txt = xml_path.read_text("utf-8", errors="ignore")
        m = re.search(r"<\s*SeriesInstanceUid[^>]*>\s*([\d.]+)\s*<", txt, re.I)
        suid = m.group(1).strip() if m else None
    if not (pid and suid):
        raise ValueError(f"Brak PID/SUID w {xml_path}")
    return pid, suid

def find_dicom(root_dir: Path, pid: str, suid: str):
    """Znajdź katalog serii DICOM z pasującym SeriesInstanceUID."""
    patient_dir = root_dir / pid
    if not patient_dir.exists():
        return None
    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir(): continue
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir(): continue
            dcm_files = list(series_dir.glob("*.dcm"))
            if not dcm_files: continue
            try:
                r = sitk.ImageFileReader()
                r.SetFileName(str(dcm_files[0]))
                r.LoadPrivateTagsOn()
                r.ReadImageInformation()
                if r.HasMetaDataKey("0020|000e"):
                    file_suid = r.GetMetaData("0020|000e").strip()
                    if file_suid == suid:
                        return series_dir
            except Exception:
                continue
    return None

def read_ct_series(dcm_dir: Path):
    """Wczytaj serię CT z ujednoliconym rozmiarem slajsów; odrzuć inne modality."""
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dcm_dir))
    if len(files) < MIN_SLICES:
        return None
    # modality
    hdr = sitk.ImageFileReader()
    hdr.SetFileName(files[0]); hdr.LoadPrivateTagsOn(); hdr.ReadImageInformation()
    modality = hdr.GetMetaData("0008|0060") if hdr.HasMetaDataKey("0008|0060") else ""
    if modality.strip() != "CT":
        return None
    # jednolity rozmiar XY → akceptuj serię tylko jeśli większość slajsów ma ten sam rozmiar
    sizes = []
    for f in files:
        rr = sitk.ImageFileReader(); rr.SetFileName(f); rr.ReadImageInformation()
        sizes.append(rr.GetSize()[:2])
    mode = max(set(sizes), key=sizes.count)
    good = [f for f, s in zip(files, sizes) if s == mode]
    if len(good) < MIN_SLICES:
        return None
    rdr = sitk.ImageSeriesReader(); rdr.SetFileNames(good)
    return rdr.Execute()

def hu_window_to_unit(arr_hu: np.ndarray, wl: int, wh: int):
    a = np.clip(arr_hu, wl, wh)
    return (a - wl) / (wh - wl + 1e-8)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_png_grayscale(path: Path, img01: np.ndarray):
    img01 = np.clip(img01, 0.0, 1.0)
    arr8 = (img01 * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), arr8)

def save_hu_npy(path: Path, ct_hu_slice: np.ndarray):
    """Zapisuje surowy slajs CT w HU jako float32 .npy."""
    np.save(str(path), ct_hu_slice.astype(np.float32), allow_pickle=False)

def to_uint8(arr):
    a = np.clip(arr.astype(np.int32), 0, 255).astype(np.uint8)
    return a

def majority_label(mask_int: np.ndarray, within_bool: np.ndarray):
    vals, counts = np.unique(mask_int[within_bool], return_counts=True)
    if len(vals) == 0:
        return None
    return int(vals[np.argmax(counts)])

def has_both_lungs(lung_slice: np.ndarray) -> bool:
    """True, jeśli na slajsie są obecne etykiety 1 i 2 (lewe i prawe płuco)."""
    vals = set(np.unique(lung_slice).tolist())
    return (1 in vals) and (2 in vals)

def distance_mm_to_boundary(bin_mask: np.ndarray, roi_bool: np.ndarray, spacing_xy):
    """
    Minimalna odległość [mm] od ROI do granicy maski (np. opłucnej).
    Uwzględnia anizotropię XY przez tymczasową izotropizację siatki.
    """
    if not np.any(roi_bool):
        return None
    sx, sy = float(spacing_xy[0]), float(spacing_xy[1])
    s_iso = min(sx, sy) if min(sx, sy) > 0 else 1.0
    fx = sx / s_iso
    fy = sy / s_iso
    iso_bin = cv2.resize(
        (bin_mask.astype(np.uint8) > 0).astype(np.uint8),
        dsize=None, fx=1.0/fx, fy=1.0/fy,
        interpolation=cv2.INTER_NEAREST
    )
    iso_roi = cv2.resize(
        (roi_bool.astype(np.uint8) > 0).astype(np.uint8),
        dsize=(iso_bin.shape[1], iso_bin.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    dist = cv2.distanceTransform(iso_bin, distanceType=cv2.DIST_L2, maskSize=3)
    if not np.any(iso_roi):
        return None
    min_iso_px = float(dist[iso_roi].min())
    return min_iso_px * s_iso

# ───────── LIDC: parsowanie anotacji + cech (średnia z ocen) ────────────────
def parse_nodules_polygons(xml_path: Path, origin_zyx, spacing_zyx, num_slices: int):
    """
    Zwraca:
      per_slice: dict[z] -> list[ { 'xs':[], 'ys':[], 'nodule_id':str, 'diam_px':float } ]
      char_mean_by_id: dict[nodule_id] -> dict cech (średnia z ocen radiologów)
    """
    root = ET.parse(xml_path).getroot()

    # 1) agregacja cech: średnia z ocen (na poziomie nodule_id)
    char_agg = defaultdict(lambda: defaultdict(list))
    for session in root.findall('.//lidc:readingSession', NS):
        for nod in session.findall('.//lidc:unblindedReadNodule', NS):
            nid_el = nod.find('.//lidc:noduleID', NS)
            if nid_el is None:
                continue
            nid = nid_el.text.strip()
            chars = nod.find('.//lidc:characteristics', NS)
            if chars is not None:
                for ch in chars:
                    tag = ch.tag.split('}')[-1]
                    if tag in LIDC_CHAR_KEYS and ch.text and ch.text.strip():
                        try:
                            char_agg[nid][tag].append(float(ch.text.strip()))
                        except:
                            pass
    # średnie
    char_mean_by_id = {}
    for nid, per_key in char_agg.items():
        entry = {}
        for key in LIDC_CHAR_KEYS:
            lst = per_key.get(key, [])
            entry[key] = (float(np.mean(lst)) if lst else None)
        char_mean_by_id[nid] = entry

    # 2) kontury per slice
    per_slice = defaultdict(list)
    for session in root.findall('.//lidc:readingSession', NS):
        for nod in session.findall('.//lidc:unblindedReadNodule', NS):
            nid_el = nod.find('.//lidc:noduleID', NS)
            if nid_el is None: 
                continue
            nid = nid_el.text.strip()
            for roi in nod.findall('.//lidc:roi', NS):
                z_el = roi.find('.//lidc:imageZposition', NS)
                if z_el is None: 
                    continue
                z_pos = float(z_el.text.strip())
                # liniowe mapowanie na indeks Z (proste, ale działa dla typowych serii LIDC)
                z_idx = int(round((z_pos - origin_zyx[0]) / spacing_zyx[0]))
                if not (0 <= z_idx < num_slices): 
                    continue
                xs, ys = [], []
                for edge in roi.findall('.//lidc:edgeMap', NS):
                    x_el = edge.find('.//lidc:xCoord', NS)
                    y_el = edge.find('.//lidc:yCoord', NS)
                    if x_el is not None and y_el is not None:
                        xs.append(int(float(x_el.text)))
                        ys.append(int(float(y_el.text)))
                if len(xs) >= 3:
                    xs_arr = np.array(xs); ys_arr = np.array(ys)
                    diam_px = max(xs_arr.max() - xs_arr.min(), ys_arr.max() - ys_arr.min())
                    if diam_px >= 1:
                        per_slice[z_idx].append({
                            "xs": xs, "ys": ys, "nodule_id": nid, "diam_px": float(diam_px)
                        })
    # de-duplikacja per slice po nodule_id
    for z, items in per_slice.items():
        seen = set(); uniq = []
        for it in items:
            nid = it["nodule_id"]
            if nid in seen: 
                continue
            seen.add(nid); uniq.append(it)
        per_slice[z] = uniq

    return per_slice, char_mean_by_id

# ───────── LOBES U-NET (stub – wstaw swój forward) ──────────────────────────
def infer_lobes_unet(ct_img: sitk.Image) -> np.ndarray:
    """
    Zwraca [Z, Y, X] z etykietami płatów:
      0=bg, 1=left_lower(LLL), 2=left_upper(LUL), 3=right_lower(RLL), 4=right_middle(RML), 5=right_upper(RUL)
    TODO: podmień wnętrze na realną inferencję Twojego U-neta (preprocess HU → model → argmax).
    """
    arr_zyx = sitk.GetArrayFromImage(ct_img)
    Z, H, W = arr_zyx.shape
    # Placeholder (same zeros). Wstaw tu: normalizacja, tiling 3D, model(...), argmax po kanałach.
    return np.zeros((Z, H, W), dtype=np.uint8)

# ───────── GŁÓWNA LOGIKA PACJENTA ────────────────────────────────────────────
def process_xml(xml_path: Path, out_root: Path):
    """
    Zwraca:
      (nodule_samples, negative_samples, skipped_low_lung)
    """
    pid, suid = get_pid_uid(xml_path)
    dcmpath = find_dicom(LIDC_ROOT, pid, suid)
    if not dcmpath:
        print(f"[MISS] {pid}/{suid} – brak DICOM")
        return 0, 0, 0

    ct_img = read_ct_series(dcmpath)
    if ct_img is None:
        print(f"[SKIP] {pid}/{suid} – nie-CT lub zbyt mało slajsów")
        return 0, 0, 0

    arr_zyx = sitk.GetArrayFromImage(ct_img)                 # [Z, Y, X] w HU
    origin_zyx = np.array(ct_img.GetOrigin())[::-1]          # Z, Y, X
    spacing_zyx = np.array(ct_img.GetSpacing())[::-1]        # Z, Y, X
    Z, H, W = arr_zyx.shape

    # Maski płuc (1=left, 2=right) – odrzucimy slajsy bez obu płuc
    lung_lab = inferer_lung.apply(ct_img).astype(np.uint8)   # [Z, Y, X], 0/1/2 (w niektórych wersjach 0/1)
    # Maski płatów (stub – same zera dopóki nie podepniesz modelu)
    lobes_lab = infer_lobes_unet(ct_img).astype(np.uint8)    # [Z, Y, X], 0..5

    # Parsowanie anotacji (poligony 2D per guzek) + średnie cech LIDC
    per_slice, char_mean_by_id = parse_nodules_polygons(xml_path, origin_zyx, spacing_zyx, Z)

    # Wyjściowy katalog serii
    series_root = out_root / "slices" / pid / suid
    ensure_dir(series_root)

    nodule_samples = 0
    negative_samples = 0
    skipped_low_lung = 0

    wl, wh = WINDOW

    for z in range(Z):
        ct_hu = arr_zyx[z]
        ct01 = hu_window_to_unit(ct_hu, wl, wh)             # [0,1]
        lung_slice = lung_lab[z]                             # 0/1/2

        # ── FILTR: minimalne pokrycie płuc ───────────────────────────────
        lung_ratio = float((lung_slice > 0).sum()) / float(lung_slice.size)
        if lung_ratio < MIN_LUNG_RATIO:
            skipped_low_lung += 1
            continue

        # ── FILTR: muszą być obecne oba płuca (1 i 2) ────────────────────
        if not has_both_lungs(lung_slice):
            skipped_low_lung += 1
            continue
        # ─────────────────────────────────────────────────────────────────

        # Negatyw, jeśli brak guzków
        nods_here = per_slice.get(z, [])
        if len(nods_here) == 0:
            subdir = series_root / f"z_{z:0{SLICE_NAME_PAD}d}_neg"
            ensure_dir(subdir)

            ct01_masked = masked_ct01(ct01, lung_slice, None)
            save_png_grayscale(subdir / "ct.png", ct01_masked)
            save_hu_npy(subdir / "ct_hu.npy", ct_hu)                         # HU (bezstratnie)
            cv2.imwrite(str(subdir / "lung_mask.png"), to_uint8(lung_slice))
            cv2.imwrite(str(subdir / "lobes_mask.png"), to_uint8(lobes_lab[z]))

            labels = {
                "patient_id": pid,
                "series_uid": suid,
                "slice_index": int(z),
                "z_position_mm": float(origin_zyx[0] + z*spacing_zyx[0]),    # uproszczone Z (bez direction)
                "image_size": [int(W), int(H)],
                "spacing_xy_mm": [float(spacing_zyx[2]), float(spacing_zyx[1])],  # X,Y
                "has_nodule": False,
                "lung_ratio": round(lung_ratio, 4)
            }
            meta = {
                "orientation": "unknown_RAS_like",
                "window": list(WINDOW),
                "lung_mask_semantics": LUNG_SEMANTICS,
                "lobes_mask_semantics": LOBES_SEMANTICS,
                "raw_hu_file": "ct_hu.npy",
                "raw_hu_dtype": "float32",
                "raw_hu_units": "HU"
            }
            (subdir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
            (subdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            negative_samples += 1
            continue

        # Pozytywy – osobno per guzek na tym slajsie
        for idx, item in enumerate(nods_here, start=1):
            xs, ys = item["xs"], item["ys"]
            nid = item["nodule_id"]

            # maska guzka 2D
            nod_mask = np.zeros((H, W), dtype=np.uint8)
            rr, cc = polygon(ys, xs, (H, W))
            if len(rr) < MIN_NODULE_PIXELS:
                continue
            nod_mask[rr, cc] = 1

            subdir = series_root / f"z_{z:0{SLICE_NAME_PAD}d}_nod-{idx:02d}"
            ensure_dir(subdir)

            # zapisz CT i maski
            ct01_masked = masked_ct01(ct01, lung_slice, nod_mask)
            save_png_grayscale(subdir / "ct.png", ct01_masked)
            save_hu_npy(subdir / "ct_hu.npy", ct_hu)                         # HU
            cv2.imwrite(str(subdir / "lung_mask.png"), to_uint8(lung_slice))
            cv2.imwrite(str(subdir / "nodule_mask.png"), to_uint8(nod_mask))
            cv2.imwrite(str(subdir / "lobes_mask.png"), to_uint8(lobes_lab[z]))

            # cechy geometryczne
            spacing_xy = (spacing_zyx[2], spacing_zyx[1])  # X, Y [mm/px]
            dx_mm = (max(xs) - min(xs)) * float(spacing_xy[0])
            dy_mm = (max(ys) - min(ys)) * float(spacing_xy[1])
            diameter_mm = float(max(dx_mm, dy_mm))

            # lewo/prawo (dominująca etykieta wewnątrz guzka)
            side_lab = majority_label(lung_slice, nod_mask==1)
            side_txt = {1:"left", 2:"right"}.get(side_lab, None)

            # dystans do opłucnej (anizotropia XY)
            lung_bin = (lung_slice > 0).astype(np.uint8)
            d_pleura = distance_mm_to_boundary(lung_bin, nod_mask==1, spacing_xy)

            # płat (dominująca etykieta wewnątrz guzka)
            lobe_id = majority_label(lobes_lab[z], nod_mask==1)  # 0..5 albo None

            # cechy LIDC – ŚREDNIE (numeryczne)
            chars_mean = char_mean_by_id.get(nid, {k: None for k in LIDC_CHAR_KEYS})

            # labels.json
            labels = {
                "patient_id": pid,
                "series_uid": suid,
                "slice_index": int(z),
                "z_position_mm": float(origin_zyx[0] + z*spacing_zyx[0]),
                "image_size": [int(W), int(H)],
                "spacing_xy_mm": [float(spacing_xy[0]), float(spacing_xy[1])],  # X,Y
                "has_nodule": True,

                "nodule_id": str(nid),
                "nodule_idx_in_slice": int(idx),
                "nodule_count_in_slice": int(len(nods_here)),
                "nodule_bbox_xyxy_px": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],

                "diameter_mm": float(diameter_mm),

                # Położenie anatomiczne:
                "side": side_txt,                                  # {"left","right"} lub None
                "lobe_id": int(lobe_id) if lobe_id is not None else None,  # 0..5

                # Metryki geometryczne:
                "distance_to_pleura_mm": float(d_pleura) if d_pleura is not None else None,
                "lung_ratio": round(lung_ratio, 4),

                # Wszystkie cechy LIDC – ŚREDNIE z ocen radiologów (numery, bez mapowań tekstowych)
                "lidc_characteristics_mean": chars_mean
            }
            meta = {
                "orientation": "unknown_RAS_like",
                "window": list(WINDOW),
                "lung_mask_semantics": LUNG_SEMANTICS,
                "lobes_mask_semantics": LOBES_SEMANTICS,
                "raw_hu_file": "ct_hu.npy",
                "raw_hu_dtype": "float32",
                "raw_hu_units": "HU"
            }

            # prompt.txt (opis kliniczny – możesz używać/nie używać ocen jako tekst)
            def make_prompt_pl(diam_mm, side, d_pleura):
                side_pl = {"left":"lewe","right":"prawe"}.get(side, side or "nieznane")
                parts = [f"Guzek {int(round(diam_mm))} mm", f"płuco: {side_pl}"]
                if d_pleura is not None:
                    parts.append(f"~{int(round(d_pleura))} mm od opłucnej")
                return ", ".join(parts) + "."
            prompt = make_prompt_pl(
                diam_mm=labels["diameter_mm"],
                side=labels["side"],
                d_pleura=labels["distance_to_pleura_mm"]
            )

            (subdir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
            (subdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            (subdir / "prompt.txt").write_text(prompt, encoding="utf-8")

            nodule_samples += 1

    return nodule_samples, negative_samples, skipped_low_lung

# ───────── GŁÓWNA FUNKCJA ────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "slices").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "splits").mkdir(parents=True, exist_ok=True)

    # Zbierz wszystkie XML
    xml_files = sorted(LIDC_ROOT.rglob("*.xml"))
    print(f"Znaleziono {len(xml_files)} plików XML")

    # Splity po pacjencie
    pids = []
    for x in xml_files:
        try:
            pid, _ = get_pid_uid(x)
            pids.append(pid)
        except:
            pids.append(None)

    pid_to_xmls = defaultdict(list)
    for x, pid in zip(xml_files, pids):
        if pid:
            pid_to_xmls[pid].append(x)
    unique_pids = sorted(pid_to_xmls.keys())

    train_pids, temp_pids = train_test_split(unique_pids, test_size=(VAL_RATIO+TEST_RATIO), random_state=42)
    val_pids, test_pids = train_test_split(temp_pids, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=42)

    def pid_set(name): 
        return set({"train":train_pids, "val":val_pids, "test":test_pids}[name])

    # Zapis list splitów
    for name, plist in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
        (OUT_DIR / "splits" / f"{name}.txt").write_text("\n".join(plist), encoding="utf-8")

    # patients.csv
    with open(OUT_DIR / "patients.csv", "w", encoding="utf-8") as fcsv:
        fcsv.write("patient_id,split\n")
        for pid in unique_pids:
            split = "train" if pid in pid_set("train") else ("val" if pid in pid_set("val") else "test")
            fcsv.write(f"{pid},{split}\n")

    # Przetwarzanie
    stats = {
        "train":{"pos":0,"neg":0,"patients":0,"skipped_low_lung":0},
        "val":{"pos":0,"neg":0,"patients":0,"skipped_low_lung":0},
        "test":{"pos":0,"neg":0,"patients":0,"skipped_low_lung":0}
    }
    total_xml = len(xml_files)

    for i, xml_path in enumerate(xml_files, 1):
        try:
            pid, suid = get_pid_uid(xml_path)
        except:
            print(f"[WARN] pomijam {xml_path.name} (brak PID/SUID)")
            continue
        split = "train" if pid in pid_set("train") else ("val" if pid in pid_set("val") else "test")
        print(f"[{i}/{total_xml}] {pid}/{suid} → {split}")

        pos, neg, skipped = process_xml(xml_path, OUT_DIR)
        if (pos+neg+skipped) > 0:
            stats[split]["pos"] += pos
            stats[split]["neg"] += neg
            stats[split]["skipped_low_lung"] += skipped
            stats[split]["patients"] += 1

    # Podsumowanie
    print("\n" + "="*60)
    print("STATYSTYKI KOŃCOWE")
    print("="*60)
    for split in ["train","val","test"]:
        s = stats[split]
        print(f"{split.upper()}: patients={s['patients']}, pos={s['pos']}, neg={s['neg']}, skipped_low_lung={s['skipped_low_lung']}")
    print(f"\nWynik zapisano do: {OUT_DIR}")
    print(f"Filtr pokrycia płuc: MIN_LUNG_RATIO = {MIN_LUNG_RATIO:.2f}")

if __name__ == "__main__":
    main()
