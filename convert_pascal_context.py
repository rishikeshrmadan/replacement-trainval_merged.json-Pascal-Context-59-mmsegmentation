"""
Pascal Context (59 classes) converter for MMSegmentation 1.x

This bypasses the dead codalab `trainval_merged.json` dependency by reading
.mat files directly, mapping the 459-class label space down to 60 classes
(59 + background), and producing the layout MMSeg's PascalContextDataset
expects:

    data/VOCdevkit/VOC2010/
        SegmentationClassContext/
            <image_id>.png         <-- 60-class index masks (255 = ignore)
        ImageSets/
            SegmentationContext/
                train.txt
                val.txt

Inputs expected (relative to data/VOCdevkit/VOC2010/):
    trainval/<image_id>.mat        <-- already extracted from trainval.tar.gz
    labels.txt                     <-- the 459-class list (in trainval.tar.gz)
    59_labels.txt                  <-- the 59-class subset
                                       https://www.cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt

The train/val split is read directly from VOC2010's own
ImageSets/Segmentation/{train,val}.txt — Pascal Context inherits VOC2010's
splits.

Run from your mmsegmentation repo root:
    python convert_pascal_context.py
"""
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.io
import tqdm
import sys


# ─── EDIT THIS IF YOUR LAYOUT IS DIFFERENT ───────────────────────────────────
VOC2010 = Path("data/VOCdevkit/VOC2010")
# ─────────────────────────────────────────────────────────────────────────────


MAT_DIR = VOC2010 / "trainval"
LABELS_459 = VOC2010 / "labels.txt"
LABELS_59 = VOC2010 / "59_labels.txt"

# Pascal Context inherits VOC2010's image splits.
VOC_TRAIN = VOC2010 / "ImageSets" / "Segmentation" / "train.txt"
VOC_VAL = VOC2010 / "ImageSets" / "Segmentation" / "val.txt"

OUT_MASK_DIR = VOC2010 / "SegmentationClassContext"
OUT_SPLIT_DIR = VOC2010 / "ImageSets" / "SegmentationContext"


def parse_labels_459(path):
    """labels.txt format: '<id>: <name>' per line."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            idx, name = line.split(":", 1)
            try:
                out[name.strip()] = int(idx.strip())
            except ValueError:
                continue
    return out


def parse_labels_59(path, dict_459):
    """59_labels.txt format: lines like '<idx>: <name>' or just '<name>'.

    Returns a list (in 0..58 order) of the 459-space ids each 59-class
    corresponds to. Class 0 in the result is the actual first listed class
    (typically 'aeroplane' or similar), not background — handled below.
    """
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                line = line.split(":", 1)[-1].strip()
            names.append(line)
    return [(name, dict_459[name]) for name in names if name in dict_459]


def main():
    # Sanity check required inputs
    for p in (MAT_DIR, LABELS_459, LABELS_59, VOC_TRAIN, VOC_VAL):
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            sys.exit(1)

    # Build the 459 -> 60 mapping
    # Class 0 in the output = "background" (everything not in the 59 list,
    # plus the original Pascal Context "unlabeled" pixels).
    # Classes 1..59 are the standard PASCAL-Context-59 categories.
    dict_459 = parse_labels_459(LABELS_459)
    pc59_pairs = parse_labels_59(LABELS_59, dict_459)

    # if len(pc59_pairs) != 59:
    #     print(f"WARNING: expected 59 class entries, got {len(pc59_pairs)}",
    #           file=sys.stderr)

    file_names = set()
    with open(LABELS_59) as f:
        for line in f:
            line = line.strip()
            if line and ":" in line:
                file_names.add(line.split(":", 1)[1].strip())
    mapped = {n for n, _ in pc59_pairs}
    missing = file_names - mapped
    assert len(pc59_pairs) == 59, f"Only {len(pc59_pairs)}/59 mapped; missing: {missing}"


    # After parse_labels_59 (and after the assertion above):
    pc59_pairs.sort(key=lambda x: x[0])  # alphabetical, matches METAINFO


    # Build a numpy lookup: lut[label_459] = label_60  (0 = background)
    lut = np.zeros(max(dict_459.values()) + 1, dtype=np.uint8)  # default 0 = bg
    for new_idx, (name, old_idx) in enumerate(pc59_pairs, start=1):
        lut[old_idx] = new_idx

    # Read VOC2010's official train/val split — Pascal Context uses the same.
    with open(VOC_TRAIN) as f:
        voc_train_ids = [l.strip() for l in f if l.strip()]
    with open(VOC_VAL) as f:
        voc_val_ids = [l.strip() for l in f if l.strip()]

    # Restrict to ids that actually have a .mat annotation
    existing_mats = {p.stem for p in MAT_DIR.glob("*.mat")}
    train_ids = [i for i in voc_train_ids if i in existing_mats]
    val_ids = [i for i in voc_val_ids if i in existing_mats]
    all_ids = sorted(existing_mats)

    print(f"VOC2010 train.txt: {len(voc_train_ids)} ids "
          f"({len(train_ids)} have .mat files)")
    print(f"VOC2010 val.txt:   {len(voc_val_ids)} ids "
          f"({len(val_ids)} have .mat files)")
    print(f"Total .mat files:  {len(all_ids)}")

    # Convert all .mat -> PNG
    OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting masks to {OUT_MASK_DIR}")
    skipped = 0
    for img_id in tqdm.tqdm(all_ids):
        mat_path = MAT_DIR / f"{img_id}.mat"
        out_path = OUT_MASK_DIR / f"{img_id}.png"
        try:
            mat = scipy.io.loadmat(str(mat_path))
            mask = mat["LabelMap"]  # shape (H, W), int values into 459 space
        except Exception as e:
            print(f"\nSkipped {img_id}: {e}", file=sys.stderr)
            skipped += 1
            continue
        # Apply LUT — pixels with label_459 not in the 59 list become 0 (bg)
        mask = mask.astype(np.int64)
        # Clip into LUT range (some unlabeled pixels can be 0, which maps to 0)
        mask = np.clip(mask, 0, len(lut) - 1)
        out = lut[mask]
        Image.fromarray(out, mode="L").save(out_path)

    print(f"Done. Skipped: {skipped}")

    # Write split files
    OUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_SPLIT_DIR / "train.txt", "w") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(OUT_SPLIT_DIR / "val.txt", "w") as f:
        f.write("\n".join(val_ids) + "\n")

    print(f"\nSplit files in {OUT_SPLIT_DIR}")
    print(f"  train.txt: {len(train_ids)} ids")
    print(f"  val.txt:   {len(val_ids)} ids")
    print("\nNext step: configure your MMSeg pascal_context dataset to point at:")
    print(f"  data_root='data/VOCdevkit/VOC2010'")
    print(f"  data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClassContext')")
    print(f"  ann_file='ImageSets/SegmentationContext/train.txt'  (or val.txt)")


if __name__ == "__main__":
    main()