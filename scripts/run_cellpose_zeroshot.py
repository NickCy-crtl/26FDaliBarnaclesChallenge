"""Zero-shot Cellpose-SAM (cpsam, cpsam_v2) on the inner quadrat, scored against the annotations.

"Zero-shot" means the pretrained weights are used as shipped: no fine-tuning, and
the Cellpose defaults for every inference setting (no diameter, flow_threshold
0.4, cellprob_threshold 0.0, min_size 15). That is the honest baseline -- any
tuning of these knobs against img1/img2 would already be fitting to the test data.

Input is the output of crop_quadrat.py: the photo cropped to the quadrat bounding
box plus the exact cell mask. Predictions whose centroid falls outside the cell
mask (the slivers of wire/rock in the corners of the box) are discarded before
counting, since the annotator never labelled anything there.

Scoring
-------
The annotations mark the barnacle *opening* (operculum), while a generic model is
likely to segment the whole shell. An opening is then a small blob inside a much
larger predicted mask, and their IoU is low even when the detection is right, so
the usual IoU>=0.5 matching (Cellpose's `average_precision`) would call a correct
detection both a false positive and a false negative. The primary metric here is
therefore a *coverage* match:

    an annotation is matched to a prediction if that prediction covers at least
    `--min-coverage` (default 50%) of the annotation's pixels,

made one-to-one with the Hungarian algorithm, so one large mask swallowing three
barnacles counts as 1 TP + 2 FN rather than 3 TP. Precision / recall / F1 and the
plain count error follow from that. AP@0.5 IoU is still reported for reference
and for comparison with the Cellpose literature.

The alternative of matching by "annotation centroid lies inside the prediction" is
simpler but ambiguous for crescent-shaped or partly-occluded openings whose
centroid lands outside their own pixels; coverage handles those.

Optional preprocessing (--preprocess) and Cellpose normalization overrides
(--norm-percentile, --tile-norm-blocksize) are implemented in preprocessing.py.

Outputs, in processed_data/cellpose_zeroshot/<model>/:
    <tag>_pred.tif        predicted label image (crop coordinates)
    <tag>_overlay.png     original RGB crop with annotations (red) and predictions (yellow)
and processed_data/cellpose_zeroshot/metrics.json with one row per model x image x
settings. <tag> is the image stem plus a suffix for each non-default setting
(e.g. img2_d50_p1-99.5_tn150, img2_d50_claheL), so new settings never overwrite
earlier results. A preprocessed model input is also saved once per image as
processed_data/cellpose_zeroshot/inputs/<stem>[_d<diameter>]_<method>_input.png.

Usage:
    python scripts/run_cellpose_zeroshot.py                         # both models, all images
    python scripts/run_cellpose_zeroshot.py --models cpsam_v2 --images img1
    python scripts/run_cellpose_zeroshot.py --images img1 img2 --diameter img1=30 img2=50 \\
        --norm-percentile 1 99.5 --tile-norm-blocksize 150 --group RGB_tilenorm
    python scripts/run_cellpose_zeroshot.py --images img1 img2 --diameter img1=30 img2=50 \\
        --preprocess clahe_l --group CLAHE_L_gray
    python scripts/run_cellpose_zeroshot.py --images img1 img2 --diameter img1=30 img2=50 \\
        --norm-percentile 1 99.5 --tile-norm-blocksize 150 --preprocess composite \\
        --composite-channels dog clahe_l gray --group Composite_DoG_CLAHE_gray_tilenorm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from skimage.segmentation import find_boundaries

from preprocessing import (
    COMPOSITE_CHANNELS, PREPROCESS_METHODS, build_normalize, preprocess, preprocess_record,
    settings_suffix,
)

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parent.parent
QUADRAT_DIR = REPO_ROOT / "processed_data" / "quadrat"
RAW = REPO_ROOT / "raw_data"
DEFAULT_OUT = REPO_ROOT / "processed_data" / "cellpose_zeroshot"
DEFAULT_IMAGES = ["img1", "img2", "unseen_img1", "unseen_img2"]


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
def load_inputs(stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(rgb crop, quadrat mask, annotation labels or None) for one image.

    Images without a grid (unseen_img2, a 500x500 close-up) have no crop; they
    are used whole, with an all-True region mask.
    """
    crop = QUADRAT_DIR / f"{stem}_crop.png"
    if crop.exists():
        rgb = np.array(Image.open(crop).convert("RGB"))
        region = np.array(Image.open(QUADRAT_DIR / f"{stem}_quadrat_mask.png")) > 0
    else:
        rgb = np.array(Image.open(RAW / f"{stem}.png").convert("RGB"))
        region = np.ones(rgb.shape[:2], bool)

    gt_path = QUADRAT_DIR / f"{stem}_masks_crop.tif"
    gt = tifffile.imread(gt_path).astype(np.int32) if gt_path.exists() else None
    return rgb, region, gt


def drop_outside(labels: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Remove labels whose centroid is outside the quadrat cell; relabel 1..N.

    Centroid rather than "any pixel outside": a barnacle sitting against the wire
    legitimately spills a few pixels past the cell edge (21 annotations in img1
    do), and should be kept just as the annotator kept it.
    """
    n = int(labels.max())
    if n == 0:
        return labels
    rr, cc = np.nonzero(labels)
    ids = labels[rr, cc]
    count = np.bincount(ids, minlength=n + 1)
    cy = np.bincount(ids, weights=rr, minlength=n + 1) / np.maximum(count, 1)
    cx = np.bincount(ids, weights=cc, minlength=n + 1) / np.maximum(count, 1)

    keep = np.zeros(n + 1, bool)
    present = count > 0
    keep[present] = region[cy[present].round().astype(int), cx[present].round().astype(int)]
    keep[0] = False

    lut = np.zeros(n + 1, np.int32)
    lut[keep] = np.arange(1, keep.sum() + 1)
    return lut[labels]


def parse_diameters(values: list[str] | None, stems: list[str]) -> dict[str, float | None]:
    """Turn the --diameter arguments into {image stem: diameter or None}.

    Per-image values exist because the photos are taken at different distances:
    the same barnacle species spans a different number of pixels in img1 and
    img2, so one global diameter would rescale one of them the wrong way. A bare
    number is kept as a shorthand for "the same for all", matching the old flag.
    """
    if not values:
        return {s: None for s in stems}
    if len(values) == 1 and "=" not in values[0]:
        return {s: float(values[0]) for s in stems}

    given = {}
    for item in values:
        stem, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--diameter: mix of plain and per-image values: {values}")
        given[stem] = float(value)
    unknown = set(given) - set(stems)
    if unknown:
        raise SystemExit(f"--diameter given for images not being run: {sorted(unknown)}")
    return {s: given.get(s) for s in stems}


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def coverage_match(gt: np.ndarray, pred: np.ndarray, min_coverage: float) -> dict:
    """One-to-one matching of annotations to predictions by annotation coverage.

    coverage[i, j] = |gt_i & pred_j| / |gt_i|. The overlap table is built with a
    sparse matrix in one pass over the pixels, which avoids an n_gt x n_pred loop
    (~1700 x ~2000 for img1). Hungarian assignment then maximises total coverage,
    and pairs below `min_coverage` are discarded.
    """
    n_gt, n_pred = int(gt.max()), int(pred.max())
    both = (gt > 0) & (pred > 0)
    overlap = coo_matrix(
        (np.ones(both.sum()), (gt[both], pred[both])), shape=(n_gt + 1, n_pred + 1)
    ).tocsr()
    gt_area = np.bincount(gt.ravel(), minlength=n_gt + 1).astype(float)

    tp = 0
    if n_gt and n_pred and overlap.nnz:
        cov = overlap.toarray()[1:, 1:] / np.maximum(gt_area[1:, None], 1)
        rows, cols = linear_sum_assignment(cov, maximize=True)
        tp = int((cov[rows, cols] >= min_coverage).sum())

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gt - tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def write_overlay(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray | None, path: Path) -> None:
    """Full-resolution crop with predicted outlines in yellow and annotations in red.

    Red is drawn last so that where both agree the annotation stays visible; a
    yellow ring with a red blob inside is the expected "shell vs opening" picture
    of a correct detection.
    """
    out = rgb.copy()
    out[find_boundaries(pred, mode="inner")] = (255, 220, 0)
    if gt is not None:
        out[find_boundaries(gt, mode="inner")] = (255, 0, 0)
    Image.fromarray(out).save(path)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=["cpsam", "cpsam_v2"])
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES, help="image stems")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--diameter",
        nargs="+",
        default=None,
        help="leave unset for true zero-shot; setting it rescales the image so objects are ~30 px. "
        "Either one value for all images ('--diameter 30') or per image ('--diameter img1=30 img2=50'); "
        "images not listed run without a diameter",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="free-text run category stored on every metrics.json row this run writes, "
        "e.g. 'Zeroshot with Diameter'",
    )
    parser.add_argument(
        "--norm-percentile", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None,
        help="override Cellpose's normalization percentiles (default 1 99)",
    )
    parser.add_argument(
        "--tile-norm-blocksize", type=int, default=0,
        help="compute normalization percentiles per tile of this size in crop pixels (0 = whole image)",
    )
    parser.add_argument(
        "--preprocess", choices=PREPROCESS_METHODS, default="none",
        help="clahe_l: CLAHE on CIELAB L, copied into 3 channels; composite: one cue per "
        "model channel, chosen with --composite-channels (see preprocessing.py)",
    )
    parser.add_argument(
        "--composite-channels", nargs=3, choices=COMPOSITE_CHANNELS, default=["dog", "clahe_l", "gray"],
        metavar="CH", help=f"composite: the 3 channels in order, from {COMPOSITE_CHANNELS} "
        "('dog' needs --diameter)",
    )
    parser.add_argument("--dog-small-sigma", type=float, default=1.5, help="composite: small DoG sigma in px")
    parser.add_argument(
        "--dog-large-frac", type=float, default=0.25,
        help="composite: large DoG sigma as a fraction of the image's diameter",
    )
    parser.add_argument("--clahe-kernel", type=int, default=150, help="CLAHE tile size in crop pixels")
    parser.add_argument("--clahe-clip", type=float, default=0.01, help="skimage CLAHE clip limit (0-1)")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true", help="force CPU instead of CUDA/MPS")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="run the network in float32 (default is Cellpose's bfloat16; use this if MPS misbehaves)",
    )
    args = parser.parse_args()
    diameters = parse_diameters(args.diameter, args.images)
    normalize = build_normalize(args.norm_percentile, args.tile_norm_blocksize)
    # Stored in metrics.json as None when at the default, so rows written before
    # these options existed (which lack the keys) compare equal to default runs.
    norm_record = normalize if isinstance(normalize, dict) else None
    prep_record = preprocess_record(
        args.preprocess, args.clahe_kernel, args.clahe_clip, args.composite_channels,
        args.dog_small_sigma, args.dog_large_frac,
    )

    # Imported here so that `--help` and the helpers above work without torch.
    import torch
    from cellpose import metrics, models

    device = None
    if args.cpu:
        device = torch.device("cpu")

    results_path = args.out_dir / "metrics.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    # Re-running the exact same group x model x image x settings combination replaces
    # its old row rather than duplicating it; anything else is kept side by side.
    # The dict-valued settings are compared as sorted JSON so they can be set members.
    def row_key(group, model, image, diameter, norm, prep):
        return (group, model, image, diameter,
                json.dumps(norm, sort_keys=True), json.dumps(prep, sort_keys=True))

    rerun = {
        row_key(args.group, m, s, diameters[s], norm_record, prep_record)
        for m in args.models for s in args.images
    }
    results = [
        r for r in results
        if row_key(r.get("group"), r["model"], r["image"], r.get("diameter"),
                   r.get("normalize"), r.get("preprocess")) not in rerun
    ]

    # Preprocessing does not depend on the model, so it runs once per image
    # rather than once per model x image.
    inputs = {}
    for stem in args.images:
        rgb, region, gt = load_inputs(stem)
        net_input = preprocess(
            rgb, args.preprocess, args.clahe_kernel, args.clahe_clip,
            diameters[stem], args.composite_channels, args.dog_small_sigma, args.dog_large_frac,
        )
        if args.preprocess != "none":
            input_dir = args.out_dir / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            # The diameter is in the name because the DoG channel depends on it.
            tag = stem + settings_suffix(
                diameters[stem], True, args.preprocess, args.composite_channels
            )
            Image.fromarray(net_input).save(input_dir / f"{tag}_input.png")
        inputs[stem] = (rgb, net_input, region, gt)

    for model_name in args.models:
        print(f"\n##### {model_name} #####")
        model = models.CellposeModel(
            gpu=not args.cpu, device=device, pretrained_model=model_name,
            use_bfloat16=not args.fp32,
        )
        out_dir = args.out_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for stem in args.images:
            rgb, net_input, region, gt = inputs[stem]
            tic = time.time()
            # channel_axis=-1: our arrays are HxWx3. Cellpose 4 takes RGB in any
            # channel order, so no `channels=` mapping is needed (it is deprecated).
            diameter = diameters[stem]
            pred, _, _ = model.eval(
                net_input, channel_axis=-1, diameter=diameter, normalize=normalize
            )
            seconds = time.time() - tic

            pred = drop_outside(pred.astype(np.int32), region)
            # Every non-default setting goes into the file name, so these runs sit
            # next to the earlier outputs instead of overwriting them.
            tag = stem + settings_suffix(diameter, normalize, args.preprocess, args.composite_channels)
            tifffile.imwrite(out_dir / f"{tag}_pred.tif", pred.astype(np.uint16), compression="zlib")
            # Drawn on the original RGB rather than the model input, so overlays from
            # every setting can be compared side by side; the input is under inputs/.
            write_overlay(rgb, pred, gt, out_dir / f"{tag}_overlay.png")

            n_pred = int(pred.max())
            row = {
                # First key so it is the first thing seen when scanning the file;
                # null when no --group is given, ready to be filled in by hand.
                "group": args.group,
                "model": model_name,
                "image": stem,
                "diameter": diameter,
                "normalize": norm_record,
                "preprocess": prep_record,
                "n_pred": n_pred,
                "seconds": round(seconds, 1),
            }
            line = f"  {tag:24s} pred {n_pred:5d}"
            if gt is not None:
                n_gt = int(gt.max())
                ap, *_ = metrics.average_precision(gt, pred, threshold=[0.5])
                row |= {
                    "n_gt": n_gt,
                    "count_error": n_pred - n_gt,
                    "count_error_pct": round(100 * (n_pred - n_gt) / n_gt, 1),
                    **coverage_match(gt, pred, args.min_coverage),
                    "ap50_iou": round(float(ap[0]), 4),
                }
                line += (
                    f"  gt {n_gt:5d}  count err {row['count_error_pct']:+6.1f}%"
                    f"  P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f}"
                    f"  AP50(IoU) {row['ap50_iou']:.3f}"
                )
            print(line + f"  ({seconds:.0f}s)")
            results.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {results_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
