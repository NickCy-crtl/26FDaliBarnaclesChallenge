"""Convert hand-drawn barnacle outline masks into Cellpose-style integer label images.

The raw annotations (raw_data/mask*.png) are *outlines*: a human drew a closed
ring around every barnacle opening with a paint brush of inconsistent width.
Cellpose instead expects a label image, where the pixels belonging to barnacle
k carry the integer value k and background is 0.

The conversion has three stages:

  1. interiors   -- every region of background fully enclosed by a stroke is one
                    barnacle. These are the *inner* circles.
  2. width       -- each outline is an annulus; from its inner contour and the
                    stroke area around it we recover the local brush width w.
  3. expansion   -- each interior grows outwards by w/2, so its boundary lands on
                    the stroke centreline, which is where the annotator intended
                    the barnacle edge to be.

Stage 3 is what stops every barnacle from being systematically undersized by the
brush, while still keeping touching barnacles apart.

Usage:
    python scripts/make_annotations.py                     # both masks, default settings
    python scripts/make_annotations.py --masks raw_data/mask1.png
    python scripts/make_annotations.py --width-method equivalent-circle --no-qc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops_table
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries

# The raw frames are 2448x3264; Pillow's decompression-bomb guard trips below that.
Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MASKS = [REPO_ROOT / "raw_data" / "mask1.png", REPO_ROOT / "raw_data" / "mask2.png"]
DEFAULT_OUT = REPO_ROOT / "processed_data"


# --------------------------------------------------------------------------- #
# Stage 0: read the painted mask                                              #
# --------------------------------------------------------------------------- #
def load_stroke_mask(path: Path) -> np.ndarray:
    """Return a boolean array that is True wherever the annotator painted---this is a simplified version that only works
    for a mask where the background pixels are alpha = 0.

    mask1.png contains two different blues (two annotation passes covering
    different horizontal bands of the image) and mask2.png one. Rather than
    hard-coding those RGB values we key off the alpha channel, which is 0 on
    untouched pixels and 255 on painted ones -- that stays correct if a future
    pass is painted in a third colour. If a mask ever arrives without alpha we
    fall back to "anything that is not the dominant background colour".
    """
    rgba = np.array(Image.open(path))

    #if rgba.ndim == 3 and rgba.shape[2] == 4 and (rgba[..., 3] == 0).any():
    return rgba[..., 3] > 0

    #these lines after this will only be reached for a colored
    flat = rgba.reshape(-1, rgba.shape[-1]) if rgba.ndim == 3 else rgba.reshape(-1, 1)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    background = colours[counts.argmax()]
    return ~(flat == background).all(axis=1).reshape(rgba.shape[:2])


# --------------------------------------------------------------------------- #
# Stage 1: interiors = background regions fully enclosed by a stroke          #
# --------------------------------------------------------------------------- #
def find_interiors(stroke: np.ndarray, min_area: int) -> tuple[np.ndarray, dict]:
    """Label every enclosed background region; drop ones too small to be a barnacle.

    Background is labelled with 4-connectivity on purpose. With 8-connectivity a
    stroke that closes only through a diagonal step would leak, and those exist:
    in mask1 a 472-pixel barnacle is held shut by a single diagonal. 4-connectivity
    never leaks, at the cost of spurious 1-pixel "holes" wherever the brush
    crosses itself diagonally -- which `min_area` then removes.

    `min_area` has to stay small. Some barnacles genuinely are tiny, and ones
    sitting on a ridge are seen side-on so their opening is small from above;
    those are real and must survive. The measured size distribution is sharply
    bimodal (a spike of 1-2 px self-touch artefacts, then a gap, then real
    barnacles from ~10 px up), so the default sits in that gap.
    """
    background = label(~stroke, connectivity=1)

    # Any background component reaching the image edge is outside, not a barnacle.
    edges = np.concatenate(
        [background[0], background[-1], background[:, 0], background[:, -1]]
    )
    outside = np.unique(edges)
    enclosed = (background > 0) & ~np.isin(background, outside)

    holes = label(enclosed, connectivity=1)
    areas = np.bincount(holes.ravel())
    areas[0] = 0

    keep = areas >= min_area
    # Remap the surviving holes onto a gapless 1..N range, as Cellpose expects.
    lut = np.zeros(areas.size, dtype=np.int32)
    lut[np.flatnonzero(keep)] = np.arange(1, keep.sum() + 1, dtype=np.int32)
    labels = lut[holes]

    stats = {
        "n_found": int((areas > 0).sum()),
        "n_kept": int(keep.sum()),
        "n_dropped": int((areas > 0).sum() - keep.sum()),
        "dropped_areas": areas[(areas > 0) & ~keep],
    }
    return labels, stats


def count_open_contours(stroke: np.ndarray, min_area: int) -> int:
    """Stroke blobs that enclose nothing -- i.e. outlines the annotator left open.

    These are barnacles we cannot recover, so the number is reported as a warning
    rather than silently patched: closing gaps automatically risks fusing
    neighbours, which would be a worse error than one missing barnacle.
    """
    comps = label(stroke, connectivity=2)
    table = regionprops_table(comps, properties=("label", "area", "euler_number"))
    big = table["area"] >= max(min_area, 20)
    # euler_number == 1 means the component has no hole in it.
    return int((big & (table["euler_number"] == 1)).sum())


# --------------------------------------------------------------------------- #
# Stage 2: per-barnacle brush width                                           #
# --------------------------------------------------------------------------- #
def estimate_widths(
    labels: np.ndarray,
    stroke: np.ndarray,
    nearest: np.ndarray,
    method: str,
) -> np.ndarray:
    """Estimate the brush width w for each barnacle from its inner/outer contour.

    The stroke pixels are first split between barnacles by nearest interior
    (`nearest`, a Voronoi partition). An isolated outline keeps its whole ring;
    where two outlines touch, the shared paint is cut down the middle, so each
    barnacle still sees a ring of its own width. That ring area is the difference
    between the outer and inner circle.

    'annulus' (default) solves the exact annulus relation using the measured
    inner perimeter P:

        ring_area = P * w + pi * w**2   ->   w = (-P + sqrt(P^2 + 4*pi*ring)) / (2*pi)

    'equivalent-circle' is the simpler reading -- treat both contours as circles
    of equal area and subtract their radii:

        w = sqrt((A_in + ring) / pi) - sqrt(A_in / pi)

    Both are the same idea; the second assumes the outline is a circle. Because
    hand-drawn outlines are wiggly their true perimeter exceeds that of the
    equal-area circle, so the circle version spreads the same ring area over too
    short a contour and overestimates w. On mask2 it reads 7.2 px against a
    directly measured brush width of ~5.5-6.3 px, which would oversize every
    barnacle; on mask1 the two agree to ~0.1 px. 'annulus' is therefore the
    default, and 'equivalent-circle' is kept for comparison.
    """
    n = int(labels.max())
    table = regionprops_table(labels, properties=("label", "area", "perimeter"))

    area_in = np.zeros(n + 1)
    perimeter = np.zeros(n + 1)
    area_in[table["label"]] = table["area"]
    perimeter[table["label"]] = table["perimeter"]

    ring = np.bincount(nearest[stroke].ravel(), minlength=n + 1).astype(float)

    if method == "annulus":
        p = perimeter
        widths = (-p + np.sqrt(p**2 + 4.0 * np.pi * ring)) / (2.0 * np.pi)
    elif method == "equivalent-circle":
        widths = np.sqrt((area_in + ring) / np.pi) - np.sqrt(area_in / np.pi)
    else:
        raise ValueError(f"unknown width method: {method}")

    widths[0] = 0.0
    return widths


def reference_widths(
    labels: np.ndarray, stroke: np.ndarray, nearest: np.ndarray
) -> np.ndarray:
    """Independent width check: twice the distance-to-edge along the stroke centreline.

    This measures the brush directly instead of inferring it from contours, so
    agreement between the two is evidence the geometry above is sound. It is not
    used to build the labels -- a skeleton's spurs and junctions bias it slightly
    low -- only reported as a diagnostic.
    """
    n = int(labels.max())
    thickness = 2.0 * distance_transform_edt(stroke)
    ridge = skeletonize(stroke)

    owner = nearest[ridge]
    total = np.bincount(owner, weights=thickness[ridge], minlength=n + 1)
    count = np.bincount(owner, minlength=n + 1)
    return np.divide(total, count, out=np.zeros(n + 1), where=count > 0)


# --------------------------------------------------------------------------- #
# Stage 3: expansion by a per-barnacle distance                               #
# --------------------------------------------------------------------------- #
def expand_labels_variable(
    labels: np.ndarray, distances: np.ndarray, edt: np.ndarray, nearest: np.ndarray
) -> np.ndarray:
    """skimage.segmentation.expand_labels, but with one distance per label.

    expand_labels only accepts a single scalar distance, and here every barnacle
    needs its own w/2. Internally skimage computes the distance transform of the
    background together with the nearest label, then keeps pixels closer than
    `distance`; the only change needed is to compare each pixel against the
    threshold of the label claiming it:

        skimage:  edt <= distance
        here:     edt <= distances[nearest]

    Because every pixel still goes to its single nearest label, regions cannot
    overlap and touching barnacles meet exactly halfway, same as the original.
    Passing a constant array reproduces expand_labels exactly (asserted in tests).

    The alternative -- calling expand_labels once per distinct rounded width and
    stitching the results -- would quantise w to whole pixels and needs a tie-break
    rule where two runs claim the same pixel, so it is both less accurate and more
    code.

    Note on sub-pixel accuracy: distances between integer pixel centres are
    sqrt(integer), so the reachable radii step through 1, 1.41, 2, 2.24, ... A
    threshold of 1.93 therefore stops at 1.41, making boundaries land up to ~0.6 px
    inside the centreline. This is inherited from expand_labels, not introduced
    here; on a 7-15 px barnacle radius it is a ~2-4% area effect and does not move
    the count, which is what these annotations are scored on.
    """
    grow = edt <= distances[nearest]
    out = np.where(grow, nearest, 0).astype(labels.dtype)
    # Interiors are at distance 0 and always keep their own label.
    out[labels > 0] = labels[labels > 0]
    return out


# --------------------------------------------------------------------------- #
# QC                                                                           #
# --------------------------------------------------------------------------- #
def write_qc_overlay(photo_path: Path, labels: np.ndarray, out_path: Path) -> None:
    """Draw the final label boundaries over the photo for eyeballing.

    Full resolution and lossless, so it can be zoomed and compared against the
    hand-checked masked_img*.png overlays.
    """
    photo = np.array(Image.open(photo_path).convert("RGB"))
    overlay = photo.copy()
    overlay[find_boundaries(labels, mode="inner")] = (255, 0, 0)
    Image.fromarray(overlay).save(out_path)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def process_mask(
    mask_path: Path,
    out_dir: Path,
    min_area: int,
    width_method: str,
    max_width_factor: float,
    qc: bool,
) -> np.ndarray:
    stem = mask_path.stem.replace("mask", "img")  # mask1.png -> img1
    print(f"\n=== {mask_path.name} -> {stem}_masks.tif ===")

    stroke = load_stroke_mask(mask_path)
    print(f"  painted pixels      : {stroke.sum():,} ({stroke.mean():.2%} of frame)")

    labels, stats = find_interiors(stroke, min_area)
    dropped = stats["dropped_areas"]
    print(
        f"  enclosed regions    : {stats['n_found']:,} found, "
        f"{stats['n_dropped']:,} dropped below {min_area} px "
        f"(largest dropped: {int(dropped.max()) if dropped.size else 0} px)"
    )
    print(f"  barnacles kept      : {stats['n_kept']:,}")

    open_contours = count_open_contours(stroke, min_area)
    if open_contours:
        print(
            f"  WARNING             : {open_contours} stroke blob(s) enclose nothing "
            f"(outlines left open -> those barnacles are missing)"
        )

    # One distance transform serves both the width estimate and the expansion.
    edt, indices = distance_transform_edt(labels == 0, return_indices=True)
    nearest = labels[tuple(indices)]

    widths = estimate_widths(labels, stroke, nearest, width_method)
    positive = widths[1:][widths[1:] > 0]
    median_w = float(np.median(positive)) if positive.size else 0.0

    # A barnacle whose Voronoi territory happens to swallow unrelated paint would
    # get an inflated w and bleed outwards, so cap the outliers and say how many.
    cap = max_width_factor * median_w
    n_capped = int((widths[1:] > cap).sum())
    widths = np.minimum(widths, cap)
    widths[0] = 0.0

    ref = reference_widths(labels, stroke, nearest)
    both = (widths[1:] > 0) & (ref[1:] > 0)
    print(
        f"  brush width w ({width_method}): median {median_w:.2f} px, "
        f"IQR {np.percentile(positive, 25):.2f}-{np.percentile(positive, 75):.2f} px"
        + (f", {n_capped} capped at {cap:.2f} px" if n_capped else "")
    )
    print(
        f"  cross-check (centreline): median {np.median(ref[1:][ref[1:] > 0]):.2f} px, "
        f"correlation {np.corrcoef(widths[1:][both], ref[1:][both])[0, 1]:.3f}"
    )

    expanded = expand_labels_variable(labels, widths / 2.0, edt, nearest)

    grown = (expanded > 0).sum() - (labels > 0).sum()
    print(
        f"  expansion by w/2    : +{grown:,} px "
        f"({(expanded > 0).sum() / max((labels > 0).sum(), 1) - 1:+.1%} area), "
        f"{(expanded[stroke] > 0).mean():.1%} of the paint absorbed"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_masks.tif"
    # uint16 is the Cellpose convention and holds ~65k labels; assert we fit.
    assert expanded.max() <= np.iinfo(np.uint16).max, "too many labels for uint16"
    # Label images are ~90% zeros, so zlib takes 16 MB down to under 1 MB and
    # tifffile/Cellpose read it back transparently.
    tifffile.imwrite(out_path, expanded.astype(np.uint16), compression="zlib")
    print(f"  wrote               : {out_path.relative_to(REPO_ROOT)}")

    if qc:
        photo = mask_path.parent / f"{stem}.png"
        if photo.exists():
            qc_path = out_dir / "qc" / f"{stem}_qc_overlay.png"
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            write_qc_overlay(photo, expanded, qc_path)
            print(f"  wrote               : {qc_path.relative_to(REPO_ROOT)}")
        else:
            print(f"  (no QC overlay: {photo.name} not found)")

    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--masks", nargs="+", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="drop enclosed regions smaller than this many pixels (default: 10)",
    )
    parser.add_argument(
        "--width-method",
        choices=("annulus", "equivalent-circle"),
        default="annulus",
    )
    parser.add_argument(
        "--max-width-factor",
        type=float,
        default=3.0,
        help="cap w at this multiple of the median w (default: 3.0)",
    )
    parser.add_argument("--no-qc", dest="qc", action="store_false")
    args = parser.parse_args()

    for mask_path in args.masks:
        process_mask(
            mask_path,
            args.out_dir,
            args.min_area,
            args.width_method,
            args.max_width_factor,
            args.qc,
        )


if __name__ == "__main__":
    main()
