"""Find the inner quadrat of the teal wire grid by colour and crop every photo to it.

Only barnacles inside the large central cell of the grid ("inner quadrat") are
annotated, so both Cellpose and the scoring must be restricted to it -- anything
detected outside would otherwise count as a false positive.

The grid wire is painted a saturated green-turquoise that occurs nowhere else in
these photos (rock is grey/brown/black, barnacles are white/beige, the ruler is
yellow-orange). A plain HSV threshold therefore separates it cleanly:

  1. wire       -- pixels with hue in the teal band and enough saturation.
  2. close      -- small morphological closing bridges the gaps where glare, a
                   barnacle or a wire-tie interrupts the wire.
  3. cells      -- the grid cells are the background components fully enclosed
                   by wire; the inner quadrat is the largest of them.
  4. crop       -- crop to that cell's bounding box and also keep the exact cell
                   shape as a boolean mask, because the wire is not perfectly
                   axis-aligned (perspective, bent wire) and a box alone would
                   include slivers of wire and outside rock in its corners.

Outputs, per image, in processed_data/quadrat/:
    <stem>_crop.png           the RGB photo cropped to the quadrat bounding box
    <stem>_quadrat_mask.png   255 inside the quadrat cell, 0 elsewhere (same size as crop)
    <stem>_masks_crop.tif     the annotation labels cropped the same way (if they exist)
    <stem>_qc.png             downscaled full frame with the found boundary drawn in red
    quadrat_boxes.json        bounding box of every image, for mapping back to the full frame

Usage:
    python scripts/crop_quadrat.py                       # every photo in raw_data/
    python scripts/crop_quadrat.py --images raw_data/img1.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
from skimage.color import rgb2hsv
from skimage.measure import label
from skimage.segmentation import find_boundaries

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "raw_data"
DEFAULT_IMAGES = [RAW / "img1.png", RAW / "img2.png", RAW / "unseen_img1.png", RAW / "unseen_img2.png"]
DEFAULT_OUT = REPO_ROOT / "processed_data" / "quadrat"

# Sampled on the wire in img1/img2/unseen_img1: hue sits at 0.50 (180 deg, teal)
# throughout, saturation 0.4-0.9, value as low as 0.1 in shadow. The hue band is
# kept wide enough to include the greener side of the wire but stops well short of
# the ruler (~0.1) and the brownish-green algae (~0.2-0.3).
HUE_RANGE = (0.42, 0.56)
MIN_SATURATION = 0.35
# Near-black pixels have noisy, meaningless hue; exclude them.
MIN_VALUE = 0.06


# --------------------------------------------------------------------------- #
# Stage 1-2: wire mask                                                        #
# --------------------------------------------------------------------------- #
def wire_mask(rgb: np.ndarray, close_px: int) -> np.ndarray:
    """Boolean mask of teal wire pixels, with small gaps closed.

    HSV rather than an RGB distance: the wire's brightness changes a lot between
    sunlit and shaded stretches, but its hue barely moves, so thresholding hue and
    saturation is robust to lighting where an RGB distance to one reference colour
    would not be.

    The closing uses a square footprint because scipy's binary_closing with a
    square is separable and fast at 8 megapixels; a disk would be marginally
    rounder at corners but several times slower, and the corners of the cell are
    not where the precision matters.
    """
    hsv = rgb2hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    wire = (h >= HUE_RANGE[0]) & (h <= HUE_RANGE[1]) & (s >= MIN_SATURATION) & (v >= MIN_VALUE)
    if close_px > 0:
        wire = binary_closing(wire, structure=np.ones((close_px, close_px), bool))
    return wire


# --------------------------------------------------------------------------- #
# Stage 3: the inner quadrat is the largest fully enclosed cell               #
# --------------------------------------------------------------------------- #
def find_quadrat(wire: np.ndarray, min_fraction: float) -> np.ndarray | None:
    """Boolean mask of the largest grid cell, or None if no grid is present.

    Cells are background components that do not touch the image border -- the
    same "enclosed region" idea used for the barnacle outlines in
    make_annotations.py. 4-connectivity again, so a diagonal-only wire junction
    cannot leak a cell into its neighbour.

    Why "largest enclosed" and not "the cell containing the image centre": the
    photographer centred the quadrat in every image so far, but nothing guarantees
    that, whereas the inner quadrat is by construction the biggest opening of this
    grid. The runner-up (the strip holding the ruler) is roughly half its area.

    Holes inside the cell (a teal-looking barnacle, a speck of algae) are filled,
    otherwise they would punch holes into the evaluation region.
    """
    cells = label(~wire, connectivity=1)
    border = np.unique(np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]]))
    areas = np.bincount(cells.ravel())
    areas[0] = 0
    areas[border] = 0

    best = int(areas.argmax())
    if areas[best] < min_fraction * wire.size:
        return None
    return binary_fill_holes(cells == best)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(row0, col0, row1, col1) with exclusive ends, i.e. ready for slicing."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def write_qc(rgb: np.ndarray, quadrat: np.ndarray | None, out_path: Path) -> None:
    """Downscaled full frame with the quadrat outline drawn thickly in red."""
    qc = rgb.copy()
    if quadrat is not None:
        edge = find_boundaries(quadrat, mode="inner")
        # Thicken so it survives the downscale.
        qc[binary_dilation(edge, iterations=4)] = (255, 0, 0)
    im = Image.fromarray(qc)
    im.thumbnail((1200, 1200))
    im.save(out_path)


def process_image(path: Path, out_dir: Path, close_px: int, min_fraction: float) -> dict | None:
    stem = path.stem
    rgb = np.array(Image.open(path).convert("RGB"))
    wire = wire_mask(rgb, close_px)
    quadrat = find_quadrat(wire, min_fraction)
    print(f"\n=== {path.name} ({rgb.shape[0]}x{rgb.shape[1]}) ===")
    print(f"  wire pixels         : {wire.mean():.1%} of frame")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_qc(rgb, quadrat, out_dir / f"{stem}_qc.png")

    if quadrat is None:
        # e.g. unseen_img2.png: a 500x500 close-up with no grid in it at all.
        print("  no grid cell found  : image is not cropped, use it whole")
        return None

    r0, c0, r1, c1 = bbox(quadrat)
    inside = quadrat[r0:r1, c0:c1]
    print(f"  quadrat bbox        : rows {r0}-{r1}, cols {c0}-{c1} ({r1 - r0}x{c1 - c0})")
    print(f"  cell fills its bbox : {inside.mean():.1%} (100% = perfectly axis-aligned)")

    Image.fromarray(rgb[r0:r1, c0:c1]).save(out_dir / f"{stem}_crop.png")
    Image.fromarray(inside.astype(np.uint8) * 255).save(out_dir / f"{stem}_quadrat_mask.png")

    labels_path = REPO_ROOT / "processed_data" / f"{stem}_masks.tif"
    if labels_path.exists():
        labels = tifffile.imread(labels_path)[r0:r1, c0:c1]
        outside = np.unique(labels[~inside])
        print(
            f"  annotations cropped : {len(np.unique(labels)) - 1} barnacles in bbox, "
            f"{len(outside) - (0 in outside)} touch the area outside the cell"
        )
        tifffile.imwrite(out_dir / f"{stem}_masks_crop.tif", labels, compression="zlib")

    return {"row0": r0, "col0": c0, "row1": r1, "col1": c1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images", nargs="+", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--close-px",
        type=int,
        default=9,
        help="side of the square closing footprint that bridges wire gaps (default: 9)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.05,
        help="smallest cell, as a fraction of the frame, accepted as the quadrat (default: 0.05)",
    )
    args = parser.parse_args()

    boxes = {}
    for path in args.images:
        box = process_image(path, args.out_dir, args.close_px, args.min_fraction)
        if box is not None:
            boxes[path.stem] = box

    out_json = args.out_dir / "quadrat_boxes.json"
    # Merge so that running on a subset of images keeps the other boxes.
    if out_json.exists():
        boxes = {**json.loads(out_json.read_text()), **boxes}
    out_json.write_text(json.dumps(boxes, indent=2))
    print(f"\nwrote {out_json.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
