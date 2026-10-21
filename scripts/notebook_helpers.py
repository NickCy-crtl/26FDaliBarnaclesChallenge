"""Display helpers for "Conclusions and Thoughts.ipynb".

The notebook only *reads* what the pipeline scripts already wrote (metrics.json,
overlays, QC images). Nothing here runs Cellpose, so the notebook opens and
re-executes in seconds, without torch, and without the warnings Cellpose prints.
The alternative of running the pipeline from notebook cells was rejected: one
full run takes about 15 minutes and its log output would bury the discussion.
reproduce.sh is the single place that regenerates the files shown here.

Usage (from the notebook at the repo root):
    import sys; sys.path.insert(0, "scripts")
    from notebook_helpers import load_metrics, results_table, show_overlay, compare_overlays
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

# The raw frames are 2448x3264; Pillow's decompression-bomb guard trips below that.
Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = REPO_ROOT / "processed_data" / "cellpose_zeroshot" / "metrics.json"
RESULTS = REPO_ROOT / "results"
PRED_DIR = REPO_ROOT / "processed_data" / "cellpose_zeroshot"

# Groups in the order they were run, so tables read as the story of the project
# rather than alphabetically.
GROUP_ORDER = [
    "first_Zeroshot",
    "Zeroshot_with_Diameter",
    "RGB_norm_p1-99.5_tile150",
    "CLAHE_L_gray",
    "Composite_tilenorm",
    "Composite_DoG_CLAHE_gray_tilenorm",
    "Composite_CLAHEx3_tilenorm",
    "Composite_CLAHEx2_DoG_tilenorm",
]
COLUMNS = ["group", "model", "image", "diameter", "n_gt", "n_pred", "count_error_pct",
           "precision", "recall", "f1"]


def load_metrics(path: Path = METRICS) -> pd.DataFrame:
    """metrics.json as one row per group x model x image, in run order.

    The nested normalize/preprocess settings are dropped: the group name already
    encodes them, and reproduce.sh lists the exact flags behind each group.
    """
    df = pd.DataFrame(json.loads(Path(path).read_text()))
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    return df.sort_values(["group", "image", "model"], kind="stable")[COLUMNS].reset_index(drop=True)


def results_table(
    df: pd.DataFrame, groups: list[str] | None = None, images: list[str] | None = ("img1", "img2")
):
    """The rows for some groups/images, styled so the best F1 per image stands out.

    Returns a pandas Styler, which renders as an HTML table in the notebook.
    Default images are the two with ground truth; pass images=None for all four.
    """
    sub = df
    if groups is not None:
        sub = sub[sub["group"].isin(groups)]
    if images is not None:
        sub = sub[sub["image"].isin(images)]
    sub = sub.copy()
    sub["group"] = sub["group"].astype(str)
    sub = sub.set_index(["group", "image", "model"])

    def bold_best(col: pd.Series) -> list[str]:
        best = col.groupby(level="image").transform("max")
        return ["font-weight: bold" if v == b else "" for v, b in zip(col, best)]

    return (
        sub.style.format(
            {"diameter": "{:g}", "n_gt": "{:.0f}", "count_error_pct": "{:+.1f}%",
             "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"},
            na_rep="-",
        )
        .apply(bold_best, subset=["f1"])
    )


def _zoom_box(ax, zoom: tuple[int, int, int]) -> None:
    r, c, size = zoom
    ax.add_patch(Rectangle((c, r), size, size, fill=False, edgecolor="cyan", linewidth=1.5))


def show_overlay(path: Path | str, title: str = "", zoom: tuple[int, int, int] | None = None):
    """An overlay at full size, plus an optional zoomed-in window next to it.

    zoom = (row, col, size) in overlay pixels. The overlays draw 1-px outlines,
    which disappear when a ~1500 px image is shrunk to notebook width, so the
    zoom panel is where the reader can actually judge single barnacles; the
    cyan box on the left panel shows where it comes from.
    """
    img = Image.open(REPO_ROOT / path)
    ncols = 2 if zoom else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 8), squeeze=False)
    axes = axes[0]
    axes[0].imshow(img)
    axes[0].set_title(title or Path(path).stem)
    if zoom:
        r, c, size = zoom
        _zoom_box(axes[0], zoom)
        axes[1].imshow(img.crop((c, r, c + size, r + size)), interpolation="nearest")
        axes[1].set_title(f"zoom: rows {r}-{r + size}, cols {c}-{c + size}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def compare_overlays(paths: list[Path | str], titles: list[str], zoom: tuple[int, int, int]):
    """The same zoom window from several overlays side by side (e.g. before/after a setting).

    Only the zoom is shown: all overlays of one image share the same crop
    coordinates, so the same window lines up exactly across runs.
    """
    r, c, size = zoom
    fig, axes = plt.subplots(1, len(paths), figsize=(6 * len(paths), 6), squeeze=False)
    for ax, path, title in zip(axes[0], paths, titles):
        ax.imshow(Image.open(REPO_ROOT / path).crop((c, r, c + size, r + size)), interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig
