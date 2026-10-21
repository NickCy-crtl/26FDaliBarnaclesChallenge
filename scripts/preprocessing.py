"""Image preprocessing and Cellpose normalization settings, used by run_cellpose_zeroshot.py.

Cellpose always normalizes its input unless told not to. Its default is a global
per-channel rescale with the 1st percentile mapped to 0 and the 99th to 1. This
module provides two independent ways to change what the network sees. They can be
combined.

1. Cellpose's own normalization (build_normalize)
   The percentiles can be changed, and a tile blocksize > 0 computes them per
   overlapping N x N tile instead of over the whole crop, which brightens dark
   regions locally. Cellpose normalizes *before* the diameter rescale, so N is in
   crop pixels, not in rescaled pixels.

2. An image transform applied before Cellpose (preprocess)
   'clahe_l': CLAHE on the L channel of CIELAB, with that grayscale image copied
   into all three channels. L separates lightness from colour, so the contrast
   stretch does not shift hues, as equalizing R, G and B separately would. Copying
   the result into three channels keeps the input shape the RGB-pretrained
   network expects. Cellpose's own normalization still runs afterwards.

   'composite': three single-channel views, each fed to the model as one of its
   3 input channels. Nothing is blended. The channels are chosen from
   COMPOSITE_CHANNELS (e.g. dog, clahe_l, gray). The Cellpose-SAM models take 3
   channels in any order, without assuming what each one means, so each channel
   can carry a different cue. See composite().

settings_suffix() names every non-default setting for output file names, so a
run with new settings writes new files instead of overwriting old ones.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import difference_of_gaussians

PREPROCESS_METHODS = ["none", "clahe_l", "composite"]
COMPOSITE_CHANNELS = ["clahe_l", "gray", "inv_sat", "dog"]


def clahe_l_gray(rgb: np.ndarray, kernel_size: int, clip_limit: float) -> np.ndarray:
    """CLAHE on CIELAB lightness, returned as uint8 grayscale copied into 3 channels.

    skimage's equalize_adapthist is used, not OpenCV's createCLAHE. It gives the
    same algorithm without adding opencv to requirements.txt. Note that the two
    libraries express the clip limit in different units: skimage uses a fraction
    in [0, 1] (default 0.01), while OpenCV uses a multiple of the mean bin count
    (default 2.0).

    The kernel size is set explicitly. skimage's default is 1/8 of each image
    dimension, which would give a different neighbourhood on every crop, since
    img1 and img2 have different sizes. A fixed size in pixels keeps the step
    comparable across images and with the tile-normalization blocksize.
    """
    gray = to_uint8(clahe_l(rgb, kernel_size, clip_limit))
    return np.repeat(gray[..., None], 3, axis=-1)


def clahe_l(rgb: np.ndarray, kernel_size: int, clip_limit: float) -> np.ndarray:
    """CLAHE-equalized CIELAB lightness as float in [0, 1] (one channel)."""
    lightness = rgb2lab(rgb)[..., 0] / 100.0  # L* is in [0, 100]
    return equalize_adapthist(lightness, kernel_size=kernel_size, clip_limit=clip_limit)


def to_uint8(channel: np.ndarray, percentile: tuple[float, float] | None = None) -> np.ndarray:
    """Scale a float channel to uint8, either from [0, 1] or by clipping to percentiles.

    The model inputs are stored as uint8 so they can be saved as PNGs for
    inspection. The quantization costs nothing here, because Cellpose rescales
    every channel by its own percentiles afterwards anyway.
    """
    in_range = (0.0, 1.0) if percentile is None else tuple(np.percentile(channel, percentile))
    return rescale_intensity(channel, in_range=in_range, out_range=(0, 255)).astype(np.uint8)


def composite(
    rgb: np.ndarray, channels: list[str], diameter: float | None, clahe_kernel: int,
    clahe_clip: float, dog_small_sigma: float, dog_large_frac: float,
) -> np.ndarray:
    """Stack three single-channel views as the model's 3 input channels (uint8 HxWx3).

    Available channels:
      clahe_l  CLAHE-equalized lightness, the same as clahe_l_gray.
      gray     The original image as plain grayscale luminance, with no
               enhancement. It keeps the unprocessed signal next to the
               enhanced ones. A single R, G or B channel would throw away
               information that luminance keeps.
      inv_sat  1 - HSV saturation. Shells are pale and nearly grey, so they come
               out bright. Run 'Composite_tilenorm' used it, and it carried
               almost no signal on img2, whose wet rock is also low-saturation.
      dog      Difference of Gaussians on (non-CLAHE) L: blur(sigma_small)
               minus blur(sigma_large), with sigma_large = diameter * frac. This
               band-pass removes pixel noise and slow illumination gradients and
               keeps structure around barnacle scale: dark openings become
               negative blobs and rims become positive rings. It is signed, so it
               is mapped to uint8 by its 1st/99th percentiles.
    Each channel is built only if it is asked for, so 'dog' is the only one that
    needs a diameter.
    """
    lightness = lambda: rgb2lab(rgb)[..., 0] / 100.0  # noqa: E731  L* is in [0, 100]

    def dog():
        if diameter is None:
            raise ValueError("the 'dog' channel needs a diameter (sigma_large = diameter * frac)")
        band = difference_of_gaussians(lightness(), dog_small_sigma, diameter * dog_large_frac)
        return to_uint8(band, percentile=(1, 99))

    builders = {
        "clahe_l": lambda: to_uint8(clahe_l(rgb, clahe_kernel, clahe_clip)),
        "gray": lambda: to_uint8(rgb2gray(rgb)),
        "inv_sat": lambda: to_uint8(1.0 - rgb2hsv(rgb)[..., 1]),
        "dog": dog,
    }
    if len(channels) != 3:
        raise ValueError(f"composite needs exactly 3 channels, got {channels}")
    return np.stack([builders[name]() for name in channels], axis=-1)


def preprocess(
    rgb: np.ndarray, method: str, clahe_kernel: int, clahe_clip: float,
    diameter: float | None = None, channels: list[str] | None = None,
    dog_small_sigma: float = 1.5, dog_large_frac: float = 0.25,
) -> np.ndarray:
    """Apply the named preprocessing method; 'none' returns the image unchanged."""
    if method == "none":
        return rgb
    if method == "clahe_l":
        return clahe_l_gray(rgb, clahe_kernel, clahe_clip)
    if method == "composite":
        return composite(
            rgb, channels, diameter, clahe_kernel, clahe_clip, dog_small_sigma, dog_large_frac
        )
    raise ValueError(f"unknown preprocessing method {method!r}; choose from {PREPROCESS_METHODS}")


def preprocess_record(
    method: str, clahe_kernel: int, clahe_clip: float, channels: list[str] | None = None,
    dog_small_sigma: float = 1.5, dog_large_frac: float = 0.25,
) -> dict | None:
    """The preprocessing settings as stored in metrics.json (None for 'none').

    The large DoG sigma is stored as a fraction of the diameter, since its pixel
    value differs per image; the row's own 'diameter' field gives the rest.
    (Group 'Composite_tilenorm' predates the 'channels' key; it used
    clahe_l, inv_sat, dog.)
    """
    if method == "clahe_l":
        return {"method": method, "kernel_size": clahe_kernel, "clip_limit": clahe_clip}
    if method == "composite":
        return {
            "method": method, "channels": list(channels),
            "kernel_size": clahe_kernel, "clip_limit": clahe_clip,
            "dog_small_sigma": dog_small_sigma, "dog_large_sigma_frac": dog_large_frac,
        }
    return None


def method_tag(method: str, channels: list[str] | None = None) -> str:
    """Short name of the preprocessing for file names, e.g. 'claheL' or 'comp-dog-clahe_l-gray'.

    The composite tag lists its channels in order, so composites with different
    channels never write to the same files.
    """
    if method == "clahe_l":
        return "claheL"
    if method == "composite":
        return "comp-" + "-".join(channels)
    return ""


def build_normalize(percentile: list[float] | None, tile_blocksize: int) -> bool | dict:
    """The `normalize` argument for CellposeModel.eval.

    Returns plain True when nothing is overridden, so the default runs call
    Cellpose exactly as before. A dict holds only the keys that were overridden:
    Cellpose merges it onto its own defaults (cellpose.models.normalize_default).
    """
    params = {}
    if percentile is not None:
        params["percentile"] = [float(p) for p in percentile]
    if tile_blocksize:
        params["tile_norm_blocksize"] = int(tile_blocksize)
    return params or True


def settings_suffix(
    diameter: float | None, normalize: bool | dict, method: str, channels: list[str] | None = None
) -> str:
    """File-name suffix naming every non-default setting, e.g. '_d50_p1-99.5_tn150'.

    Defaults add nothing, so the original zero-shot and diameter runs keep their
    existing file names and can be reproduced in place.
    """
    suffix = "" if diameter is None else f"_d{diameter:g}"
    if isinstance(normalize, dict):
        if "percentile" in normalize:
            lo, hi = normalize["percentile"]
            suffix += f"_p{lo:g}-{hi:g}"
        if "tile_norm_blocksize" in normalize:
            suffix += f"_tn{normalize['tile_norm_blocksize']}"
    if method != "none":
        suffix += "_" + method_tag(method, channels)
    return suffix
