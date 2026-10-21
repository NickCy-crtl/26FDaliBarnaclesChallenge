#!/usr/bin/env bash
# Rebuild every result in "Conclusions and Thoughts.ipynb" from raw_data/.
#
# Run from the repository root with the virtualenv active:
#     bash reproduce.sh
# Total runtime on an Apple M4 (MPS) is about 15 minutes, almost all of it Cellpose.
#
# CELLPOSE_OUT redirects the Cellpose predictions, overlays and metrics.json (for
# example to check a rerun against the committed numbers without overwriting them).
# It must be a folder inside the repo, because the scripts print paths relative to it:
#     CELLPOSE_OUT=rerun_check bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"

# Absolute, because the scripts print output paths relative to the repo root and
# Path.relative_to fails on a relative path.
OUT="$(pwd)/${CELLPOSE_OUT:-processed_data/cellpose_zeroshot}"
RUN="python scripts/run_cellpose_zeroshot.py --out-dir $OUT"
D="--images img1 img2 --diameter img1=30 img2=50"
TN="--norm-percentile 1 99.5 --tile-norm-blocksize 150"

# 1. Ground truth: outline masks -> integer label images (processed_data/img{1,2}_masks.tif).
python scripts/make_annotations.py

# 2. Crop every photo (and the labels) to the inner quadrat (processed_data/quadrat/).
#    Must run after step 1, because it crops the label images too.
python scripts/crop_quadrat.py

# 3. Zero-shot baseline, Cellpose defaults, both models, all four images.
$RUN --group first_Zeroshot

# 4. Same, with a per-image diameter estimated by eye.
$RUN $D --group Zeroshot_with_Diameter
$RUN --models cpsam --images unseen_img1 unseen_img2 --diameter 30 --group Zeroshot_with_Diameter

# 5. Preprocessing experiments (all with the diameters from step 4).
$RUN $D $TN --group RGB_norm_p1-99.5_tile150
$RUN $D --preprocess clahe_l --group CLAHE_L_gray
$RUN $D $TN --preprocess composite --composite-channels clahe_l inv_sat dog --group Composite_tilenorm
$RUN $D $TN --preprocess composite --composite-channels dog clahe_l gray --group Composite_DoG_CLAHE_gray_tilenorm
$RUN $D $TN --preprocess composite --composite-channels clahe_l clahe_l clahe_l --group Composite_CLAHEx3_tilenorm
$RUN $D $TN --preprocess composite --composite-channels clahe_l clahe_l dog --group Composite_CLAHEx2_DoG_tilenorm

# 6. The deliverable overlays: cpsam, plain RGB, with diameter.
mkdir -p results
for tag in img1_d30 img2_d50 unseen_img1_d30 unseen_img2_d30; do
    cp "$OUT/cpsam/${tag}_overlay.png" "results/${tag}_overlay.png"
done
