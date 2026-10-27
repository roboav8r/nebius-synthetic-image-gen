"""Shared names for this project.

Blog post: "From Isaac Sim output to a curated training set: auditing and
blending synthetic data in FiftyOne."

Three naming decisions worth stating once here, because each of them is a
measurement decision disguised as a string:

- ``EVAL_CLASSES`` is ``pallet_truck`` + ``pallet``, and neither the
  SmartSpaces taxonomy nor the obvious reading of the brief would have
  predicted that. LOCO has **no person class**, so a sim-trained model that
  correctly fires on a warehouse worker scores as a false positive on every
  LOCO image -- person never enters the headline. And LOCO holds only **598
  forklift instances across all 5,097 annotated images** (five of them in
  subset-1), so no held-out subset carries enough forklifts to resolve a
  two-point effect. Forklift is reported as the class this test set cannot
  answer for. ``pallet`` joins the headline because it is the one class that
  is abundant (120,445) and present in every subset, and it is a basic
  SimReady asset we can render.
- ``domain`` and ``viewpoint`` are separate fields on purpose. Our synthetic
  data is rendered under two camera regimes, so viewpoint is *not* a function
  of domain -- which is the only reason the sim-to-real gap here can be
  decomposed rather than just quoted. See DEMO.md.
- ``gt_source`` records who drew each box. Sim GT is exhaustive; LOCO's is
  not; the hand-audited subset is neither. Scoring across those without
  tracking which is which is the error the previous project (RoCoG) spent
  three rounds recovering from.
"""

from __future__ import annotations

DATASET_NAME = "isaac-sim-blend"

GT_FIELD = "ground_truth"

# Everything we model. `person` and `forklift` come from the SmartSpaces 2026
# taxonomy; `pallet` and `pallet_truck` are what LOCO can actually support a
# number on. Our Replicator output is generated to match these names exactly.
CLASSES = ("person", "forklift", "pallet_truck", "pallet")

# The headline is scored on these only. See the module docstring.
EVAL_CLASSES = ("pallet_truck", "pallet")

# Reported, never headlined: too rare in LOCO to resolve an effect.
UNDERPOWERED_CLASSES = ("forklift",)

# LOCO's own five classes, as they appear in the COCO `categories` block.
# `small_load_carrier` and `stillage` have no synthetic counterpart and are
# dropped on import; they are also wildly non-IID across subsets (subset-4 has
# zero small load carriers, subset-5 has one stillage).
LOCO_CATEGORY_MAP = {
    5: "forklift",
    7: "pallet",
    11: "pallet_truck",
    # 3: small_load_carrier  -- dropped, no synthetic counterpart
    # 10: stillage           -- dropped, no synthetic counterpart
}

# LOCO publishes a subset-level train/val split (2/3/5 train, 1/4 val) and no
# test split. We keep their train set intact and cut their val in two, so the
# split is grouped by physical location by construction rather than by
# remembering -- the RoCoG `real-all` contamination trap, which recurs the
# moment this is split by image.
LOCO_SPLITS = {
    "subset-1": "val",
    "subset-2": "train",
    "subset-3": "train",
    "subset-4": "test",
    "subset-5": "train",
}

SYNTHETIC_DOMAINS = ("synthetic-own", "synthetic-nvidia", "synthetic-nvidia-ct25")
REAL_DOMAINS = ("real-loco", "real-nvidia")

# Only `real-loco` carries labels. `real-nvidia` is Warehouse_026/027, whose
# ground truth NVIDIA withholds -- audit and embeddings only, never a number.
LABELLED_REAL_DOMAINS = ("real-loco",)

VIEWPOINTS = ("elevated-fixed", "ground-handheld")

# Who drew the boxes. `human_audit` is the re-audited slice of the real test
# split; `none` is Warehouse_026/027.
GT_SOURCES = ("replicator", "nvidia_gt", "human_coco", "human_audit", "none")

# SmartSpaces scenes we pull. 020/021 is the only Cosmos-Transfer-2.5 pair in
# the val split, and only train/val ship ground_truth.json -- so it is the one
# pair where identical geometry, identical labels and different appearance are
# all available at once.
SMARTSPACES_RELEASE = "MTMC_Tracking_2026"
SMARTSPACES_SCENES = {
    "Warehouse_020": "synthetic-nvidia",
    "Warehouse_021": "synthetic-nvidia-ct25",  # CT2.5 re-render of 020
    "Warehouse_026": "real-nvidia",  # real, GT withheld
    "Warehouse_027": "real-nvidia",  # real, GT withheld
}
CT25_PAIRS = {"Warehouse_021": "Warehouse_020"}

# Frames are stride-sampled to roughly this rate before anything else. Video
# at 30 fps is not 30 samples per second of information, and leaving it dense
# inflates every density and coverage statistic downstream.
TARGET_FPS = 1.0


# Cloud media lives here. john-dev's service account renders gs://voxel51-test
# but NOT gs://voxel51-demo-datasets, so this is the only bucket that works
# for App rendering on that deployment.
GCS_PREFIX = "gs://voxel51-test/datasets/isaac-sim-blend"


# --------------------------------------------------------------- Replicator
#
# Read off `KittiWriter.__init__` in the 4.5.0 container
# (omni.replicator.core-1.11.35), not off the docs, which do not state these.
# Re-read in the 6.1.0 container (omni.replicator.core-1.13.36): the
# signature and all three defaults below are unchanged.
# Three of its defaults are measurement decisions we would otherwise inherit
# silently:
#
# - `bbox_height_threshold=25` **discards every box under 25 px tall before
#   it is ever written**. That is a filter applied upstream of any audit we
#   do, on data we would then describe as "raw simulator output." LOCO's own
#   GT goes far smaller, so leaving this at 25 would hand the synthetic arms
#   an easier label set than the real one and flatter their mAP. We set it to
#   1 and do our own size filtering downstream, where it is visible.
# - `use_kitti_dir_names=False` is the default, so output is `rgb/` +
#   `object_detection/`, *not* the `image_2/` + `label_2/` that every KITTI
#   loader expects. We **stay with the default**. Opting in was the earlier
#   plan, on the theory that standard names are safer -- but we read this
#   output with our own `load_replicator.py`, never with a stock KITTI
#   loader, so the rename buys nothing and would only desynchronize the
#   loader from the smoke output it was verified against.
# - Occlusion is **estimated from the tight/loose bbox area ratio** and
#   emitted only as KITTI's discrete 0/1/2 flag, thresholded at these two
#   values. That is not the continuous visibility fraction the plan calls
#   for, so we attach `bounding_box_2d_tight` and `..._loose` annotators
#   ourselves and compute the ratio per box.
#
# Also: the writer emits **PNG**, while LOCO is JPEG. That is a free domain
# signal with nothing to do with rendering, and it is exactly what
# `normalize.py` exists to remove.
# `KittiWriter`'s `mapping_dict`: class -> RGBA in the semantic-segmentation
# PNGs it writes alongside the labels. Those masks are what lets us measure
# how much of each written box is *actually the object* rather than
# background seen through or around it -- `KittiWriter`'s own `occluded`
# 0/1/2 flag is estimated from the tight/loose area ratio, and its own
# docstring warns that estimate reports a split-by-an-occluder object as
# fully visible. Hand-synced with `sdg/generate_replicator.py`.
SEG_COLOURS = {
    "pallet": (211, 210, 223, 255),
    "pallet_truck": (11, 110, 223, 255),
    "forklift": (255, 140, 0, 255),
}

# A box whose own class covers less than this share of its area is mostly
# not the object: the tight box bounds the visible pixels, so a heavily
# occluded object still gets a box around whatever sliver shows.
MIN_FILL_FRAC = 0.10

# LOCO's *practical annotation floor*: the 0.1st percentile of its relative
# box area (box area / image area), measured over all 123,870 boxes in the
# eval classes by `measure_loco_box_stats.py`. 0.000063 is 32.4 px^2 at our
# 960x540, and relative rather than absolute because LOCO mixes 640x480
# through 1920x1080.
#
# This is a *measured* threshold, not a taste call, and it is the answer to
# "are the tiny distant shelf pallets acceptable?":
#
#   - The bulk of our distribution matches LOCO closely -- median relative
#     area 0.00210 synthetic vs 0.00228 real.
#   - The small tail does not. **8.7% of synthetic boxes fall below this
#     floor against LOCO's 0.1%**, and below 10 px^2 the synthetic rate is
#     ~900x LOCO's. Those are labels no human annotator produced.
#
# They exist because we set `bbox_height_threshold=1` deliberately (LOCO is
# 32% sub-25px, so the writer's default of 25 would bias the other way).
# The fix is therefore *not* to raise the writer's threshold -- that would
# re-introduce the bias -- but to keep emitting them, flag them, and let
# curation drop them on an explicit rule. Nothing needs re-rendering.
LOCO_AREA_FLOOR = 0.000063

# Per-box audit flags attached by `load_replicator.py`. Ordered roughly by
# how much each one would hurt a detector trained on it.
AUDIT_DEFECTS = (
    "mostly_hidden",   # < MIN_FILL_FRAC of the box is actually the object
    "below_loco_floor",  # smaller than LOCO's annotators ever drew
    "dup_geom",        # exact-geometry duplicate of another row
    "full_frame",      # covers >= 95% of the image
    "touches_edge",    # clipped against an image border
)

# The curation ladder. `keep` is False for any box carrying a defect in this
# set; `touches_edge` and `under_25px` are deliberately NOT in it, because
# LOCO is full of both (32% of its boxes are sub-25px) and dropping them
# would make the synthetic arm easier than the test set.
CURATION_DROPS = ("mostly_hidden", "below_loco_floor", "dup_geom", "full_frame")

REPLICATOR_WRITER = "KittiWriter"
REPLICATOR_WRITER_CONFIG = {
    "use_kitti_dir_names": False,
    # 30.5% of LOCO's pallet boxes are under 25 px tall (measured), so the
    # writer's default of 25 would train the synthetic arms on a strictly
    # easier label set than the real test set scores them against.
    "bbox_height_threshold": 1,
    "partly_occluded_threshold": 0.5,
    "fully_visible_threshold": 0.95,
    "colorize_instance_segmentation": False,
    "omit_semantic_type": True,
    # A fourth default that is a measurement decision: `semantic_filter_predicate`
    # defaults to `"class:*"`, i.e. annotate every semantic on the stage. That
    # is how Round 4 got whole-image `floor` boxes and Round 7 got `box`,
    # `barel` and the multi-label `pallet,prop_general`. Narrowing it here is
    # also the *only* lever that reaches semantics baked into instance
    # prototypes -- `remove_all_semantics` cannot author over an instance
    # proxy, so clearing the stage silently leaves them standing.
    "semantic_filter_predicate": "class: pallet | pallet_truck | forklift",
}

# Camera regimes. Focal lengths assume Replicator's default 20.955 mm
# horizontal aperture: 24 mm ~= 47 deg hFOV, 16 mm ~= 66 deg.
#
# `elevated-fixed` is mounted on the perimeter of the scatter area and looks
# obliquely across it, which is what a *fixed* camera is. Sampling its
# position uniformly inside that area instead put it straight above the
# objects: median box 1.48% of frame against LOCO's 0.22%, and only 10% of
# boxes under 25 px. From the perimeter at ~13 m range it measures 0.64%
# and 31%.
#
# `ground-handheld` aims at a widely-jittered point rather than taking a free
# yaw. Free yaw is the more honest model of a person not framing carefully,
# but measured it put 55% of ground frames on a blank wall, and LOCO contains
# no such images -- someone took each of those photographs *of* something.
# The imperfect framing now comes from the spread of the aim point.
CAMERA_REGIMES = {
    "elevated-fixed": {"height_m": (5.5, 8.0), "focal_mm": 24.0, "mount": "perimeter"},
    "ground-handheld": {"height_m": (1.2, 1.7), "focal_mm": 16.0, "mount": "in-scene"},
}

# Object pool per frame, tuned against LOCO_DENSITY below on 20-frame
# validation runs. Visible counts emerge from camera geometry, and the stock
# warehouse contributes its own 158 labelled pallets on top of these, which
# is why the spawned pool is smaller than LOCO's per-image count.
#
# Kept in sync by hand with `sdg/generate_replicator.py`'s CONFIG: that script
# runs inside the Isaac Sim container and cannot import this module.
SDG_POOL = {
    "n_pallet": 18,
    "n_pallet_truck": 8,
    "n_forklift": 4,
    "n_cardbox": 20,
    "n_distractor": 10,
}

# Measured from LOCO, and the reason our scenes are dense rather than the
# stock example's one-forklift-one-pallet: 23.6 pallets per image, median
# box 0.22% of frame area. Sim that renders a few large objects cannot
# produce the small-object regime the real test set is full of.
LOCO_DENSITY = {
    "pallet": {"mean_per_image": 23.63, "p50": 15, "p90": 50},
    "pallet_truck": {"mean_per_image": 0.55, "p50": 0, "p90": 2},
    "forklift": {"mean_per_image": 0.12, "p50": 0, "p90": 0},
}

# KITTI label columns. The writer sets alpha, dimensions, location,
# rotation_y and truncated to 0.0 -- they are placeholders, not measurements.
# `load_replicator.py` must read only `type`, `occluded` and the 2D box.
KITTI_UNSUPPORTED_FIELDS = (
    "truncated", "alpha", "height", "width", "length", "x", "y", "z", "rotation_y",
)

RENDERER = "RaytracedLighting"  # never PathTracing on a 70 W Turing card


def pred_field(arm: str, seed: int) -> str:
    """``("curated", 0) -> "curated_s0_pred"``."""
    return f"{arm.replace('-', '_')}_s{seed}_pred"
