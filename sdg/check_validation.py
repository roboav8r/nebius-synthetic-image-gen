#!/usr/bin/env python
"""Gate a Replicator validation run before committing to the long one.

Round 7's failure was not that the renderer broke -- it hit 11.47 s/image,
exactly as predicted -- but that nobody looked at what it was rendering until
after the fact. This makes looking cheap: it prints a PASS/FAIL line per
criterion and writes a contact sheet with the boxes drawn on, so twenty
frames can be judged in one glance.

    python sdg/check_validation.py                        # _out_sdg_val
    python sdg/check_validation.py --out-dir <dir>

The criteria are LOCO's measured statistics, not round numbers: the synthetic
arm is scored against a real test set, so anything the generator makes
systematically easier than LOCO is a bias we would be training in.
"""

from __future__ import annotations

import argparse
import ast
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import (  # noqa: E402
    LOCO_AREA_FLOOR,
    LOCO_DENSITY,
    MIN_FILL_FRAC,
    SEG_COLOURS,
)

TARGET_CLASSES = {"pallet", "pallet_truck", "forklift"}

# LOCO: median box is 0.22% of frame area and 30.5% of pallet boxes are under
# 25 px tall. A generator that produces only big, comfortable boxes is not
# reproducing the problem the real test set poses.
LOCO_P50_BOXES = LOCO_DENSITY["pallet"]["p50"]
DENSITY_BAND = (8, 30)          # acceptable p50 boxes/frame around LOCO's 15
MAX_EMPTY_FRAC = 0.15           # frames with no labels at all
MIN_SMALL_BOX_FRAC = 0.15       # boxes under 25 px tall
FULL_FRAME_FRAC = 0.95
MAX_FULL_FRAME_FRAC = 0.005    # share of rows allowed to fill the frame
MAX_HIDDEN_FRAC = 0.03         # share of rows that may be mostly hidden
# The below-floor share is a **drift monitor, not a match target.** LOCO's
# own 0.1% is tautological -- the floor is defined as LOCO's p0.1 -- so
# "matching" it is not a meaningful goal. The tail here is structural: from
# floor level this warehouse always has a rack run full of real pallets in
# shot, and three camera configurations moved the number only between 10.3%
# and 15.3%. Those boxes are correct labels on genuine pallets, kept on
# purpose (unlabelled positives are worse), and removed by curation rather
# than by re-rendering. This bound exists to catch the number *changing*.
MAX_BELOW_FLOOR_FRAC = 0.20

# What curation actually has to hand a trainer. This is the decision-relevant
# number: if the ladder is throwing away more than this, the generator has a
# problem no amount of filtering will fix.
MIN_CURATION_YIELD = 0.85

ROW = re.compile(r'^"(?P<type>.*?)"\s+(?P<rest>.*)$')


def parse_type(raw: str) -> str:
    """`omit_semantic_type=False` writes `"{'class': 'forklift'}"`."""
    raw = raw.strip().strip('"')
    if raw.startswith("{"):
        try:
            return str(next(iter(ast.literal_eval(raw).values())))
        except Exception:  # noqa: BLE001
            return raw
    return raw


def read_labels(path: Path):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        m = ROW.match(line)
        cls, rest = (parse_type(m["type"]), m["rest"]) if m else (
            parse_type(line.split()[0]), " ".join(line.split()[1:])
        )
        f = rest.split()
        try:
            x1, y1, x2, y2 = (float(v) for v in f[3:7])
        except (IndexError, ValueError):
            continue
        out.append((cls, x1, y1, x2, y2))
    return out


def fill_fractions(det_dir: Path, rows_by_frame):
    """Per-box share of its own class's pixels, from the semantic mask.

    `KittiWriter` writes the tight box, which bounds only an object's
    *visible* pixels -- so an object almost entirely behind something else
    still gets a box, drawn around the sliver that shows. Its `occluded`
    0/1/2 column is meant to flag that and does not: measured here, boxes
    flagged `occluded==2` have a *higher* median fill (0.75) than boxes
    flagged fully visible (0.56). The flag is estimated from the
    tight/loose area ratio, which on a rotated or frame-clipped object moves
    for reasons that have nothing to do with occlusion.

    The semantic mask the writer saves alongside is unambiguous, so we use
    that instead.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    seg_dir = det_dir.parent / "semantic_segmentation"
    out = []
    for stem, rows in rows_by_frame:
        seg_path = seg_dir / f"{stem}.png"
        if not seg_path.is_file():
            continue
        seg = np.array(Image.open(seg_path).convert("RGBA"))
        h, w = seg.shape[:2]
        for cls, x1, y1, x2, y2 in rows:
            colour = SEG_COLOURS.get(cls)
            xa, ya = max(0, int(x1)), max(0, int(y1))
            xb, yb = min(w, int(round(x2))), min(h, int(round(y2)))
            if colour is None or xb <= xa or yb <= ya:
                continue
            crop = seg[ya:yb, xa:xb]
            hit = int((crop == colour).all(axis=-1).sum())
            out.append(hit / float((xb - xa) * (yb - ya)))
    return out


def regimes(out_dir: Path):
    for det in sorted(out_dir.glob("*/*/object_detection")):
        yield det.parent.parent.name, det.parent, det


def contact_sheet(rgb_dir: Path, labels_dir: Path, dest: Path, cols: int = 5, w: int = 384,
                  max_frames: int = 30):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print(f"  (no Pillow; skipping contact sheet {dest.name})")
        return None
    files = sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not files:
        return None
    if len(files) > max_frames:
        step = len(files) / float(max_frames)
        files = [files[int(i * step)] for i in range(max_frames)]
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        sx, sy = w / im.width, (w * im.height // im.width) / im.height
        im = im.resize((w, w * im.height // im.width))
        d = ImageDraw.Draw(im)
        lf = labels_dir / f"{f.stem}.txt"
        if lf.exists():
            for cls, x1, y1, x2, y2 in read_labels(lf):
                colour = {"pallet": (80, 200, 255), "pallet_truck": (255, 200, 40),
                          "forklift": (255, 90, 90)}.get(cls, (160, 160, 160))
                d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=colour, width=1)
        thumbs.append(im)
    tw, th = thumbs[0].size
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (18, 18, 18))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="scratch/isaac/_out_sdg_val")
    ap.add_argument("--log", default=None, help="run log to scan for renderer errors")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        print(f"no such output dir: {out_dir}")
        return 1

    log = Path(args.log) if args.log else Path("scratch/isaac/logs") / f"sdg_{out_dir.name}.log"
    failures = []

    if log.exists():
        n_scatter = len(re.findall(r"Scatter2D Error", log.read_text(errors="ignore")))
        ok = n_scatter == 0
        print(f"[{'PASS' if ok else 'FAIL'}] renderer errors: {n_scatter} Scatter2D")
        if not ok:
            failures.append("Scatter2D errors in the run log")
    else:
        print(f"[ -- ] run log not found at {log}")

    for regime, cam_dir, det_dir in regimes(out_dir):
        rgb_dir = cam_dir / "rgb"
        print(f"\n=== {regime} ===")
        per_frame, classes, areas, heights = [], Counter(), [], []
        full_frame = 0
        dup_rows = 0
        label_files = sorted(det_dir.glob("*.txt"), key=lambda p: int(p.stem))
        rows_by_frame = []
        for lf in label_files:
            rows = read_labels(lf)
            rows_by_frame.append((lf.stem, rows))
            per_frame.append(len(rows))
            # Round 4: 268 of 330 stock rows were exact-geometry duplicates,
            # because multi-label semantics emit one row per label string.
            # Assets keep their baked semantics now, so this can come back.
            seen = set()
            for r in rows:
                if r[1:] in seen:
                    dup_rows += 1
                seen.add(r[1:])
            for cls, x1, y1, x2, y2 in rows:
                classes[cls] += 1
                heights.append(y2 - y1)
                areas.append((x2 - x1) * (y2 - y1))
                if (x2 - x1) * (y2 - y1) >= FULL_FRAME_FRAC * 960 * 540:
                    full_frame += 1

        if not per_frame:
            print("  no label files")
            failures.append(f"{regime}: no labels")
            continue

        p50 = statistics.median(per_frame)
        empty = sum(1 for n in per_frame if n == 0) / len(per_frame)
        small = sum(1 for h in heights if h < 25) / len(heights) if heights else 0.0
        med_area = statistics.median(areas) / (960 * 540) * 100 if areas else 0.0

        rel_areas = [
            ((x2 - x1) * (y2 - y1)) / (960.0 * 540.0)
            for _, rows in rows_by_frame
            for _, x1, y1, x2, y2 in rows
        ]
        below_floor = (
            sum(1 for a in rel_areas if a < LOCO_AREA_FLOOR) / len(rel_areas)
            if rel_areas else 0.0
        )

        fills = fill_fractions(det_dir, rows_by_frame)
        hidden = (
            sum(1 for f in fills if f < MIN_FILL_FRAC) / len(fills) if fills else 0.0
        )

        # Mirror `load_replicator.CURATION_DROPS` so the gate reports the same
        # number the importer will. `touches_edge` is deliberately not a drop.
        n_rows = len(rel_areas)
        dropped = sum(
            1
            for i, a in enumerate(rel_areas)
            if a < LOCO_AREA_FLOOR
            or (i < len(fills) and fills[i] < MIN_FILL_FRAC)
        ) + dup_rows + full_frame
        yield_frac = 1.0 - (dropped / n_rows) if n_rows else 0.0

        checks = [
            ("label vocabulary", set(classes) <= TARGET_CLASSES and set(classes),
             f"{dict(classes)}"),
            (f"p50 boxes/frame in {DENSITY_BAND} (LOCO {LOCO_P50_BOXES})", 
             DENSITY_BAND[0] <= p50 <= DENSITY_BAND[1], f"p50={p50}"),
            (f"empty frames <= {MAX_EMPTY_FRAC:.0%}", empty <= MAX_EMPTY_FRAC, f"{empty:.0%}"),
            (f"sub-25px boxes >= {MIN_SMALL_BOX_FRAC:.0%}", small >= MIN_SMALL_BOX_FRAC,
             f"{small:.0%}"),
            # Round 4's defect was 17 near-full-frame boxes in 330 rows (5%),
            # emitted systematically for `floor` and `forklift`. A single
            # frame-filling forklift because the camera is standing next to
            # one is real data, and LOCO contains those, so this gates the
            # rate rather than demanding zero.
            (f"full-frame boxes <= {MAX_FULL_FRAME_FRAC:.1%}",
             full_frame <= MAX_FULL_FRAME_FRAC * max(1, sum(per_frame)),
             f"{full_frame}/{sum(per_frame)}"),
            ("no duplicate-geometry rows", dup_rows == 0,
             f"{dup_rows}/{sum(per_frame)}"),
            # Objects that are physically inside one another produce boxes
            # around things you cannot see. This is the measurement that
            # catches it, and it is why placement is collision-free.
            (f"mostly-hidden boxes <= {MAX_HIDDEN_FRAC:.0%}",
             hidden <= MAX_HIDDEN_FRAC,
             f"{hidden:.1%} below {MIN_FILL_FRAC:.0%} fill"),
            # Boxes smaller than LOCO's annotators ever drew. A five-pixel
            # sliver of a pallet between two rack beams is a label no human
            # produced, and `bbox_height_threshold=1` lets it through.
            (f"below LOCO's area floor <= {MAX_BELOW_FLOOR_FRAC:.0%} (drift)",
             below_floor <= MAX_BELOW_FLOOR_FRAC,
             f"{below_floor:.1%}"),
            (f"curation yield >= {MIN_CURATION_YIELD:.0%}",
             yield_frac >= MIN_CURATION_YIELD,
             f"{yield_frac:.1%} of boxes survive the ladder"),
        ]
        for name, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            if not ok:
                failures.append(f"{regime}: {name} ({detail})")
        print(f"   .... median box area {med_area:.2f}% of frame (LOCO 0.22%)")
        if fills:
            print(f"   .... median fill {statistics.median(fills):.2f} "
                  f"(share of box that is actually the object)")
        srt = sorted(per_frame)
        qs = [0, 10, 25, 50, 75, 90, 100]
        pcts = " ".join(
            f"p{q}={srt[min(len(srt) - 1, q * len(srt) // 100)]:g}" for q in qs)
        print(f"   .... boxes/frame {pcts}")

        sheet = contact_sheet(rgb_dir, det_dir, out_dir / f"contact_{regime}.jpg")
        if sheet:
            print(f"   .... contact sheet {sheet}")

    print()
    if failures:
        print(f"GATE FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
