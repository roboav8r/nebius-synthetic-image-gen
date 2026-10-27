"""Quality gate for a finished run: one PASS/FAIL line per check, per regime.

    python -m sdg.gate <run output dir>

It also writes `contact_<regime>.jpg` next to the regimes, a grid of frames
with their boxes drawn on, so a run can be judged at a glance.

The thresholds come from LOCO (Logistics Objects in Context), the real
warehouse dataset this synthetic data is meant to match. A generator that
produces boxes systematically easier than LOCO's trains in a bias, so each
check compares against a LOCO measurement rather than a round number.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sdg.scene import SEG_COLOURS, TARGET_CLASSES

# --- thresholds ------------------------------------------------------------

# LOCO has a median of 15 pallets per image.
DENSITY_BAND = (8, 30)        # median boxes per frame
MAX_EMPTY_FRAC = 0.15         # frames with no boxes at all
# 30.5% of LOCO's pallet boxes are under 25 px tall. Fewer than this means the
# synthetic boxes are easier than the real ones.
SMALL_BOX_PX = 25
MIN_SMALL_BOX_FRAC = 0.15
# A camera standing next to a forklift is real data; a box that fills the
# frame on every other image is a labelling bug. Gate the rate, not the count.
FULL_FRAME_FRAC = 0.95        # of the image area
MAX_FULL_FRAME_FRAC = 0.005   # of all boxes
# A tight box bounds only the visible pixels, so an object almost entirely
# behind another still gets a box around the sliver that shows. "Fill" is the
# share of the box that is the object's own class in the semantic mask.
MIN_FILL = 0.10
MAX_HIDDEN_FRAC = 0.03
# LOCO's smallest boxes: the 0.1st percentile of box area / image area. Boxes
# below it are labels no human annotator drew. Curation drops them, so this
# check only watches for the share changing.
LOCO_AREA_FLOOR = 0.000063
MAX_BELOW_FLOOR_FRAC = 0.20
# The share of boxes that survives curation (dropping hidden, below-floor,
# duplicate and full-frame boxes). Below this, filtering cannot fix the run.
MIN_CURATION_YIELD = 0.85

# Box outlines on the contact sheets. Brighter than the mask colours, which
# are close to the colour of a pale floor.
SHEET_COLOURS = {"pallet": (80, 200, 255), "pallet_truck": (255, 200, 40), "forklift": (255, 90, 90)}


# --- reading a run ---------------------------------------------------------

def regime_dirs(out_dir: Path) -> dict[str, Path]:
    """Regime name -> the directory holding its `rgb/`, `object_detection/`
    and `semantic_segmentation/` folders."""
    return {d.parent.name: d.parent for d in sorted(Path(out_dir).glob("*/*/object_detection"))}


def frames(cam_dir: Path) -> list[str]:
    """Frame stems in numeric order."""
    return sorted((p.stem for p in (cam_dir / "rgb").glob("*.png")), key=int)


def read_labels(path: Path) -> list[tuple[str, float, float, float, float]]:
    """(class, x1, y1, x2, y2) per row of a KITTI label file."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 8:
            rows.append((parts[0], *(float(v) for v in parts[4:8])))
    return rows


def _fill(mask: np.ndarray | None, cls: str, x1, y1, x2, y2) -> float | None:
    if mask is None or cls not in SEG_COLOURS:
        return None
    h, w = mask.shape[:2]
    xa, ya, xb, yb = max(0, int(x1)), max(0, int(y1)), min(w, round(x2)), min(h, round(y2))
    if xb <= xa or yb <= ya:
        return None
    crop = mask[ya:yb, xa:xb]
    return float((crop == SEG_COLOURS[cls]).all(axis=-1).mean())


def regime_stats(cam_dir: Path) -> dict:
    """Per-frame box counts and per-box measurements for one regime."""
    stems = frames(cam_dir)
    width, height = Image.open(cam_dir / "rgb" / f"{stems[0]}.png").size if stems else (1, 1)
    image_area = float(width * height)
    per_frame, boxes = [], []
    for stem in stems:
        label_file = cam_dir / "object_detection" / f"{stem}.txt"
        rows = read_labels(label_file) if label_file.exists() else []
        mask_file = cam_dir / "semantic_segmentation" / f"{stem}.png"
        mask = np.array(Image.open(mask_file).convert("RGBA")) if mask_file.exists() else None
        per_frame.append(len(rows))
        seen = set()
        for cls, x1, y1, x2, y2 in rows:
            area = (x2 - x1) * (y2 - y1) / image_area
            boxes.append({
                "cls": cls,
                "height_px": y2 - y1,
                "area": area,
                "fill": _fill(mask, cls, x1, y1, x2, y2),
                # One object carrying two labels is written as two rows with
                # the same geometry.
                "duplicate": (x1, y1, x2, y2) in seen,
                "full_frame": area >= FULL_FRAME_FRAC,
            })
            seen.add((x1, y1, x2, y2))
    return {"per_frame": per_frame, "boxes": boxes}


# --- judging it ------------------------------------------------------------

def _hidden(box: dict) -> bool:
    return box["fill"] is not None and box["fill"] < MIN_FILL


def _curated_out(box: dict) -> bool:
    return _hidden(box) or box["area"] < LOCO_AREA_FLOOR or box["duplicate"] or box["full_frame"]


def _share(items, predicate) -> float:
    items = list(items)
    return sum(map(predicate, items)) / len(items) if items else 0.0


def checks(stats: dict) -> list[tuple[str, bool, str]]:
    """(check name, passed, measured value) for one regime."""
    per_frame, boxes = stats["per_frame"], stats["boxes"]
    if not boxes:
        return [("any boxes at all", False, f"0 boxes in {len(per_frame)} frames")]
    classes = Counter(b["cls"] for b in boxes)
    p50 = statistics.median(per_frame)
    empty = _share(per_frame, lambda n: n == 0)
    small = _share(boxes, lambda b: b["height_px"] < SMALL_BOX_PX)
    full = sum(b["full_frame"] for b in boxes)
    dups = sum(b["duplicate"] for b in boxes)
    hidden = _share(boxes, _hidden)
    below = _share(boxes, lambda b: b["area"] < LOCO_AREA_FLOOR)
    kept = 1.0 - _share(boxes, _curated_out)
    lo, hi = DENSITY_BAND
    return [
        ("label vocabulary", set(classes) <= set(TARGET_CLASSES), str(dict(classes))),
        (f"median boxes/frame in [{lo}, {hi}]", lo <= p50 <= hi, f"{p50:g}"),
        (f"empty frames <= {MAX_EMPTY_FRAC:.0%}", empty <= MAX_EMPTY_FRAC, f"{empty:.0%}"),
        (f"boxes under {SMALL_BOX_PX} px >= {MIN_SMALL_BOX_FRAC:.0%}",
         small >= MIN_SMALL_BOX_FRAC, f"{small:.0%}"),
        (f"full-frame boxes <= {MAX_FULL_FRAME_FRAC:.1%}",
         full <= MAX_FULL_FRAME_FRAC * len(boxes), f"{full}/{len(boxes)}"),
        ("no duplicate-geometry rows", dups == 0, f"{dups}/{len(boxes)}"),
        (f"mostly-hidden boxes <= {MAX_HIDDEN_FRAC:.0%}", hidden <= MAX_HIDDEN_FRAC,
         f"{hidden:.1%} under {MIN_FILL:.0%} fill"),
        (f"boxes below LOCO's area floor <= {MAX_BELOW_FLOOR_FRAC:.0%}",
         below <= MAX_BELOW_FLOOR_FRAC, f"{below:.1%}"),
        (f"curation yield >= {MIN_CURATION_YIELD:.0%}", kept >= MIN_CURATION_YIELD,
         f"{kept:.1%}"),
    ]


def summary_lines(stats: dict) -> list[str]:
    """Informational numbers, not judged."""
    boxes, per_frame = stats["boxes"], sorted(stats["per_frame"])
    if not boxes:
        return []
    fills = [b["fill"] for b in boxes if b["fill"] is not None]
    pct = " ".join(f"p{q}={per_frame[min(len(per_frame) - 1, q * len(per_frame) // 100)]}"
                   for q in (0, 10, 25, 50, 75, 90, 100))
    lines = [f"median box area {statistics.median(b['area'] for b in boxes):.2%} of frame (LOCO 0.22%)",
             f"boxes/frame {pct}"]
    if fills:
        lines.append(f"median fill {statistics.median(fills):.2f}")
    return lines


def contact_sheet(cam_dir: Path, dest: Path, cols: int = 5, width: int = 384,
                  max_frames: int = 30) -> Path | None:
    """A grid of up to `max_frames` evenly spaced frames, boxes drawn on."""
    stems = frames(cam_dir)
    if not stems:
        return None
    n = min(max_frames, len(stems))
    stems = [stems[i * len(stems) // n] for i in range(n)]
    cols = min(cols, n)
    thumbs = []
    for stem in stems:
        image = Image.open(cam_dir / "rgb" / f"{stem}.png").convert("RGB")
        scale = width / image.width
        image = image.resize((width, round(image.height * scale)))
        draw = ImageDraw.Draw(image)
        label_file = cam_dir / "object_detection" / f"{stem}.txt"
        for cls, x1, y1, x2, y2 in read_labels(label_file) if label_file.exists() else []:
            draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                           outline=SHEET_COLOURS.get(cls, (160, 160, 160)))
        thumbs.append(image)
    tw, th = thumbs[0].size
    sheet = Image.new("RGB", (cols * tw, -(-len(thumbs) // cols) * th), (18, 18, 18))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * tw, (i // cols) * th))
    sheet.save(dest)
    return dest


def run_gate(out_dir: Path) -> tuple[bool, str]:
    """Judge every regime in `out_dir` and write its contact sheet. Returns
    whether all checks passed, and the report."""
    out_dir = Path(out_dir)
    lines, failures = [], []
    regimes = regime_dirs(out_dir)
    if not regimes:
        failures.append(f"no regimes found under {out_dir}")
    for regime, cam_dir in regimes.items():
        stats = regime_stats(cam_dir)
        lines.append(f"\n=== {regime} ===")
        for name, ok, detail in checks(stats):
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            if not ok:
                failures.append(f"{regime}: {name} ({detail})")
        lines += [f"   .... {s}" for s in summary_lines(stats)]
        sheet = contact_sheet(cam_dir, out_dir / f"contact_{regime}.jpg")
        if sheet:
            lines.append(f"   .... contact sheet {sheet.name}")
    lines.append("")
    if failures:
        lines.append(f"GATE FAILED ({len(failures)}):")
        lines += [f"  - {f}" for f in failures]
    else:
        lines.append("GATE PASSED")
    return not failures, "\n".join(lines).lstrip("\n") + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    passed, report = run_gate(ap.parse_args().out_dir)
    print(report, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
