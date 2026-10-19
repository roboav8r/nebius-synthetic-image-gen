"""Helpers for `sdg_walkthrough.ipynb`: local checks, audit, FiftyOne, bucket I/O.

Runs happen in Nebius jobs (`nebius_job.py`); this module only looks at what
they produced, after pulling it down from the bucket into runs/<run_name>/.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sdg"))
from check_validation import read_labels, regimes  # noqa: E402
from run_sdg import load_config, remote_prefix, s3_client  # noqa: E402

GATE_LINE = re.compile(r"^\s*\[(PASS|FAIL)\]|^===|^GATE")
ENV_NEEDED = ("SDG_BUCKET", "SDG_S3_ENDPOINT", "SDG_S3_REGION",
              "NEBIUS_S3_ACCESS_KEY_ID", "NEBIUS_S3_SECRET_ACCESS_KEY")


# --- 0. local checks -------------------------------------------------------

def local_check() -> bool:
    """Is this machine ready to launch jobs and read the bucket? Secrets are
    checked by name only."""
    checks = []
    cli = shutil.which("nebius")
    checks.append(("nebius CLI on PATH", bool(cli), cli or "see NEBIUS_SETUP.md step 1"))
    if cli:
        r = subprocess.run(["nebius", "ai", "job", "list", "--format", "json"],
                           capture_output=True, text=True)
        checks.append(("CLI profile can list jobs", r.returncode == 0,
                       "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:120]))
    for k in ENV_NEEDED:
        checks.append((f"{k} set", bool(os.environ.get(k)), ""))
    if all(os.environ.get(k) for k in ENV_NEEDED):
        try:
            cfg = load_config(HERE / "configs" / "common.json")
            s3_client(cfg).head_bucket(Bucket=cfg["upload"]["bucket"])
            checks.append(("bucket reachable with .env keys", True, cfg["upload"]["bucket"]))
        except Exception as e:  # noqa: BLE001
            checks.append(("bucket reachable with .env keys", False, str(e)[:120]))
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return all(ok for _, ok, _ in checks)


# --- audit -----------------------------------------------------------------

def run_dir(config: str) -> Path:
    """Where a run's files land locally: runs/<run_name>/, mirroring the bucket."""
    return HERE / "runs" / load_config(config)["run_name"]


def fetch_run(config: str, include_frames: bool = True) -> Path:
    """Download a run from the bucket into `run_dir(config)`. With
    include_frames=False only the top-level files come down (gate output,
    summary, contact sheets, sim_params); use that for the 2,500-step run."""
    cfg = load_config(config)
    client = s3_client(cfg)
    bucket, prefix = cfg["upload"]["bucket"], remote_prefix(cfg)
    dest = run_dir(config)
    keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix + "/") for o in page.get("Contents", [])]
    rel = [k[len(prefix) + 1:] for k in keys]
    if not include_frames:
        rel = [k for k in rel if "/" not in k]
    if not rel:
        raise FileNotFoundError(f"nothing at s3://{bucket}/{prefix}/ -- did the job upload?")

    def get(k):
        (dest / k).parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, f"{prefix}/{k}", str(dest / k))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(get, rel))
    print(f"fetched {len(rel)} files from s3://{bucket}/{prefix}/ to {dest}")
    return dest


def summary(config: str) -> dict:
    """`run_summary.json` minus the echoed config, which the notebook already shows."""
    s = json.loads((run_dir(config) / "run_summary.json").read_text())
    s.pop("config", None)
    return s


def show_audit(path: Path, gate_only: bool = True) -> None:
    """Print the gate's verdict and display both contact sheets."""
    from IPython.display import Image, display

    text = (Path(path) / "validation.txt").read_text()
    for line in text.splitlines():
        if not gate_only or GATE_LINE.match(line) or line.startswith("  - "):
            print(line)
    for sheet in sorted(Path(path).glob("contact_*.jpg")):
        print(f"\n{sheet.stem}")
        display(Image(filename=str(sheet), width=960))


# --- FiftyOne --------------------------------------------------------------

def load_run_into_fiftyone(path: Path, name: str | None = None):
    """A FiftyOne dataset of one run: rgb frames, KITTI boxes as `ground_truth`,
    plus `regime`, `frame` and `n_placed` from `sim_params.json`.

    Each detection carries `height_px`, so "boxes under 25 px" (30.5% of real
    LOCO pallets) is a one-line filter.
    """
    import fiftyone as fo
    from PIL import Image

    path = Path(path)
    name = name or path.name
    frames = {}
    sp = path / "sim_params.json"
    if sp.exists():
        frames = json.loads(sp.read_text()).get("frames", {})

    samples = []
    for regime, cam_dir, det_dir in regimes(path):
        for rgb in sorted((cam_dir / "rgb").glob("*.png"), key=lambda p: int(p.stem)):
            w, h = Image.open(rgb).size
            dets = [
                fo.Detection(label=cls,
                             bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                             height_px=round(y2 - y1, 1))
                for cls, x1, y1, x2, y2 in read_labels(det_dir / f"{rgb.stem}.txt")
            ] if (det_dir / f"{rgb.stem}.txt").exists() else []
            samples.append(fo.Sample(
                filepath=str(rgb),
                tags=[name, regime],
                run=name,
                regime=regime,
                frame=int(rgb.stem),
                n_placed=frames.get(rgb.stem, {}).get("n_placed"),
                ground_truth=fo.Detections(detections=dets),
            ))

    dataset = fo.Dataset(name, overwrite=True)
    dataset.add_samples(samples)
    dataset.persistent = True
    return dataset


def box_stats(dataset) -> dict:
    """Per regime: the README's headline numbers, computed from FiftyOne."""
    from fiftyone import ViewField as F

    out = {}
    for regime in sorted(dataset.distinct("regime")):
        view = dataset.match(F("regime") == regime)
        counts = view.values(F("ground_truth.detections").length())
        heights = view.values("ground_truth.detections[].height_px")
        out[regime] = {
            "frames": len(counts),
            "p50 boxes/frame": statistics.median(counts) if counts else 0,
            "empty frames": f"{sum(c == 0 for c in counts) / max(1, len(counts)):.0%}",
            "sub-25px boxes": f"{sum(h < 25 for h in heights) / max(1, len(heights)):.0%}",
            "classes": dict(Counter(view.values("ground_truth.detections[].label"))),
        }
    return out


def print_stats(**datasets) -> None:
    """Side-by-side `box_stats` for several runs, e.g. print_stats(naive=a, audited=b)."""
    stats = {run: box_stats(ds) for run, ds in datasets.items()}
    rows = [(run, regime, s) for run, by_regime in stats.items() for regime, s in by_regime.items()]
    keys = ["frames", "p50 boxes/frame", "empty frames", "sub-25px boxes"]
    print(f"{'run':10} {'regime':16} " + " ".join(f"{k:>16}" for k in keys))
    for run, regime, s in rows:
        print(f"{run:10} {regime:16} " + " ".join(f"{str(s[k]):>16}" for k in keys))


# --- bucket ----------------------------------------------------------------

def verify_bucket(config: str, fetch_to: Path | None = None) -> bool:
    """Check a run in the bucket: rgb counts per regime, `_SUCCESS`, the gate,
    and pull back the small files (validation, summary, contact sheets)."""
    cfg = load_config(config)
    client = s3_client(cfg)
    bucket, prefix = cfg["upload"]["bucket"], remote_prefix(cfg)
    expected = cfg["generator"]["num_frames"]

    rgb, keys = Counter(), []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"][len(prefix) + 1:]
            keys.append(key)
            parts = key.split("/")
            if len(parts) == 4 and parts[2] == "rgb" and parts[3].endswith(".png"):
                rgb[parts[0]] += 1

    fetch_to = Path(fetch_to or run_dir(config))
    fetch_to.mkdir(parents=True, exist_ok=True)
    small = [k for k in keys if "/" not in k and k.endswith((".txt", ".json", ".jpg"))]
    for k in small + ["_SUCCESS"]:
        if k in keys:
            client.download_file(bucket, f"{prefix}/{k}", str(fetch_to / k))

    gate = (fetch_to / "validation.txt").read_text() if (fetch_to / "validation.txt").exists() else ""
    checks = [(f"{r}: {n} rgb frames (expected {expected})", n == expected) for r, n in sorted(rgb.items())]
    checks += [
        ("two regimes present", len(rgb) == 2),
        ("_SUCCESS present", "_SUCCESS" in keys),
        ("gate passed", "GATE PASSED" in gate),
    ]
    print(f"s3://{bucket}/{prefix}/  ({len(keys)} objects)")
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"small files fetched to {fetch_to}")
    return all(ok for _, ok in checks)
