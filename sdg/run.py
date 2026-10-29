"""The job's entrypoint: one run, from config to bucket.

    /isaac-sim/python.sh -m sdg.run --config <config.json>

Steps: refuse early if the run is already in the bucket, link Isaac's caches,
generate, check the frame count, run the gate, upload. `_SUCCESS` is written
last and is the only sign a run is complete. A config with `"probe": true`
runs `sdg.probe` instead.

Exit codes, so that a job ending in FAILED always means something went wrong:
    0  the gate's result was the one expected (`expect_gate`, default "pass")
    1  the gate's result was not the one expected, or the probe failed
    2  the generator wrote no frames (it aborts early on a failed scene check)
    3  the generator wrote fewer frames than asked for
    4  the upload was refused or failed

Environment:
    SDG_WORK_DIR      scratch for run output (default: runs/work)
    SDG_CACHE_ROOT    if set, Isaac's caches are linked under it
    SDG_ISAAC_PYTHON  interpreter for the generator (see `isaac.python`)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sdg import isaac, probe, storage
from sdg.config import load_config
from sdg.gate import frames, regime_dirs, run_gate
from sdg.scene import REGIMES

ROOT = Path(__file__).resolve().parents[1]

# Generator log lines that bracket the render loop, for timing.
_LOOP_START = re.compile(r"^\[sdg\] rendering ")
_LOOP_END = re.compile(r"^\[sdg\] placed ")


def generate(cmd: list[str], log_path: Path) -> dict:
    """Run the generator, keeping its full log but echoing only its own
    `[sdg]` lines (Kit logs thousands). Returns the exit code and timings."""
    t0 = time.time()
    t_loop = t_end = None
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=ROOT)
        for line in proc.stdout:
            log.write(line)
            if line.startswith("[sdg]"):
                sys.stdout.write(line)
            if t_loop is None and _LOOP_START.match(line):
                t_loop = time.time()
            elif _LOOP_END.match(line):
                t_end = time.time()
        returncode = proc.wait()
    if returncode != 0 or t_end is None:
        print(f"==> generator exit {returncode}; last 40 lines of {log_path.name}:")
        sys.stdout.writelines(log_path.read_text(errors="ignore").splitlines(True)[-40:])
    return {
        "returncode": returncode,
        "wall_s": round(time.time() - t0, 1),
        "startup_s": round(t_loop - t0, 1) if t_loop else None,
        "render_s": round(t_end - t_loop, 1) if t_loop and t_end else None,
    }


def run(cfg: dict, work_dir: Path) -> int:
    gen, name, upload = cfg["generator"], cfg["run_name"], cfg["upload"]
    prefix = storage.run_prefix(cfg)
    out_dir = work_dir / name
    print(f"==> {name}: {cfg.get('description', '')}")
    print(f"    {gen['num_frames']} frames x {len(REGIMES)} regimes, output {out_dir}")

    if upload["enabled"]:
        try:
            if storage.is_done(cfg, prefix) and not upload.get("overwrite"):
                print(f"{storage.url(cfg, prefix)} is already complete; "
                      "set upload.overwrite to replace it", file=sys.stderr)
                return 4
        except Exception as e:  # noqa: BLE001
            print(f"cannot reach the bucket: {e}", file=sys.stderr)
            return 4

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    generator_config = work_dir / f"{name}.generator.json"
    generator_config.write_text(json.dumps(gen, indent=1))
    if os.environ.get("SDG_CACHE_ROOT"):
        isaac.link_caches(Path(os.environ["SDG_CACHE_ROOT"]))
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"], check=False)

    timing = generate(isaac.python() + ["-m", "sdg.generate", "--config", str(generator_config),
                                        "--output-dir", str(out_dir)],
                      out_dir / "run.log")
    n_frames = sum(len(frames(d)) for d in regime_dirs(out_dir).values())
    expected = gen["num_frames"] * len(REGIMES)
    summary = {"run_name": name, "config": cfg, "frames": n_frames, "expected_frames": expected,
               **timing}
    print(f"==> {n_frames}/{expected} frames, startup {timing['startup_s']} s, "
          f"render {timing['render_s']} s")

    def finish(code: int) -> int:
        summary["exit_code"] = code
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1))
        return code

    if n_frames == 0:
        return finish(2)
    if n_frames != expected:
        return finish(3)

    passed, report = run_gate(out_dir)
    print(report, end="")
    (out_dir / "validation.txt").write_text(report)
    summary["gate"] = "pass" if passed else "fail"
    summary["gate_expected"] = cfg.get("expect_gate", "pass")
    code = 0 if summary["gate"] == summary["gate_expected"] else 1
    print(f"==> gate {summary['gate']} (expected {summary['gate_expected']})")

    if upload["enabled"]:
        summary["uploaded_to"] = storage.url(cfg, prefix)
        finish(code)
        try:
            storage.upload_dir(cfg, out_dir, prefix, marker=json.dumps(summary, indent=1))
        except Exception as e:  # noqa: BLE001
            print(f"upload failed: {e}", file=sys.stderr)
            del summary["uploaded_to"]
            return finish(4)
    return finish(code)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    cfg = load_config(ap.parse_args().config)
    work_dir = Path(os.environ.get("SDG_WORK_DIR", ROOT / "runs" / "work")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("probe"):
        return probe.main(cfg, work_dir)
    return run(cfg, work_dir)


if __name__ == "__main__":
    sys.exit(main())
