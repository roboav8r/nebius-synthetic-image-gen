"""Can this job render? Host checks, then the warehouse in each anti-aliasing mode.

Run by `sdg.run` for a config with `"smoke_test": true`. It checks the GPU's
graphics capability and Vulkan driver, the shared cache directory, network
access to NVIDIA's assets and to our bucket, and then renders one view per
anti-aliasing mode. It uploads those frames and `result.txt` to
`<upload.prefix>/smoke_test/<job name>/`, so the modes can be compared by eye.

The render runs in a child process (`python -m sdg.smoke_test --render <dir>`):
Kit ends its process on close, so everything after it has to live elsewhere.

`edge` is the mean absolute difference between horizontally adjacent pixels.
A clean frame scores about 1-2; the speckled frames Isaac Sim 4.5 produced on
Blackwell scored 14-19.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from sdg import isaac, storage
from sdg.scene import RESOLUTION, WAREHOUSE_USD

ROOT = Path(__file__).resolve().parents[1]
ISAAC_ASSETS_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1"
AA_MODES = {
    "dlss_performance": {"/rtx/post/aa/op": 3, "/rtx/post/dlss/execMode": 0},  # Kit's default
    "dlss_quality": {"/rtx/post/aa/op": 3, "/rtx/post/dlss/execMode": 2},
    "dlaa": {"/rtx/post/aa/op": 4},
    "no_aa": {"/rtx/post/aa/op": 0},
}
MIN_PIXEL_STD = 1.0  # below this the frame is blank


def render(out_dir: Path) -> None:
    """Under Isaac's interpreter: write `<mode>.png` for each AA mode."""
    from isaacsim import SimulationApp

    app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})
    import carb
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.storage.native import get_assets_root_path

    omni.usd.get_context().open_stage(get_assets_root_path() + WAREHOUSE_USD)
    for _ in range(60):
        app.update()
    settings = carb.settings.get_settings()
    camera = rep.create.camera(position=(-8, 4, 1.45), look_at=(-8, -4, 0.3), focal_length=16.0)
    render_product = rep.create.render_product(camera, RESOLUTION)
    for mode, values in AA_MODES.items():
        for key, value in values.items():
            settings.set(key, value)
        frames_dir = out_dir / ".frames" / mode
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(frames_dir), rgb=True)
        writer.attach([render_product])
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=8)
        rep.orchestrator.wait_until_complete()
        writer.detach()
        pngs = sorted(glob.glob(f"{frames_dir}/**/*.png", recursive=True))
        if pngs:
            shutil.copy(pngs[-1], out_dir / f"{mode}.png")
        print(f"[smoke_test] rendered {mode}: {len(pngs)} frame(s)", flush=True)
    app.close()


def scores(png: Path) -> tuple[float, float]:
    """(pixel std, edge score) of a frame, in greyscale."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(png).convert("L"), float)
    return float(a.std()), float(np.abs(np.diff(a, axis=1)).mean())


def host_checks(cfg: dict) -> list[tuple[bool, str]]:
    checks = []
    caps = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "")
    checks.append(("all" in caps or "graphics" in caps,
                   f"driver capabilities include graphics ({caps or 'unset'})"))
    icds = glob.glob("/usr/share/vulkan/icd.d/nvidia*.json") + glob.glob("/etc/vulkan/icd.d/nvidia*.json")
    checks.append((bool(icds), f"NVIDIA Vulkan ICD present ({', '.join(icds) or 'none'})"))
    cache_root = os.environ.get("SDG_CACHE_ROOT")
    if cache_root:
        try:
            Path(cache_root).mkdir(parents=True, exist_ok=True)
            (Path(cache_root) / f".smoke-test-{os.getpid()}").touch()
            (Path(cache_root) / f".smoke-test-{os.getpid()}").unlink()
            checks.append((True, f"cache directory writable ({cache_root})"))
        except OSError as e:
            checks.append((False, f"cache directory writable ({cache_root}): {e}"))
    try:
        request = urllib.request.Request(ISAAC_ASSETS_URL + WAREHOUSE_USD, method="HEAD")
        status = urllib.request.urlopen(request, timeout=20).status
        checks.append((status == 200, f"HTTPS to NVIDIA's asset bucket: HTTP {status}"))
    except OSError as e:
        checks.append((False, f"HTTPS to NVIDIA's asset bucket: {e}"))
    # A secret without these entry names still starts the job; the bucket then just says 403.
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        checks.append((bool(os.environ.get(key)),
                       f"{key} received from the MysteryBox secret (the entry must have this name)"))
    try:
        storage.client(cfg).head_bucket(Bucket=cfg["upload"]["bucket"])
        checks.append((True, f"bucket {cfg['upload']['bucket']} reachable"))
    except Exception as e:  # noqa: BLE001
        checks.append((False, f"bucket {cfg['upload']['bucket']} reachable: {e}"))
    return checks


def main(cfg: dict, work_dir: Path) -> int:
    """Run every check, upload the frames, and return 0 if all passed."""
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"], check=False)
    checks = host_checks(cfg)
    if os.environ.get("SDG_CACHE_ROOT"):
        isaac.link_caches(Path(os.environ["SDG_CACHE_ROOT"]))

    out_dir = work_dir / "smoke_test"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    print("==> rendering each anti-aliasing mode (cold start about 6 min, warm about 1)", flush=True)
    log = out_dir / "render.log"
    with open(log, "w") as f:
        subprocess.run(isaac.python() + ["-m", "sdg.smoke_test", "--render", str(out_dir)],
                       stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
    shutil.rmtree(out_dir / ".frames", ignore_errors=True)

    lines = []
    for mode in AA_MODES:
        png = out_dir / f"{mode}.png"
        if png.exists():
            std, edge = scores(png)
            checks.append((std > MIN_PIXEL_STD, f"{mode} rendered (pixel std {std:.1f})"))
            lines.append(f"{mode:17} std {std:5.1f}  edge {edge:5.1f}")
        else:
            checks.append((False, f"{mode} rendered"))
    if not all(ok for ok, _ in checks[-len(AA_MODES):]):
        print("==> last lines of the render log:")
        print("".join(log.read_text(errors="ignore").splitlines(True)[-40:]))

    report = [f"[{'PASS' if ok else 'FAIL'}] {name}" for ok, name in checks] + [""] + lines
    passed = all(ok for ok, _ in checks)
    report.append("SMOKE TEST PASSED" if passed else "SMOKE TEST FAILED")
    (out_dir / "result.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report), flush=True)

    prefix = f"{storage.run_prefix(cfg)}/{os.environ.get('SDG_JOB_NAME', 'local')}"
    try:
        storage.upload_dir(cfg, out_dir, prefix)
    except Exception as e:  # noqa: BLE001
        print(f"upload failed: {e}", file=sys.stderr)
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render", type=Path, required=True, metavar="OUT_DIR")
    render(ap.parse_args().render)
