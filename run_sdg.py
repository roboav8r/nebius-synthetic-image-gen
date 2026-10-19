#!/usr/bin/env python
"""One entry point for every SDG run: naive, audited and full.

    python run_sdg.py --config configs/naive_20.json
    python run_sdg.py --config configs/full_2500.json --dry-run

The config file is the experiment: generator overrides, frames, subframes,
seed, whether to validate, where to upload. The environment is the runtime:
which Isaac interpreter, where caches and scratch live, and the storage keys.
That split is what lets the notebook and the Nebius job run the same command
with the same config and get the same run.

Steps: link caches -> generate -> frame-count guard -> validate -> upload,
with `_SUCCESS` written last. The generator and the gate are called through
their existing CLIs and are not modified.

Configs hold no account details: strings like "${SDG_BUCKET}" (or
"${VAR:-default}") are filled from the environment, which locally comes from
`.env` and in a Nebius job from the launcher's `--env` flags. An unset
variable becomes "<unset:VAR>", which the upload checks refuse.

Runtime env (all optional):
    SDG_ISAAC_PYTHON   interpreter for the generator. Default: /isaac-sim/python.sh
                       if it exists (container), else this interpreter (pip venv)
    SDG_WORK_DIR       local scratch for run output         (default: ./runs)
    SDG_CACHE_ROOT     overrides cache.root from the config

Exit codes: 0 gate result as expected (`expect_gate`, default "pass"), 1 gate result
not as expected, 2 generator failed, 3 frame count short, 4 upload refused or failed.
So the naive run, whose gate is meant to fail, exits 0, and a job that ends in
FAILED always means something went wrong.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "sdg" / "generate_replicator.py"
VALIDATOR = HERE / "sdg" / "check_validation.py"

# The 8 paths `sdg/run_sdg_local_reference.sh` mounts. Together they take a
# cold start from 336 s to 19.5 s. Container layout; the pip install keeps
# its kit cache inside site-packages and is left alone.
CACHE_PATHS = {
    "kit": "/isaac-sim/kit/cache",
    "ov": "/root/.cache/ov",
    "pip": "/root/.cache/pip",
    "glcache": "/root/.cache/nvidia/GLCache",
    "computecache": "/root/.nv/ComputeCache",
    "logs": "/root/.nvidia-omniverse/logs",
    "data": "/root/.local/share/ov/data",
    "documents": "/root/Documents",
}

# Generator log lines that bracket the render loop, used for timing.
RE_LOOP_START = re.compile(r"^\[sdg\] \d+ frames x 2 regimes")
RE_LOOP_END = re.compile(r"^\[sdg\] placed ")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


RE_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(v):
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    if isinstance(v, str):
        return RE_VAR.sub(lambda m: os.environ.get(m[1]) or (
            m[2] if m[2] is not None else f"<unset:{m[1]}>"), v)
    return v


def _load(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(_load((path.parent / parent).resolve()), cfg)
    return cfg


def load_config(path) -> dict:
    """Read a config, resolving `extends` (relative to the file) recursively,
    then filling `${VAR}` placeholders from the environment."""
    return _expand(_load(Path(path).resolve()))


def isaac_python() -> list[str]:
    if os.environ.get("SDG_ISAAC_PYTHON"):
        return [os.environ["SDG_ISAAC_PYTHON"]]
    if Path("/isaac-sim/python.sh").exists():
        return ["/isaac-sim/python.sh"]
    return [sys.executable]


def s3_client(cfg: dict):
    """boto3 client for the config's bucket. Keys are read from the env vars
    the config names; their values are never printed or written anywhere."""
    import boto3

    up = cfg["upload"]
    return boto3.client(
        "s3",
        endpoint_url=up["endpoint_url"],
        region_name=up["region"],
        aws_access_key_id=os.environ[up["access_key_env"]],
        aws_secret_access_key=os.environ[up["secret_key_env"]],
    )


def remote_prefix(cfg: dict) -> str:
    return f"{cfg['upload']['prefix'].strip('/')}/{cfg['run_name']}"


def upload_problems(cfg: dict) -> list[str]:
    up = cfg["upload"]
    probs = [f"upload.{k} is still a placeholder"
             for k in ("bucket", "endpoint_url", "region") if "<" in str(up.get(k, "<"))]
    probs += [f"env var {up[k]} is not set"
              for k in ("access_key_env", "secret_key_env") if not os.environ.get(up[k])]
    return probs


def link_caches(root: Path) -> None:
    for name, target in CACHE_PATHS.items():
        src, dst = root / name, Path(target)
        src.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            dst.rename(dst.with_name(dst.name + ".orig"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=True)
    print(f"==> caches linked under {root}")


def generate(cmd: list[str], log_path: Path, env: dict, quiet: bool = False) -> dict:
    """Run the generator, tee its log, and time startup vs the render loop.
    `quiet` echoes only the generator's own `[sdg]` lines (Kit logs thousands),
    plus the log's tail if it fails; the full log is always written."""
    t0 = time.time()
    t_loop = t_end = None
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, cwd=HERE)
        for line in proc.stdout:
            if not quiet or line.startswith("[sdg]"):
                sys.stdout.write(line)
            log.write(line)
            if t_loop is None and RE_LOOP_START.match(line):
                t_loop = time.time()
            elif RE_LOOP_END.match(line):
                t_end = time.time()
        rc = proc.wait()
    if quiet and (rc != 0 or t_end is None):
        print(f"==> generator exit {rc}; last 40 lines of {log_path}:")
        sys.stdout.writelines(log_path.read_text(errors="ignore").splitlines(True)[-40:])
    wall = time.time() - t0
    return {
        "returncode": rc,
        "wall_s": round(wall, 1),
        "startup_s": round(t_loop - t0, 1) if t_loop else None,
        "render_s": round(t_end - t_loop, 1) if t_loop and t_end else None,
    }


def check_not_done(cfg: dict) -> None:
    """Never overwrite a completed run unless the config says so. Nothing is
    ever deleted remotely. Checked before generating, so a rerun of the long
    job fails in seconds rather than after an hour of GPU time."""
    bucket, prefix = cfg["upload"]["bucket"], remote_prefix(cfg)
    done = s3_client(cfg).list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/_SUCCESS", MaxKeys=1)
    if done.get("KeyCount") and not cfg["upload"].get("overwrite", False):
        raise RuntimeError(f"s3://{bucket}/{prefix}/_SUCCESS already exists; "
                           "set upload.overwrite to replace")


def upload(cfg: dict, out_dir: Path, summary: dict) -> str:
    check_not_done(cfg)
    client = s3_client(cfg)
    bucket, prefix = cfg["upload"]["bucket"], remote_prefix(cfg)
    dest = f"s3://{bucket}/{prefix}"

    files = [p for p in sorted(out_dir.rglob("*")) if p.is_file()]
    print(f"==> uploading {len(files)} files to {dest}")

    def put(p: Path):
        client.upload_file(str(p), bucket, f"{prefix}/{p.relative_to(out_dir).as_posix()}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(put, files))

    # Last, so a partial upload can never be mistaken for a complete one.
    client.put_object(Bucket=bucket, Key=f"{prefix}/_SUCCESS",
                      Body=json.dumps(summary, indent=1).encode())
    print(f"==> uploaded; _SUCCESS written at {dest}/_SUCCESS")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved plan and exit without running anything")
    ap.add_argument("--quiet", action="store_true",
                    help="echo only the generator's [sdg] lines; the full log is still kept")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gen = cfg["generator"]
    run_name = cfg["run_name"]
    work = Path(os.environ.get("SDG_WORK_DIR", HERE / "runs")).resolve()
    out_dir = work / run_name
    log_path = work / f"{run_name}.log"
    overrides_path = work / f"{run_name}.generator.json"
    cache_root = os.environ.get("SDG_CACHE_ROOT") or cfg.get("cache", {}).get("root")

    cmd = isaac_python() + [
        str(GENERATOR), "--config", str(overrides_path),
        "--num-frames", str(gen["num_frames"]), "--rt-subframes", str(gen["rt_subframes"]),
        "--seed", str(gen["seed"]), "--output-dir", str(out_dir),
    ]

    print(f"==> run {run_name}: {cfg.get('description', '')}")
    print(f"    {gen['num_frames']} steps x 2 regimes, rt_subframes={gen['rt_subframes']}, "
          f"seed={gen['seed']}, overrides={gen.get('overrides') or 'none'}")
    print(f"    output {out_dir}")
    print(f"    caches {cache_root or 'none (cold start)'}")
    if cfg["upload"]["enabled"]:
        print(f"    upload s3://{cfg['upload']['bucket']}/{remote_prefix(cfg)}")
        for p in upload_problems(cfg):
            print(f"    !! {p}")
    else:
        print("    upload disabled")

    if args.dry_run:
        print("==> dry run; generator command:\n    " + " ".join(cmd))
        return 0

    if cfg["upload"]["enabled"]:
        if upload_problems(cfg):
            print("upload is enabled but not configured; fix the above first", file=sys.stderr)
            return 4
        try:
            check_not_done(cfg)  # also proves the endpoint and keys work
        except Exception as e:  # noqa: BLE001
            print(f"upload preflight failed: {e}", file=sys.stderr)
            return 4

    work.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # local scratch only
    out_dir.mkdir(parents=True)
    overrides_path.write_text(json.dumps(gen.get("overrides", {}), indent=1))
    if cache_root:
        link_caches(Path(cache_root))

    env = dict(os.environ)
    if cfg.get("accept_nvidia_eula"):
        env.update(ACCEPT_EULA="Y", PRIVACY_CONSENT="Y", OMNI_KIT_ACCEPT_EULA="YES")

    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"], check=False)
    timing = generate(cmd, log_path, env, quiet=args.quiet)

    n_rgb = len(list(out_dir.glob("*/*/rgb/*.png")))
    expected = 2 * gen["num_frames"]
    steps = gen["num_frames"]
    summary = {
        "run_name": run_name,
        "config": cfg,
        "isaac_python": cmd[0],
        "rgb_frames": n_rgb,
        "expected_frames": expected,
        **timing,
        "s_per_step": round(timing["render_s"] / steps, 2) if timing["render_s"] else None,
    }
    print(f"==> generator exit {timing['returncode']}, wall {timing['wall_s']}s "
          f"(startup {timing['startup_s']}s, render {timing['render_s']}s, "
          f"{summary['s_per_step']} s/step), {n_rgb}/{expected} rgb frames")

    def finish(code: int) -> int:
        summary["exit_code"] = code
        shutil.copy(log_path, out_dir / "run.log")
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1))
        return code

    if n_rgb == 0:
        print("no frames; read the log -- an early abort is usually a content gate", file=sys.stderr)
        return finish(2)
    if n_rgb != expected:
        print("frame count short; not uploading", file=sys.stderr)
        return finish(3)

    code = 0
    if cfg.get("validate", True):
        res = subprocess.run([sys.executable, str(VALIDATOR), "--out-dir", str(out_dir),
                              "--log", str(log_path)],
                             capture_output=True, text=True, cwd=HERE)
        print(res.stdout, end="")
        print(res.stderr, end="", file=sys.stderr)
        (out_dir / "validation.txt").write_text(res.stdout + res.stderr)
        summary["gate"] = "pass" if res.returncode == 0 else "fail"
        summary["gate_expected"] = cfg.get("expect_gate", "pass")
        code = 0 if summary["gate"] == summary["gate_expected"] else 1
        print(f"==> gate {summary['gate']} (expected {summary['gate_expected']})")
    finish(code)

    if cfg["upload"]["enabled"]:
        # Recorded before uploading so the copy in the bucket carries it too.
        summary["uploaded_to"] = f"s3://{cfg['upload']['bucket']}/{remote_prefix(cfg)}"
        finish(code)
        try:
            upload(cfg, out_dir, summary)
        except Exception as e:  # noqa: BLE001
            print(f"upload failed: {e}", file=sys.stderr)
            summary.pop("uploaded_to")
            return finish(4)
    return finish(code)


if __name__ == "__main__":
    raise SystemExit(main())
