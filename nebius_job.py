#!/usr/bin/env python
"""Launch an SDG run as a Nebius serverless job, from a run config.

    python nebius_job.py --config configs/naive_20.json --dry-run   # print the commands
    python nebius_job.py --config configs/naive_20.json --wait      # build, launch and follow

    python nebius_job.py --config configs/probe.json --build        # only build and push the image

The run config's `job` block is the job spec: base image, registry, platform,
preset, disk, timeout, the MysteryBox secret holding the S3 keys, the shared
filesystem, and an optional subnet. This turns it into one `nebius ai job
create` command.

- **Image:** NVIDIA's digest-pinned Isaac Sim 6.1 image with this repo's code
  copied in (see Dockerfile), pushed to a Nebius Container Registry. It is
  tagged with a hash of the code, so only a code change makes a new tag.
- **Configs:** small, so injected into every job at /workspace/configs. All of
  a job's injected files share a 64 KiB limit; the code would not fit, which is
  why it is in the image.
- **Caches:** Isaac's caches live on the shared filesystem, at /shared/cache-6.1.
  The image runs as root: the base image's non-root user cannot write there.
- **Bucket details:** passed in with `--env`.
- **S3 keys:** referenced by secret name with `--env-secret`, so their values never
  appear on this machine's command line or in the job spec.

Every launch is recorded in runs/jobs.json, so `wait()` and `logs()` can pick a
job back up by run name after a kernel restart. Image pushes are recorded under
`_build`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_sdg import load_config

HERE = Path(__file__).resolve().parent
JOBS_FILE = HERE / "runs" / "jobs.json"
CONTAINER_ROOT = "/workspace"
MAX_INJECT_BYTES = 64 * 1024
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}

# What goes into the image; the Dockerfile copies exactly these. Their hash,
# with the Dockerfile's, is the image tag.
CODE_FILES = [
    "run_sdg.py",
    "constants.py",
    "sdg/generate_replicator.py",
    "sdg/check_validation.py",
    "nebius/job_entry.sh",
]
IMAGE_FILES = CODE_FILES + ["Dockerfile", ".dockerignore"]
IMAGE_NAME = "sdg"
# Bucket details the configs reference as ${VAR}; forwarded to the job as-is.
FORWARD_ENV = ["SDG_BUCKET", "SDG_S3_ENDPOINT", "SDG_S3_REGION"]
JOB_WORK_DIR = "/tmp/sdg_runs"  # on the container disk, not the shared filesystem
SHARED_MOUNT = "/shared"
CACHE_DIR = f"{SHARED_MOUNT}/cache-6.1"  # 4.5's caches are no use to 6.1
BUILD_RUN = "_build"


def config_files() -> list[str]:
    return sorted(p.relative_to(HERE).as_posix() for p in (HERE / "configs").glob("*.json"))


def check_inject_size(paths: list[Path]) -> int:
    """All of a job's injected files are encrypted together, and the total may
    not exceed 64 KiB, so check the sum, not each file."""
    total = sum(p.stat().st_size for p in paths)
    if total > MAX_INJECT_BYTES:
        raise ValueError(f"injected files total {total} bytes; a job allows {MAX_INJECT_BYTES}")
    return total


def code_hash() -> str:
    """Content hash of everything that goes into the image. Configs are not
    included, so changing one does not need a new image."""
    h = hashlib.sha256()
    for f in sorted(IMAGE_FILES):
        h.update(f.encode() + b"\0" + (HERE / f).read_bytes() + b"\0")
    return h.hexdigest()[:12]


def image_tag(cfg: dict) -> str:
    return f"{cfg['job']['registry'].rstrip('/')}/{IMAGE_NAME}:{code_hash()}"


def problems(cfg: dict) -> list[str]:
    """Placeholders still unset, and tools the launch needs locally."""
    out = []
    job, up = cfg["job"], cfg["upload"]
    for k in ("registry", "platform", "preset", "disk_size", "timeout", "secret", "shared_fs"):
        if "<unset:" in str(job.get(k, "<unset:")):
            out.append(f"job.{k} is unset ({job.get(k)})")
    for k in ("bucket", "endpoint_url", "region"):
        if "<unset:" in str(up[k]):
            out.append(f"upload.{k} is unset ({up[k]})")
    if not shutil.which("docker"):
        out.append("docker is not on PATH (needed to build the job image)")
    return out


def _job_spec(name: str, spec: dict, image: str, fs: str) -> list[str]:
    cmd = [
        "nebius", "ai", "job", "create",
        "--name", name,
        "--image", image,
        "--platform", spec["platform"],
        "--preset", spec["preset"],
        "--disk-size", spec["disk_size"],
        "--timeout", spec["timeout"],
        "--restart-policy", "never",
        "--volume", f"{fs}:{SHARED_MOUNT}:rw",
    ]
    if spec.get("subnet_id"):
        cmd += ["--subnet-id", spec["subnet_id"]]
    return cmd


def build_command(config: str, image: str, name: str | None = None) -> list[str]:
    cfg = load_config(config)
    job, up = cfg["job"], cfg["upload"]
    run_name = cfg["run_name"]
    name = name or f"sdg-{run_name.replace('_', '-')}-{datetime.now():%Y%m%d-%H%M%S}"
    entry_arg = "--probe" if cfg.get("probe") else Path(config).name

    # The image's entrypoint is job_entry.sh, so the job only supplies its argument.
    # `=` so the CLI cannot read `--probe` as one of its own flags.
    cmd = _job_spec(name, job, image, job["shared_fs"]) + [f"--args={entry_arg}"]
    configs = config_files()
    check_inject_size([HERE / f for f in configs])
    for f in configs:
        cmd += ["--inject-file", f"{HERE / f}:{CONTAINER_ROOT}/{f}"]

    env = {k: os.environ.get(k, f"<unset:{k}>") for k in FORWARD_ENV}
    env["SDG_WORK_DIR"] = JOB_WORK_DIR
    env["SDG_CACHE_ROOT"] = CACHE_DIR
    env["SDG_CONFIG_DIR"] = f"{CONTAINER_ROOT}/configs"
    env["SDG_JOB_NAME"] = name
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    # The value comes from the secret's payload entry whose key matches the
    # variable name, so one secret holds both keys.
    for k in (up["access_key_env"], up["secret_key_env"]):
        cmd += ["--env-secret", f"{k}={job['secret']}"]
    # Return once the job is submitted; `wait()` follows it from there.
    return cmd + ["--async", "--format", "json"]


def _show(cmd: list[str]) -> str:
    """The command as you'd type it, with injected paths shortened for reading."""
    return " \\\n  ".join(
        shlex.quote(a.replace(str(HERE) + "/", "")) if a.startswith(str(HERE)) else shlex.quote(a)
        for a in cmd)


# --- job records --------------------------------------------------------------

def _records() -> dict:
    return json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else {}


def _save_record(run_name: str, rec: dict) -> None:
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    recs = _records()
    runs = recs.setdefault(run_name, [])
    for i, r in enumerate(runs):
        if r["id"] == rec["id"]:
            runs[i] = rec
            break
    else:
        runs.append(rec)
    JOBS_FILE.write_text(json.dumps(recs, indent=1))


def last_job(run_name: str) -> dict:
    """The most recent launch for a run, e.g. last_job("full_2500")["id"]."""
    runs = _records().get(run_name)
    if not runs:
        raise KeyError(f"no recorded job for {run_name}; see {JOBS_FILE}")
    return runs[-1]


def _job_id(created: dict) -> str:
    for path in (("metadata", "id"), ("resource_id",), ("id",)):
        v = created
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if v:
            return v
    raise RuntimeError(f"no job id in CLI output: {json.dumps(created)[:500]}")


# --- build / init / launch / wait / logs ----------------------------------------

def _create(cmd: list[str], run_name: str, record: dict) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"nebius ai job create failed ({res.returncode}):\n{res.stderr.strip()}")
    job_id = _job_id(json.loads(res.stdout))
    name = cmd[cmd.index("--name") + 1]
    _save_record(run_name, {"id": job_id, "name": name, "created": _now(),
                            "timeline": {}, "state": "CREATED", **record})
    print(f"==> created job {name} ({job_id})")
    return job_id


def build(config: str, push: bool = True, dry_run: bool = False) -> str:
    """Build the job image and push it to the registry; return its tag. Both
    steps are no-ops when nothing changed: Docker reuses its layer cache, and
    the registry already has the layers. The first push uploads the ~11 GB base
    layers once. Needs `nebius registry configure-helper` to have been run."""
    cfg = load_config(config)
    tag = image_tag(cfg)
    steps = [["docker", "build", "--platform", "linux/amd64", "-t", tag, str(HERE)]]
    if push:
        steps.append(["docker", "push", tag])
    for cmd in steps:
        print("==> " + " ".join(shlex.quote(a) for a in cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)
    if push and not dry_run:
        _save_record(BUILD_RUN, {"id": tag, "name": tag, "created": _now(), "timeline": {},
                                 "state": "PUSHED", "bundle": code_hash()})
    return tag


def launch(config: str, dry_run: bool = False) -> str | None:
    """Build and push the image if the code changed, then create the job and
    return its ID. With dry_run, only print the commands."""
    cfg = load_config(config)
    print(f"==> {cfg['run_name']}: {cfg.get('description', '')}")
    missing = problems(cfg) + [f"env {k} is not set" for k in FORWARD_ENV if not os.environ.get(k)]
    for p in missing:
        print(f"!! {p}")
    if missing and not dry_run:
        raise RuntimeError("fix the problems above before launching")
    tag = build(config, dry_run=dry_run)
    cmd = build_command(config, tag)
    print(_show(cmd))
    if dry_run:
        print("==> dry run; nothing launched")
        return None
    return _create(cmd, cfg["run_name"], {"config": config, "image": tag})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state(job_id: str) -> str:
    res = subprocess.run(["nebius", "ai", "job", "get", job_id, "--format", "json"],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout).get("status", {}).get("state", "UNKNOWN")


def wait(job_id: str | None = None, run_name: str | None = None, poll_s: int = 15) -> dict:
    """Poll until the job finishes, recording when each state was first seen.
    Pass a job ID, or a run name to resume the last recorded launch."""
    rec = last_job(run_name) if run_name else next(
        (r for runs in _records().values() for r in runs if r["id"] == job_id), None)
    job_id = job_id or rec["id"]
    timeline = dict(rec["timeline"]) if rec else {}
    t0 = time.time()
    while True:
        s = state(job_id)
        if s not in timeline:
            timeline[s] = _now()
            print(f"  {datetime.now():%H:%M:%S}  {s}  (+{time.time() - t0:.0f}s)", flush=True)
            if rec:
                rec.update(timeline=timeline, state=s)
                _save_record(_run_of(job_id), rec)
        if s in TERMINAL:
            break
        time.sleep(poll_s)
    return {"id": job_id, "state": s, "timeline": timeline, "durations_s": durations(timeline)}


def _run_of(job_id: str) -> str:
    return next(run for run, recs in _records().items() if any(r["id"] == job_id for r in recs))


def durations(timeline: dict) -> dict:
    """Seconds spent in each state, from the first time each was seen. Only as
    fine as the polling interval."""
    seen = sorted(timeline.items(), key=lambda kv: kv[1])
    ts = [datetime.fromisoformat(t) for _, t in seen]
    return {s: (ts[i + 1] - ts[i]).total_seconds() for i, (s, _) in enumerate(seen[:-1])}


def logs(job_id: str, tail: int | None = None, follow: bool = False, since: str = "72h") -> None:
    # The CLI only looks back 1h by default, which hides older jobs' logs.
    cmd = ["nebius", "ai", "job", "logs", job_id, "--since", since]
    if tail:
        cmd += ["--tail", str(tail)]
    if follow:
        cmd += ["--follow"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        for line in p.stdout:
            print(line, end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--wait", action="store_true", help="wait for the job, then print its logs")
    ap.add_argument("--build", action="store_true", help="only build and push the job image")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except ImportError:
        pass
    if args.build:
        build(args.config, dry_run=args.dry_run)
        return 0
    job_id = launch(args.config, dry_run=args.dry_run)
    if job_id and args.wait:
        result = wait(job_id)
        logs(job_id, tail=200)
        return 0 if result["state"] == "COMPLETED" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
