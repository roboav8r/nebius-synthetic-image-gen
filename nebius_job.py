#!/usr/bin/env python
"""Build the job image and launch a run config as a Nebius serverless job.

    python nebius_job.py --check                                  # is this machine ready?
    python nebius_job.py --config configs/naive_20.json --dry-run # print what would run
    python nebius_job.py --config configs/naive_20.json --wait    # build, launch, follow

In a notebook: `import nebius_job as nj`, then `nj.check()`, `nj.build(...)`,
`nj.launch(...)`, `nj.wait(...)`, `nj.logs(...)`.

How a job is put together:
- Image: the Dockerfile, which is Isaac Sim 6.1 plus the `sdg/` package,
  tagged with a hash of those files, so only a code change makes a new tag.
- Config: resolved here, where `.env` is, and injected as one file at
  /workspace/config.json. Injected files are limited to 64 KiB per job.
- Caches: a shared filesystem mounted at /shared keeps Isaac's caches warm
  between jobs.
- S3 keys: read in the job from a MysteryBox secret with `--env-secret`, so
  their values never appear on a command line or in the job spec.

Every launch is recorded in runs/jobs.json, so `last_job(run_name)` can pick
a job back up after a kernel restart.
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

from sdg import storage
from sdg.config import load_config

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
JOBS_FILE = RUNS / "jobs.json"
IMAGE_NAME = "sdg"
SHARED_MOUNT = "/shared"
CACHE_ROOT = f"{SHARED_MOUNT}/cache-6.1"  # per Isaac version; 4.5's caches are no use to 6.1
CONFIG_IN_JOB = "/workspace/config.json"
MAX_INJECT_BYTES = 64 * 1024
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}
# What .env must define; see .env.example.
REQUIRED_ENV = ("SDG_BUCKET", "SDG_S3_ENDPOINT", "SDG_S3_REGION", "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY", "SDG_REGISTRY", "SDG_SHARED_FS")


# --- image -----------------------------------------------------------------

def image_files() -> list[Path]:
    """Everything that goes into the image, which is also what its tag hashes."""
    code = sorted(p for p in (ROOT / "sdg").rglob("*.py"))
    return [ROOT / "Dockerfile", ROOT / ".dockerignore", *code]


def image_tag(cfg: dict) -> str:
    h = hashlib.sha256()
    for path in image_files():
        h.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return f"{cfg['job']['registry'].rstrip('/')}/{IMAGE_NAME}:{h.hexdigest()[:12]}"


def build(config: str | Path, push: bool = True, dry_run: bool = False) -> str:
    """Build the image and push it; return its tag. Both steps are quick when
    nothing changed. The first push uploads the ~11 GB Isaac base layers."""
    tag = image_tag(load_config(config))
    steps = [["docker", "build", "--platform", "linux/amd64", "-t", tag, str(ROOT)]]
    if push:
        steps.append(["docker", "push", tag])
    for cmd in steps:
        print("==> " + shlex.join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)
    return tag


# --- job -------------------------------------------------------------------

def create_command(cfg: dict, image: str, name: str, config_file: Path) -> list[str]:
    """The `nebius ai job create` command for a resolved config."""
    job = cfg["job"]
    cmd = [
        "nebius", "ai", "job", "create",
        "--name", name,
        "--image", image,
        "--platform", job["platform"],
        "--preset", job["preset"],
        "--disk-size", job["disk_size"],
        "--timeout", job["timeout"],
        "--restart-policy", "never",
        "--volume", f"{job['shared_fs']}:{SHARED_MOUNT}:rw",
        "--inject-file", f"{config_file}:{CONFIG_IN_JOB}",
        "--env", f"SDG_CACHE_ROOT={CACHE_ROOT}",
        "--env", f"SDG_JOB_NAME={name}",
        # The secret's payload entry named like the variable supplies its value.
        "--env-secret", f"AWS_ACCESS_KEY_ID={job['secret']}",
        "--env-secret", f"AWS_SECRET_ACCESS_KEY={job['secret']}",
        # `=` so the CLI does not read `--config` as one of its own flags.
        f"--args=--config {CONFIG_IN_JOB}",
        "--async",
        "--format", "jsonpath={.resource_id}",
    ]
    if job.get("subnet_id"):
        cmd += ["--subnet-id", job["subnet_id"]]
    return cmd


def _show(cmd: list[str]) -> str:
    """The command as you would type it, one flag and its value per line."""
    lines, i = [shlex.join(cmd[:4])], 4
    while i < len(cmd):
        takes_value = "=" not in cmd[i] and i + 1 < len(cmd) and not cmd[i + 1].startswith("--")
        lines.append(shlex.join(cmd[i:i + 1 + takes_value]))
        i += 1 + takes_value
    return " \\\n  ".join(lines)


def launch(config: str | Path, dry_run: bool = False) -> str | None:
    """Build and push the image if the code changed, then create the job.
    Returns the job ID, or None for a dry run."""
    cfg = load_config(config)
    print(f"==> {cfg['run_name']}: {cfg.get('description', '')}")
    name = f"sdg-{cfg['run_name'].replace('_', '-')}-{datetime.now():%Y%m%d-%H%M%S}"
    config_file = RUNS / "configs" / f"{name}.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(cfg, indent=1))
    if config_file.stat().st_size > MAX_INJECT_BYTES:
        raise ValueError(f"{config_file} is over the {MAX_INJECT_BYTES} byte limit for injected files")

    tag = build(config, dry_run=dry_run)
    cmd = create_command(cfg, tag, name, config_file)
    print(_show(cmd))
    if dry_run:
        print("==> dry run; nothing launched")
        return None
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"nebius ai job create failed:\n{res.stderr.strip()}")
    job_id = res.stdout.strip()
    if not job_id.startswith("aijob-"):
        raise RuntimeError(f"expected a job ID from the CLI, got: {res.stdout[:300]}")
    _save_record(cfg["run_name"], {"id": job_id, "name": name, "created": _now(), "config": str(config),
                                   "image": tag, "state": "CREATED", "timeline": {}})
    print(f"==> created job {name} ({job_id})")
    return job_id


def state(job_id: str) -> str:
    res = subprocess.run(["nebius", "ai", "job", "get", job_id, "--format", "json"],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout).get("status", {}).get("state", "UNKNOWN")


def wait(job_id: str, poll_s: int = 15) -> str:
    """Poll until the job ends, printing each new state; return the last one."""
    run_name, rec = _find_record(job_id)
    t0 = time.time()
    while True:
        s = state(job_id)
        if s not in rec["timeline"]:
            rec["timeline"][s] = _now()
            rec["state"] = s
            _save_record(run_name, rec)
            print(f"  {datetime.now():%H:%M:%S}  {s}  (+{time.time() - t0:.0f} s)", flush=True)
        if s in TERMINAL_STATES:
            return s
        time.sleep(poll_s)


def logs(job_id: str, tail: int | None = None, follow: bool = False) -> None:
    # The CLI only looks back 1 h by default.
    cmd = ["nebius", "ai", "job", "logs", job_id, "--since", "72h"]
    if tail:
        cmd += ["--tail", str(tail)]
    if follow:
        cmd.append("--follow")
    subprocess.run(cmd, check=False)


# --- job records -----------------------------------------------------------

def _records() -> dict[str, list[dict]]:
    return json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else {}


def _save_record(run_name: str, rec: dict) -> None:
    recs = _records()
    runs = [r for r in recs.get(run_name, []) if r["id"] != rec["id"]]
    recs[run_name] = runs + [rec]
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(recs, indent=1))


def _find_record(job_id: str) -> tuple[str, dict]:
    for run_name, runs in _records().items():
        for rec in runs:
            if rec["id"] == job_id:
                return run_name, rec
    raise KeyError(f"job {job_id} is not in {JOBS_FILE}")


def last_job(run_name: str) -> dict:
    """The most recent launch of a run, e.g. `last_job("smoke_test")["name"]`."""
    runs = _records().get(run_name)
    if not runs:
        raise KeyError(f"no recorded job for {run_name} in {JOBS_FILE}")
    return runs[-1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- local checks ----------------------------------------------------------

def check() -> bool:
    """Is this machine ready to build, launch and read the bucket? Prints one
    PASS/FAIL line per check. Secrets are checked by name only."""
    results = []

    def run_ok(cmd: list[str]) -> tuple[bool, str]:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:120]

    cli = shutil.which("nebius")
    results.append(("nebius CLI on PATH", bool(cli), cli or "see docs/NEBIUS_SETUP.md"))
    if cli:
        results.append(("CLI profile can list jobs", *run_ok(["nebius", "ai", "job", "list"])))
    if shutil.which("docker"):
        results.append(("docker daemon reachable", *run_ok(["docker", "info"])))
    else:
        results.append(("docker daemon reachable", False, "docker is not on PATH"))
    for var in REQUIRED_ENV:
        results.append((f"{var} set", bool(os.environ.get(var)), ""))
    if os.environ.get("SDG_REGISTRY"):
        host = os.environ["SDG_REGISTRY"].split("/")[0]
        docker_cfg = Path.home() / ".docker" / "config.json"
        helpers = json.loads(docker_cfg.read_text()).get("credHelpers", {}) if docker_cfg.exists() else {}
        results.append(("docker can push to the registry", host in helpers,
                        "" if host in helpers else "run: nebius registry configure-helper"))
    if cli and os.environ.get("SDG_SHARED_FS"):
        results.append(("shared filesystem exists", *run_ok(
            ["nebius", "compute", "filesystem", "get", os.environ["SDG_SHARED_FS"]])))
    if all(os.environ.get(v) for v in REQUIRED_ENV):
        cfg = load_config(ROOT / "configs" / "common.json")
        try:
            storage.client(cfg).head_bucket(Bucket=cfg["upload"]["bucket"])
            results.append(("bucket reachable with the .env keys", True, cfg["upload"]["bucket"]))
        except Exception as e:  # noqa: BLE001
            results.append(("bucket reachable with the .env keys", False, str(e)[:120]))
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return all(ok for _, ok, _ in results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    ap.add_argument("--check", action="store_true", help="check this machine is ready, then exit")
    ap.add_argument("--build", action="store_true", help="only build and push the image")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--wait", action="store_true", help="wait for the job, then print its logs")
    args = ap.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if args.check:
        return 0 if check() else 1
    if not args.config:
        ap.error("--config is required")
    if args.build:
        build(args.config, dry_run=args.dry_run)
        return 0
    job_id = launch(args.config, dry_run=args.dry_run)
    if job_id and args.wait:
        final = wait(job_id)
        logs(job_id, tail=200)
        return 0 if final == "COMPLETED" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
