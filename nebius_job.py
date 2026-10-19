#!/usr/bin/env python
"""Launch an SDG run as a Nebius serverless job, from a run config.

    python nebius_job.py --config configs/naive_20.json --dry-run   # print the command
    python nebius_job.py --config configs/naive_20.json --wait      # launch and follow

The run config's `job` block is the job spec: image, platform, preset, disk,
timeout, the MysteryBox secret holding the S3 keys, and optional subnet and
cache volume. This turns it into one `nebius ai job create` command.

- **Image:** the stock, digest-pinned NVIDIA Isaac Sim image. Nothing is built or pushed.
- **Code:** this bundle's files are injected at /workspace with `--inject-file`.
- **Bucket details:** passed in with `--env`.
- **S3 keys:** referenced by secret name with `--env-secret`, so their values never
  appear on this machine's command line or in the job spec.

Every launch is recorded in runs/jobs.json, so `wait()` and `logs()` can pick a
job back up by run name after a kernel restart.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
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

# Everything the job needs, at the same relative paths inside the container.
CODE_FILES = [
    "run_sdg.py",
    "constants.py",
    "sdg/generate_replicator.py",
    "sdg/check_validation.py",
    "nebius/job_entry.sh",
]
# Bucket details the configs reference as ${VAR}; forwarded to the job as-is.
FORWARD_ENV = ["SDG_BUCKET", "SDG_S3_ENDPOINT", "SDG_S3_REGION"]
JOB_WORK_DIR = "/root/sdg_runs"  # on the container disk, not the injected tree
CACHE_MOUNT = "/cache"


def inject_files() -> list[str]:
    files = CODE_FILES + sorted(p.relative_to(HERE).as_posix() for p in (HERE / "configs").glob("*.json"))
    for f in files:
        size = (HERE / f).stat().st_size
        if size > MAX_INJECT_BYTES:
            raise ValueError(f"{f} is {size} bytes; --inject-file allows {MAX_INJECT_BYTES}")
    return files


def problems(cfg: dict) -> list[str]:
    """Placeholders still unset, and env the launch needs locally."""
    out = []
    job, up = cfg["job"], cfg["upload"]
    for k in ("image", "platform", "preset", "disk_size", "timeout", "secret"):
        if "<unset:" in str(job.get(k, "<unset:")):
            out.append(f"job.{k} is unset ({job.get(k)})")
    for k in ("bucket", "endpoint_url", "region"):
        if "<unset:" in str(up[k]):
            out.append(f"upload.{k} is unset ({up[k]})")
    return out


def build_command(config: str, name: str | None = None) -> list[str]:
    cfg = load_config(config)
    job, up = cfg["job"], cfg["upload"]
    run_name = cfg["run_name"]
    name = name or f"sdg-{run_name.replace('_', '-')}-{datetime.now():%Y%m%d-%H%M%S}"
    entry_arg = "--probe" if cfg.get("probe") else Path(config).name

    cmd = [
        "nebius", "ai", "job", "create",
        "--name", name,
        "--image", job["image"],
        "--platform", job["platform"],
        "--preset", job["preset"],
        "--disk-size", job["disk_size"],
        "--timeout", job["timeout"],
        "--restart-policy", "never",
        "--container-command", "bash",
        "--args", f"{CONTAINER_ROOT}/nebius/job_entry.sh {entry_arg}",
        "--working-dir", CONTAINER_ROOT,
    ]
    for f in inject_files():
        cmd += ["--inject-file", f"{HERE / f}:{CONTAINER_ROOT}/{f}"]

    env = {k: os.environ.get(k, f"<unset:{k}>") for k in FORWARD_ENV}
    env["SDG_WORK_DIR"] = JOB_WORK_DIR
    if job.get("cache_volume"):
        env["SDG_CACHE_ROOT"] = CACHE_MOUNT
        cmd += ["--volume", f"{job['cache_volume']}:{CACHE_MOUNT}:rw"]
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    # The value comes from the secret's payload entry whose key matches the
    # variable name, so one secret holds both keys.
    for k in (up["access_key_env"], up["secret_key_env"]):
        cmd += ["--env-secret", f"{k}={job['secret']}"]
    if job.get("subnet_id"):
        cmd += ["--subnet-id", job["subnet_id"]]
    cmd += ["--format", "json"]
    return cmd


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


# --- launch / wait / logs -------------------------------------------------------

def launch(config: str, dry_run: bool = False) -> str | None:
    """Create the job and return its ID. With dry_run, only print the command."""
    cfg = load_config(config)
    cmd = build_command(config)
    print(f"==> {cfg['run_name']}: {cfg.get('description', '')}")
    print(_show(cmd))
    missing = problems(cfg) + [f"env {k} is not set" for k in FORWARD_ENV if not os.environ.get(k)]
    for p in missing:
        print(f"!! {p}")
    if dry_run:
        print("==> dry run; nothing launched")
        return None
    if missing:
        raise RuntimeError("fix the problems above before launching")

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"nebius ai job create failed ({res.returncode}):\n{res.stderr.strip()}")
    job_id = _job_id(json.loads(res.stdout))
    name = cmd[cmd.index("--name") + 1]
    _save_record(cfg["run_name"], {"id": job_id, "name": name, "config": config,
                                   "created": _now(), "timeline": {}, "state": "CREATED"})
    print(f"==> created job {name} ({job_id})")
    return job_id


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


def logs(job_id: str, tail: int | None = None, follow: bool = False) -> None:
    cmd = ["nebius", "ai", "job", "logs", job_id]
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
    ap.add_argument("--dry-run", action="store_true", help="print the command, launch nothing")
    ap.add_argument("--wait", action="store_true", help="wait for the job, then print its logs")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except ImportError:
        pass
    job_id = launch(args.config, dry_run=args.dry_run)
    if job_id and args.wait:
        result = wait(job_id)
        logs(job_id, tail=200)
        return 0 if result["state"] == "COMPLETED" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
