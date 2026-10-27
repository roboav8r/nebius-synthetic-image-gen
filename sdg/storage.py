"""Nebius Object Storage, through its S3 API.

Credentials come from `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, which
boto3 reads from the environment itself: from `.env` locally, and from the
MysteryBox secret in a job. The bucket, endpoint and region come from the
config's `upload` block.

A run is complete only once its `_SUCCESS` marker exists. `upload_dir` writes
the marker last, so a partial upload can never pass for a complete one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MARKER = "_SUCCESS"
_WORKERS = 16


def client(cfg: dict):
    import boto3

    up = cfg["upload"]
    return boto3.client("s3", endpoint_url=up["endpoint_url"], region_name=up["region"])


def run_prefix(cfg: dict) -> str:
    """Where a run lives in the bucket: `<upload.prefix>/<run_name>`."""
    return f"{cfg['upload']['prefix'].strip('/')}/{cfg['run_name']}"


def url(cfg: dict, prefix: str) -> str:
    return f"s3://{cfg['upload']['bucket']}/{prefix}"


def list_keys(cfg: dict, prefix: str) -> list[str]:
    """Every key under `prefix/`, relative to it."""
    pages = client(cfg).get_paginator("list_objects_v2").paginate(
        Bucket=cfg["upload"]["bucket"], Prefix=prefix + "/")
    return [obj["Key"][len(prefix) + 1:] for page in pages for obj in page.get("Contents", [])]


def is_done(cfg: dict, prefix: str) -> bool:
    return MARKER in list_keys(cfg, prefix)


def download(cfg: dict, prefix: str, dest: Path, top_level_only: bool = False) -> Path:
    """Copy everything under `prefix/` into `dest`, keeping the layout.

    With `top_level_only`, skip subdirectories: that fetches a run's summary,
    gate report and contact sheets without its frames.
    """
    keys = list_keys(cfg, prefix)
    if top_level_only:
        keys = [k for k in keys if "/" not in k]
    if not keys:
        raise FileNotFoundError(f"nothing at {url(cfg, prefix)}/")
    s3, bucket, dest = client(cfg), cfg["upload"]["bucket"], Path(dest)

    def get(key: str) -> None:
        (dest / key).parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, f"{prefix}/{key}", str(dest / key))

    with ThreadPoolExecutor(_WORKERS) as pool:
        list(pool.map(get, keys))
    print(f"==> downloaded {len(keys)} files from {url(cfg, prefix)}/ to {dest}")
    return dest


def upload_dir(cfg: dict, src: Path, prefix: str, marker: str | None = None) -> None:
    """Copy every file under `src` to `prefix/`. If `marker` is given, write
    it to `prefix/_SUCCESS` after everything else has landed."""
    s3, bucket = client(cfg), cfg["upload"]["bucket"]
    files = sorted(p for p in Path(src).rglob("*") if p.is_file())
    print(f"==> uploading {len(files)} files to {url(cfg, prefix)}/")

    def put(path: Path) -> None:
        s3.upload_file(str(path), bucket, f"{prefix}/{path.relative_to(src).as_posix()}")

    with ThreadPoolExecutor(_WORKERS) as pool:
        list(pool.map(put, files))
    if marker is not None:
        s3.put_object(Bucket=bucket, Key=f"{prefix}/{MARKER}", Body=marker.encode())
        print(f"==> wrote {url(cfg, prefix)}/{MARKER}")
