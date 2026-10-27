# CLAUDE.md

Synthetic warehouse images (pallets, pallet trucks, forklifts) rendered with Isaac Sim 6.1 Replicator as Nebius serverless jobs, gated, uploaded to Nebius Object Storage, then audited locally. `README.md` covers the run flow, configs and output layout; `docs/NEBIUS_SETUP.md` the one-time cloud setup; `docs/LESSONS.md` every failure so far. Read `docs/LESSONS.md` before changing the generator, the image or the launcher.

## Layout
- `nebius_job.py` (`nj` in the notebook) runs locally: `check()`, `build()`, `launch()`, `wait()`, `logs()`, `last_job()`. It resolves the config with `.env` and injects it at `/workspace/config.json`. Jobs are recorded in `runs/jobs.json`.
- `sdg/` is everything in the job image; the image tag hashes it. Its modules run as `python -m sdg.<name>` from the repo root, or from `/opt/sdg` in the image.
  - `run.py` is the entrypoint. `generate.py` and the render half of `probe.py` run under Isaac's interpreter, so they may import only `sdg.scene` from this package.
  - `_SUCCESS` is written last and is the only completion marker.
- `configs/*.json`: one per experiment, `extends` deep-merges with `common.json`, `${VAR}` comes from `.env`. An unset var without a default raises.
- `runs/` is gitignored.

## Environments
- Project Python: `~/miniforge3/envs/nebius-image-sdg/bin/python` (3.10). `python` is not on PATH. Tests: `.../bin/python -m pytest`.
- `~/miniforge3/envs/isaacsim` is a pip install of Isaac Sim **5.1**, not the job's 6.1. Compare rendering only against the 6.1 container.
- S3 credentials are `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: in `.env` locally, from the MysteryBox secret in a job.

## Before spending GPU time
Run the change in the real image locally; see "Testing locally" in `README.md`. A cold Isaac start is about 6 min. After any change to camera, lights or labels, look at a contact sheet: the gate's density checks show the result, not the cause.
