# CLAUDE.md

Synthetic warehouse images (pallets, pallet trucks, forklifts) generated with Isaac Sim Replicator as Nebius serverless jobs, uploaded to Nebius Object Storage, then audited locally in FiftyOne. One-time cloud setup is in `docs/NEBIUS_SETUP.md`; the notebook `nebius_synthetic_image_gen.ipynb` is the driver.

## Layout
- `configs/*.json`: one file per experiment. `extends` deep-merges with `common.json`; `${VAR}` / `${VAR:-default}` come from `.env`. An unset var becomes `<unset:VAR>`, which `nebius_job.problems()` and the upload checks refuse.
- `run_sdg.py`: runs inside the job. Links caches, runs the generator, guards the frame count, runs the gate (`sdg/check_validation.py`), then uploads. It writes `_SUCCESS` last, and that is the only completion marker; `uploaded_to` in `run_summary.json` is recorded before the upload starts.
- `sdg/generate_replicator.py`: the Replicator scene. It runs under Isaac's own Python, so it cannot import project modules; constants are duplicated in its `CONFIG`.
- `nebius_job.py` (imported as `nj` in the notebook): `build()` builds and pushes the image, `launch()` creates the job, and `wait()` / `logs()` follow it. Every job is recorded in `runs/jobs.json`.
- `nebius/job_entry.sh`: the image's entrypoint. Takes `<config.json>` or `--probe`.
- `notebook_utils.py` (`nu`): local checks, `fetch_run`, `probe_frames`, FiftyOne loading.
- `runs/` is gitignored: job records, downloaded runs, logs (`runs/logs/`).

## Environments
- Project Python: `~/miniforge3/envs/nebius-image-sdg/bin/python` (3.10). `python` is not on PATH.
- Local Isaac Sim pip install: `~/miniforge3/envs/isaacsim` is **5.1**, not the job's 6.1. Compare rendering only against the 6.1 container, not this env.
- The job image: `Dockerfile` = `nvcr.io/nvidia/isaac-sim:6.1.0@sha256:af1d2b4e…` plus the code at `/opt/sdg`, tagged `$SDG_REGISTRY/sdg:<hash of IMAGE_FILES>`. Configs are injected at `/workspace/configs`, so changing a config needs no rebuild.

## Testing locally before spending GPU time
Run the real image with Docker (the NVIDIA runtime is installed); a cold Isaac start is about 6 min.
```bash
docker build --platform linux/amd64 -t sdg-local:test .
docker run --rm --gpus all --user root -v $PWD/runs/logs/shared:/shared \
  -e SDG_CACHE_ROOT=/shared/cache-6.1 sdg-local:test --probe
```
- To iterate on the generator without rebuilding, mount the repo at `/opt/sdg:ro` and run `/isaac-sim/python.sh /opt/sdg/sdg/generate_replicator.py --config <overrides.json> --num-frames 2 --output-dir <dir>`.
- For warm starts, mount host directories for `/isaac-sim/kit/cache`, `/isaac-sim/.cache` and `/isaac-sim/.nv`. Mount the whole `.cache` directory: mounting a subdirectory makes Docker create `.cache` as root, and Warp then fails.

## Lessons (each cost a failed job)

**Nebius platform**
- **Injected files:** all `--inject-file` contents of one job are encrypted together and must total ≤ 64 KiB. That is why the code ships in the image.
- **The shared filesystem:** a non-root container user cannot write to it, even to a directory chowned to that user. The image therefore runs as **root** (`USER root` in the Dockerfile), which also makes `HOME=/root`.
- **Logs:** `nebius ai job logs` only looks back 1 h by default. `nj.logs()` passes `--since 72h`.
- **CLI login:** it expires, and the next CLI call (including the `docker push` credential helper) then waits for a browser sign-in. A background task can hang there silently. Run `nebius iam whoami` before long background work.
- **Capacity:** `gpu-rtx6000` can be unavailable for 20+ min (`NotEnoughResources` / "VM schedule timeout"). The job ends in ERROR without ever running, and nothing is billed.
- **Hardware:** the job GPU is an RTX PRO 6000 Blackwell on driver 580.173.02. 6.1 is tested by NVIDIA on 595, but works on 580. H100/H200/B200 have no RT cores and cannot render.

**Isaac Sim / Kit**
- **The 4.5 image** rendered speckled, streaky frames on Blackwell. 6.1 renders clean frames in every anti-aliasing mode.
- **Anti-aliasing:** Kit's default is DLSS Performance, which renders at half resolution. `common.json` sets DLAA (`rtx_settings: {"/rtx/post/aa/op": 4}`), which is sharper.
- **Process exit:** `SimulationApp.close()` ends the process without Python's normal exit, so buffered stdout is lost and `SystemExit` codes after it are ignored (a scale-check abort exits 0). Print with `flush=True` and write results to a file before `close()`.
- **Units (6.1):** the metrics assembler adds `xformOp:scale:unitsResolve` to centimetre DigitalTwin assets, but inconsistently, on top of our own `unit_scale()`. Forklifts came out 0.02 m long. The generator sets `/metricsAssembler/operationMode = 0`, so `unit_scale()` alone owns units.
- **Semantics (6.1):** `add_update_semantics`, `get_semantics` and `remove_all_semantics` are gone. Use `add_labels` / `get_labels` / `remove_labels`, calling `upgrade_prim_semantics_to_labels` first. They live in `isaacsim.core.utils.semantics`, now under `extsDeprecated`.
- **Label filtering (Kit 109+):** annotators created without an explicit filter get `semanticFilter="*:*"`, which overrides `KittiWriter`'s `semantic_filter_predicate`, so `box`, `rack`, `floor` and `wall` were labelled again. The generator recreates KittiWriter's box and segmentation annotators with the predicate passed explicitly.
- **Camera aiming (6.1):** `rep.create.camera` makes an `_Xform` that follows the USD camera convention (looks down −Z, +Y up), so `_look_at_quat` builds rows (right, up, −forward). 4.5 needed (−forward, right, up); on 6.1 that basis aimed every camera at the floor. Check a contact sheet after any Isaac upgrade: the gate's density checks only show the result, not the cause.
- **Unchanged in 6.1:** `KittiWriter` defaults are the same as 4.5 (`bbox_height_threshold=25`, occlusion thresholds 0.5 / 0.95); see `constants.py`.
- **Asset paths:** they resolve under `…/Assets/Isaac/6.1/`; `DigitalTwin/` and `simready_content/` are version-independent siblings of `Isaac/`.

**Diagnosing render quality**
- Score a frame by its mean absolute horizontal pixel difference. Clean frames score about 1–2; the speckled 4.5 frames scored 14–19. The probe prints this for each anti-aliasing mode and uploads the frames to `s3://<bucket>/probe/<job-name>/`, and `nu.probe_frames(name)` shows them.
- Always change one variable at a time. A clean local render on a different Isaac version proves nothing about the job's GPU.
