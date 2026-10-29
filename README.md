# nebius-synthetic-image-gen

Synthetic warehouse images of pallets, pallet trucks and forklifts, rendered using Isaac Sim Replicator via Nebius serverless GPU jobs, then uploaded to Nebius Object Storage.

To isolate the effect of camera perspective, every frame is rendered from two camera regimes at once: an elevated, CCTV-style camera (`elevated_fixed`) and a handheld camera at eye height (`ground_handheld`).

## Requirements

- Linux with Docker and the NVIDIA container runtime. Specifically, this repo was developed on Ubuntu 24.04.
- A Python environment manager. The commands below use `mamba`.
- A Nebius AI Cloud project with GPU quota. [`docs/NEBIUS_SETUP.md`](docs/NEBIUS_SETUP.md) walks through the one-time setup: CLI, bucket, keys, secret, registry and shared filesystem.

## Quickstart

```bash
git clone https://github.com/roboav8r/nebius-synthetic-image-gen && cd nebius-synthetic-image-gen
mamba env create -f environment.yml && mamba activate nebius-image-sdg
cp .env.example .env    # then follow docs/NEBIUS_SETUP.md
```

Once the setup is complete, you can kick off jobs via the `nebius_synthetic_image_gen.ipynb` notebook, or the Python script:

```bash
python nebius_job.py --check                                   # Verify the setup will work
python nebius_job.py --config configs/probe.json --wait        # Can a job render?
python nebius_job.py --config configs/naive_20.json --wait     # Perform a 20-frame run
```

## How a run works

1. **Build.** `nebius_job.py` builds the image (Isaac Sim 6.1 plus the `sdg/` package) and pushes it to your Container Registry. The tag is a hash of the code, so this is quick when nothing changed.
2. **Launch.** It resolves the run config with your `.env`, injects it into the job, mounts the shared filesystem for Isaac's caches, and passes the S3 keys as MysteryBox secrets. Each launch is recorded in `runs/jobs.json`.
3. **In the job** (`sdg/run.py`):
   1. Refuse early if the run is already complete in the bucket, unless the config allows overwriting.
   2. Link Isaac's caches to the shared filesystem. A warm start takes about 80 s instead of 6 min.
   3. Generate the frames (`sdg/generate.py`).
   4. Check the frame count, then run the quality gate (`sdg/gate.py`).
   5. Upload the run, and write `_SUCCESS` last. That marker is the only sign a run is complete.
4. **Audit.** Download the run and look at the gate's report, the contact sheets, and the frames.

The job's exit code says whether the run went as expected, so a job ending in FAILED always means something went wrong. See the docstring of `sdg/run.py` for the codes.

## Layout

| Path | What it is |
|---|---|
| `nebius_job.py` | Runs on your machine: local checks, image build, job launch, wait and logs. |
| `nebius_synthetic_image_gen.ipynb` | The driver notebook: check, probe, run, audit. |
| `configs/` | One JSON file per experiment. |
| `sdg/` | Everything that runs in the job, copied whole into the image. |
| `sdg/run.py` | The job's entrypoint. |
| `sdg/generate.py` | The Replicator scene. Runs under Isaac Sim's own interpreter. |
| `sdg/gate.py` | The quality gate, also runnable on its own: `python -m sdg.gate <run dir>`. |
| `sdg/probe.py` | The probe job: host checks and a render per anti-aliasing mode. |
| `sdg/config.py`, `sdg/storage.py` | Config loading, and the bucket. Shared by the job and your machine. |
| `sdg/scene.py`, `sdg/isaac.py` | Shared constants (classes, colours, regimes), and Isaac's interpreter and caches. |
| `tests/` | Tests that need neither Isaac nor the cloud: `pytest`. |
| `docs/` | Setup, and the lessons learned along the way. |
| `runs/` | Gitignored: job records, resolved configs, downloaded runs. |

## Run configs

A config inherits from a parent (e.g. `"extends": "common.json"`), then overrides any differences. The parent and config are deep-merged. Account detail strings like `"${SDG_BUCKET}"` or `"${SDG_SUBNET_ID:-}"` stay in `.env`. Required & unset variables result in errors.

| Key | Meaning |
|---|---|
| `run_name` | The run's name in the bucket and in `runs/`. |
| `description` | Printed at launch. |
| `probe` | `true` runs the probe instead of the generator. |
| `expect_gate` | `"pass"` (default) or `"fail"`. A run that is meant to fail the gate exits 0 when it does. |
| `generator` | Passed to the generator: `num_frames`, `rt_subframes`, `seed`, `rtx_settings`, and any key in `DEFAULTS` in `sdg/generate.py`, such as the pool sizes `n_pallet` and `n_forklift`. |
| `upload` | `enabled`, `bucket`, `endpoint_url`, `region`, `prefix`, and `overwrite`, which allows replacing a completed run. |
| `job` | The Nebius job: `registry`, `platform`, `preset`, `disk_size`, `timeout`, `secret`, `shared_fs`, `subnet_id`. |

## What a run produces

A run lands at `s3://<bucket>/<upload.prefix>/<run_name>/` with the following layout:

```
<run_name>/
├── _SUCCESS                    written last; holds a copy of run_summary.json
├── run_summary.json            config, frame count, timings, gate result
├── validation.txt              the gate's report
├── contact_<regime>.jpg        frames with boxes drawn on, one sheet per regime
├── sim_params.json             per frame: each camera's pose and aim point, objects placed
├── run.log                     the generator's full log
└── <regime>/<regime>/          elevated_fixed and ground_handheld
    ├── rgb/<frame>.png
    ├── object_detection/<frame>.txt     KITTI labels
    ├── semantic_segmentation/<frame>.png
    ├── instance_segmentation/<frame>.png
    └── depth/<frame>.png
```

In the KITTI label files only the class, the occlusion flag (column 3) and the 2D box (columns 5–8, in pixels) are populated. The writer fills the other columns with zeros. The segmentation masks use the class colors in `sdg/scene.py`.

## Testing locally

The real image runs locally with Docker. A cold Isaac start takes ~5 minutes on a modern gaming PC.

```bash
docker build --platform linux/amd64 -t sdg-local:test .
python -c "from dotenv import load_dotenv; load_dotenv(); import json; from sdg.config import load_config; \
  c = load_config('configs/naive_20.json'); c['upload']['enabled'] = False; c['generator']['num_frames'] = 2; \
  open('runs/local.json', 'w').write(json.dumps(c))"
docker run --rm --gpus all -v $PWD/runs:/runs -e SDG_WORK_DIR=/runs/work sdg-local:test --config /runs/local.json
```

- To iterate on the code without rebuilding, add `-v $PWD/sdg:/opt/sdg/sdg:ro`.
- For warm starts, mount host directories at `/isaac-sim/kit/cache`, `/root/.cache` and `/root/.nv`. Mount the whole `.cache` directory: mounting a subdirectory makes Docker create `.cache` as root-owned, and Warp then fails.
- Compare rendering only against this container, not a local pip install of a different Isaac Sim version.

## Troubleshooting

[`docs/LESSONS.md`](docs/LESSONS.md) lists every failure met so far, with its cause and fix. The most common:

- **The job ends in ERROR without running.** The GPU platform is out of capacity. Nothing is billed; try again later.
- **A CLI call hangs.** The Nebius login expired and is waiting for a browser sign-in. Run `nebius iam whoami`.
- **The upload is refused.** The run is already complete in the bucket. Set `"upload": {"overwrite": true}` in the config, or change `run_name`.
- **Frames look wrong after an Isaac upgrade.** Check the contact sheets first. The camera aiming and the label filter both changed between 4.5 and 6.1.
