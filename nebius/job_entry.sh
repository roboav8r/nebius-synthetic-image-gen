#!/usr/bin/env bash
# The entrypoint of the job image (see Dockerfile): NVIDIA's Isaac Sim 6.1 image
# with this repo's code at /opt/sdg. `nebius_job.py` injects the configs at
# /workspace/configs and passes one argument:
#
#   <config.json>   # one SDG run via run_sdg.py
#   --probe         # can this job render at all, and which anti-aliasing mode is clean?
#
# Env from the launcher: SDG_BUCKET, SDG_S3_ENDPOINT, SDG_S3_REGION, SDG_WORK_DIR,
# SDG_CACHE_ROOT, SDG_CONFIG_DIR, SDG_JOB_NAME; the two S3 keys arrive as
# MysteryBox secrets.
set -euo pipefail
export ACCEPT_EULA=Y PRIVACY_CONSENT=Y
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # /opt/sdg in the image
PY="${SDG_ISAAC_PYTHON:-/isaac-sim/python.sh}"

# boto3 for the upload, Pillow for the gate's masks and contact sheets. The
# 6.1 base image ships all three, so nothing is installed.
"${PY}" -c "import boto3, numpy, PIL; print('==> deps ok: boto3', boto3.__version__, 'numpy', numpy.__version__, 'Pillow', PIL.__version__)"

if [[ "${1:-}" != "--probe" ]]; then
  exec "${PY}" "${ROOT}/run_sdg.py" --config "${SDG_CONFIG_DIR:-${ROOT}/configs}/${1:?config file name}" --quiet
fi

# ---------------------------------------------------------------- probe ------
fail=0
check() { if [[ "$1" == ok ]]; then echo "[PASS] $2"; else echo "[FAIL] $2"; fail=1; fi; }

echo "==> GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo "==> $(id)"
echo "==> NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-(unset)}"
[[ "${NVIDIA_DRIVER_CAPABILITIES:-}" =~ (all|graphics) ]] && c=ok || c=no
check "$c" "driver capabilities include graphics"
icds=$(ls /usr/share/vulkan/icd.d/nvidia*.json /etc/vulkan/icd.d/nvidia*.json 2>/dev/null || true)
echo "${icds:-no NVIDIA ICD files found}"
[[ -n "${icds}" ]] && c=ok || c=no
check "$c" "NVIDIA Vulkan ICD present"
echo "==> $(ldd --version | head -1)"
probe_file="${SDG_CACHE_ROOT:-/shared/cache-6.1}/.probe-$$"
mkdir -p "$(dirname "${probe_file}")" && touch "${probe_file}" && rm -f "${probe_file}" && c=ok || c=no
check "$c" "shared filesystem writable at ${SDG_CACHE_ROOT:-/shared/cache-6.1}"

"${PY}" - <<'EOF' || fail=1
import os, sys, urllib.request
import boto3
ok = True
url = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
       "Isaac/6.1/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")
try:
    code = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20).status
    print(f"[{'PASS' if code == 200 else 'FAIL'}] HTTPS to NVIDIA asset bucket: HTTP {code}")
    ok &= code == 200
except Exception as e:
    print(f"[FAIL] HTTPS to NVIDIA asset bucket: {e}"); ok = False
try:
    s3 = boto3.client("s3", endpoint_url=os.environ["SDG_S3_ENDPOINT"],
                      region_name=os.environ["SDG_S3_REGION"],
                      aws_access_key_id=os.environ["NEBIUS_S3_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["NEBIUS_S3_SECRET_ACCESS_KEY"])
    s3.head_bucket(Bucket=os.environ["SDG_BUCKET"])
    print(f"[PASS] bucket {os.environ['SDG_BUCKET']} reachable with the job's secret keys")
except Exception as e:
    print(f"[FAIL] bucket access: {e}"); ok = False
sys.exit(0 if ok else 1)
EOF

# Point Isaac's caches at the shared filesystem, so the first probe's cold start
# warms them for later probes and real runs.
"${PY}" -c "import os, sys; from pathlib import Path; sys.path.insert(0, '${ROOT}'); import run_sdg; \
run_sdg.link_caches(Path(os.environ.get('SDG_CACHE_ROOT', '/shared/cache-6.1')))"

# The real test: start Isaac Sim headless and render the warehouse at the
# generator's resolution, once per anti-aliasing mode. Kit's default is DLSS
# Performance, which renders at half resolution; on the 4.5 image that came out
# speckled on this GPU. `edge` is the mean absolute difference between
# horizontally adjacent pixels: about 1-2 for a clean frame here, 14-19 for the
# speckled 4.5 frames.
echo "==> warehouse render per anti-aliasing mode (cold start several minutes, warm well under one)"
SECONDS=0
cat > /tmp/probe_render.py <<'EOF'
from isaacsim import SimulationApp
app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})
import glob, os
import carb
import numpy as np
import omni.replicator.core as rep
from PIL import Image
from isaacsim.storage.native import get_assets_root_path
import omni.usd
omni.usd.get_context().open_stage(get_assets_root_path() + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")
for _ in range(60):
    app.update()
settings = carb.settings.get_settings()
cam = rep.create.camera(position=(-8, 4, 1.45), look_at=(-8, -4, 0.3), focal_length=16.0)
rp = rep.create.render_product(cam, (960, 540))
variants = {
    "dlss_performance": {"/rtx/post/aa/op": 3, "/rtx/post/dlss/execMode": 0},
    "dlss_quality": {"/rtx/post/aa/op": 3, "/rtx/post/dlss/execMode": 2},
    "dlaa": {"/rtx/post/aa/op": 4},
    "no_aa": {"/rtx/post/aa/op": 0},
}
print(f"Kit default: aa/op {settings.get('/rtx/post/aa/op')}, "
      f"dlss/execMode {settings.get('/rtx/post/dlss/execMode')}", flush=True)
lines = []
for name, values in variants.items():
    for k, v in values.items():
        settings.set(k, v)
    out = f"/tmp/probe_out/{name}"
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=out, rgb=True)
    writer.attach([rp])
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=8)
    rep.orchestrator.wait_until_complete()
    writer.detach()
    files = sorted(glob.glob(f"{out}/**/*.png", recursive=True))
    a = np.asarray(Image.open(files[-1]).convert("L"), float) if files else np.zeros((2, 2))
    lines.append(f"{name:17} std {a.std():5.1f}  edge {np.abs(np.diff(a, axis=1)).mean():5.1f}")
    print(lines[-1], flush=True)
    if name == "dlss_performance":
        ok = a.std() > 1
        lines.insert(0, f"[{'PASS' if ok else 'FAIL'}] rendered {len(files)} image(s), pixel std {a.std():.1f}")
with open("/tmp/probe_result", "w") as f:
    f.write("\n".join(lines) + "\n")
app.close()  # ends the process without Python's normal exit; unflushed output is lost
EOF
# Kit's own exit code on close is not reliable, so judge by the result file.
rm -rf /tmp/probe_result /tmp/probe_out
"${PY}" /tmp/probe_render.py > /tmp/probe_render.log 2>&1 || true
echo "==> render step took ${SECONDS}s"
if grep -q '^\[PASS\] rendered' /tmp/probe_result 2>/dev/null; then
  cat /tmp/probe_result
else
  cat /tmp/probe_result 2>/dev/null || true
  echo "[FAIL] one-frame render; last lines of the Kit log:"; tail -40 /tmp/probe_render.log; fail=1
fi

# Upload the frames so the modes can be compared by eye (nu.probe_frames()).
"${PY}" - <<'EOF' || true
import os
from pathlib import Path
import boto3
s3 = boto3.client("s3", endpoint_url=os.environ["SDG_S3_ENDPOINT"],
                  region_name=os.environ["SDG_S3_REGION"],
                  aws_access_key_id=os.environ["NEBIUS_S3_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["NEBIUS_S3_SECRET_ACCESS_KEY"])
prefix = f"probe/{os.environ.get('SDG_JOB_NAME', 'local')}"
for p in sorted(Path("/tmp/probe_out").glob("*/**/*.png")):
    s3.upload_file(str(p), os.environ["SDG_BUCKET"], f"{prefix}/{p.parts[3]}/{p.name}")
s3.upload_file("/tmp/probe_result", os.environ["SDG_BUCKET"], f"{prefix}/result.txt")
print(f"==> frames uploaded to s3://{os.environ['SDG_BUCKET']}/{prefix}/")
EOF

[[ "${fail}" == 0 ]] && echo "PROBE PASSED" || { echo "PROBE FAILED"; exit 1; }
