#!/usr/bin/env bash
# Runs inside the stock NVIDIA Isaac Sim 4.5 image as a Nebius serverless job.
# `nebius_job.py` injects this bundle's files under /workspace and calls:
#
#   bash /workspace/nebius/job_entry.sh <config.json>   # one SDG run via run_sdg.py
#   bash /workspace/nebius/job_entry.sh --probe         # can this job render at all?
#
# Env from the launcher: SDG_BUCKET, SDG_S3_ENDPOINT, SDG_S3_REGION, SDG_WORK_DIR,
# optionally SDG_CACHE_ROOT; the two S3 keys arrive as MysteryBox secrets.
set -euo pipefail
export ACCEPT_EULA=Y PRIVACY_CONSENT=Y
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # /workspace in a job
PY="${SDG_ISAAC_PYTHON:-/isaac-sim/python.sh}"

# boto3 for the upload, Pillow for the gate's masks and contact sheets. numpy
# ships with Isaac and is deliberately not reinstalled.
"${PY}" -m pip install -q --no-cache-dir --disable-pip-version-check boto3 Pillow
"${PY}" -c "import boto3, numpy, PIL; print('==> deps ok: boto3', boto3.__version__, 'numpy', numpy.__version__)"

if [[ "${1:-}" != "--probe" ]]; then
  exec "${PY}" "${ROOT}/run_sdg.py" --config "${ROOT}/configs/${1:?config file name}" --quiet
fi

# ---------------------------------------------------------------- probe ------
fail=0
check() { if [[ "$1" == ok ]]; then echo "[PASS] $2"; else echo "[FAIL] $2"; fail=1; fi; }

echo "==> GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo "==> NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-(unset)}"
[[ "${NVIDIA_DRIVER_CAPABILITIES:-}" =~ (all|graphics) ]] && c=ok || c=no
check "$c" "driver capabilities include graphics"
icds=$(ls /usr/share/vulkan/icd.d/nvidia*.json /etc/vulkan/icd.d/nvidia*.json 2>/dev/null || true)
echo "${icds:-no NVIDIA ICD files found}"
[[ -n "${icds}" ]] && c=ok || c=no
check "$c" "NVIDIA Vulkan ICD present"
echo "==> $(ldd --version | head -1)"

"${PY}" - <<'EOF' || fail=1
import os, sys, urllib.request
import boto3
ok = True
url = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
       "Isaac/4.5/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")
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

# The real test: start Isaac Sim headless and render one frame with the same
# renderer the generator uses.
echo "==> one-frame render (cold start, several minutes)"
cat > /tmp/probe_render.py <<'EOF'
from isaacsim import SimulationApp
app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})
import glob
import numpy as np
import omni.replicator.core as rep
from PIL import Image
rep.create.light(light_type="distant", intensity=3000)
rep.create.cube(position=(0, 0, 0), scale=2)
cam = rep.create.camera(position=(6, 6, 4), look_at=(0, 0, 0))
rp = rep.create.render_product(cam, (320, 240))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir="/tmp/probe_out", rgb=True)
writer.attach([rp])
rep.orchestrator.step(rt_subframes=4)
rep.orchestrator.wait_until_complete()
files = sorted(glob.glob("/tmp/probe_out/**/*.png", recursive=True))
std = float(np.asarray(Image.open(files[-1]).convert("L")).std()) if files else 0.0
print(f"[{'PASS' if std > 1 else 'FAIL'}] rendered {len(files)} image(s), pixel std {std:.1f}")
app.close()
EOF
# Kit's own exit code on close is not reliable, so judge by the printed result.
"${PY}" /tmp/probe_render.py > /tmp/probe_render.log 2>&1 || true
if grep -q '^\[PASS\] rendered' /tmp/probe_render.log; then
  grep '^\[PASS\] rendered' /tmp/probe_render.log
else
  echo "[FAIL] one-frame render; last lines of the Kit log:"; tail -40 /tmp/probe_render.log; fail=1
fi

[[ "${fail}" == 0 ]] && echo "PROBE PASSED" || { echo "PROBE FAILED"; exit 1; }
