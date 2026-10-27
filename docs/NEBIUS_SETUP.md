# One-time Nebius setup

You do this once per Nebius project. Afterwards, every run is launched from `nebius_synthetic_image_gen.ipynb`, or `python nebius_job.py --config ...`, on your machine.

## 0. Create a Nebius AI Cloud account, tenant, and project

At [Nebius' website](https://nebius.com/), create an account (if needed) and log in to the AI Cloud.

Then, at the [Nebius console](https://console.nebius.com):
- Create a project in an appropriate region
- Note the region and project name for the following step


## 1. Install the CLI and create a profile
Follow [Install the CLI](https://docs.nebius.com/cli/install), then:
```bash
nebius profile create          # opens a browser to sign in; pick your project
cat ~/.nebius/config.yaml      # confirm parent-id is your project ID
nebius ai job list             # should print an empty list, not an error
```
When prompted, I used these values in my profile:
```
Profile name: nebius-image-sdg
Set api endpoint: api.nebius.cloud
Select authorization type: federation
Set federation endpoint: auth.nebius.com
Switch to your browser to complete the authentication process. If it didn't open automatically, use the following link: <...>
Choose Tenant ID to use as a default: <your tenant from above>
Choose Project ID (parent-id) to use as a default: <your project from above>
```

Note: Your user needs the `editor` role in the project (the default `editors` group is enough).

## 2. Check GPU quota
In the [console quotas page](https://console.nebius.com/quota), Compute section, check that the project can run at least one **L40s** or **RTX PRO 6000** GPU VM in the project's region (`compute.instance.gpu.l40s` or `compute.instance.gpu.rtx6000` > 0). Raise the quota if not.

The jobs run Isaac Sim 6.1. NVIDIA tested 6.1 on Linux driver 595.58.03; Nebius's GPU workers report 580.173.02, so check `nvidia-smi` in the probe log if a new release misbehaves. Nebius's own [Isaac Sim example](https://github.com/nebius/serverless-ai-cookbook/tree/main/robotics/isaac-pick-place-sweep) runs 6.1 on that 580 driver.

## 3. Create the bucket
Create it in the console (**Storage → Object Storage → Create bucket**) or with `nebius storage bucket create --name <bucket>`. On the bucket's page, note:
- the **region**, such as `eu-north1` or `us-central1`
- the **endpoint**, `https://storage.<region>.nebius.cloud`

## 4. Create a service account and an S3 access key
The jobs and the notebook use an S3-style key belonging to a service account, not your personal login ([access keys](https://docs.nebius.com/iam/service-accounts/access-keys)).
1. Create a service account, e.g. `sdg-writer`, in **Administration → IAM → Service accounts**. Add it to a group with at least `editor` on the project, so it can write to the bucket. Note the service account ID.
2. Create an access key for it. Creating keys needs the `admin` role in the tenant.
   ```bash
   nebius iam v2 service-account list # find your service account's ID
   
   nebius iam v2 access-key create --account-service-account-id <service-account-id> --description sdg # note the access key "resource_id" field, e.g. accesskey-...

   nebius iam v2 access-key list 
   ```
Find the access key you just created. Note the `aws_access_key_id` and `secret` entries.

## 5. Fill in `.env`
```bash
cp .env.example .env    # .env is gitignored
```
Fill in `SDG_BUCKET`, `SDG_S3_ENDPOINT` and `SDG_S3_REGION` from step 3, and `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from step 4. These are the names boto3 reads by itself. Leave `SDG_SECRET=sdg-s3` unless you name the secret differently in step 6.

## 6. Store the keys in MysteryBox for the jobs
Jobs can't read your `.env`. They get the keys from a MysteryBox (SecretStash) secret, referenced by name with `--env-secret`. Create one secret with both keys as payload entries. The entry keys must equal the variable names. This reads the values from `.env` without printing them:
```bash
set -a; source .env; set +a
nebius mysterybox secret create --name "${SDG_SECRET}" --secret-version-payload "$(python3 -c '
import json, os
print(json.dumps([{"key": k, "string_value": os.environ[k]}
                  for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")]))')"
```
See [creating secrets](https://docs.nebius.com/mysterybox/secrets/create). To rotate the key, or rename its entries, add a new version to the same secret; jobs read the primary version:
```bash
SECRET_ID=$(nebius mysterybox secret get-by-name --name "${SDG_SECRET}" --format 'jsonpath={.metadata.id}')
nebius mysterybox secret-version create --parent-id "${SECRET_ID}" --set-primary --payload "$(python3 -c '
import json, os
print(json.dumps([{"key": k, "string_value": os.environ[k]}
                  for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")]))')"
```

## 7. Create a Container Registry for the job image
The jobs run our own image: NVIDIA's Isaac Sim 6.1 image with this repo's code copied in (see `Dockerfile`). It is built locally with Docker and pushed to a [Container Registry](https://docs.nebius.com/container-registry/quickstart) in your project. The registry is free; you pay only for the storage.
1. Install Docker with the NVIDIA container runtime (needed only to build the image, and to test it locally).
2. Create the registry in the **same region** as the jobs:
   ```bash
   nebius registry create --name sdg --parent-id <project-id>
   nebius registry configure-helper    # lets `docker push` use your Nebius login
   ```
3. Put the registry path from its console page, `cr.<region>.nebius.cloud/<registry-id>`, in `SDG_REGISTRY` in `.env`.

### How code and config reach the job
All of a job's `--inject-file` contents share a 64 KiB limit, and the code is larger than that, so it goes into the image:
1. `nj.build()` tags the image with a hash of the `sdg/` package, the Dockerfile and `.dockerignore`, then runs `docker build` and `docker push`. Both are quick when nothing changed. The first push uploads the ~11 GB Isaac base layers once.
2. `nj.launch()` calls `nj.build()` itself, so a code change is picked up automatically.
3. The launcher resolves the run config on your machine, filling in `.env`, and injects that one file at `/workspace/config.json`. Editing a config does not need a new image.

## 8. Create the shared filesystem for Isaac's caches
Every job mounts one shared filesystem at `/shared`, which holds Isaac Sim's caches at `/shared/cache-6.1/`. Each job starts on a fresh VM, so without these, Isaac starts cold every time: shader compilation and asset downloads take roughly 5–6 min. Locally, the same caches took startup from 336 s to 19.5 s.

To set it up:
1. Create a filesystem (**Compute → Shared filesystems**, about 50 GiB is plenty) in the **same region** as the jobs, or with `nebius compute filesystem create --name sdg-shared --type network_ssd --size-gibibytes 50`.
2. Put its ID (`computefilesystem-…`) in `SDG_SHARED_FS` in `.env`.

The launcher adds `--volume <id>:/shared:rw` and `SDG_CACHE_ROOT=/shared/cache-6.1` to every job, and `sdg/isaac.py` links Isaac's cache directories into it. The job image runs as root: the stock image's non-root `isaac-sim` user cannot write to the shared filesystem, even to a directory it owns.

## 9. Optional: subnet
If the project has more than one subnet, set `SDG_SUBNET_ID` in `.env`. Otherwise jobs use the default subnet.

## Done
Open `nebius_synthetic_image_gen.ipynb`. Section 0 checks all of the above (the CLI, Docker and its registry login, `.env`, that the shared filesystem exists, and bucket access with your keys). Section 2 builds and pushes the image and runs the probe job, which checks the job side: GPU, Vulkan, the shared filesystem mount, network, the MysteryBox keys, and a render under four anti-aliasing modes. The probe uploads those frames to `s3://<bucket>/<upload.prefix>/probe/<job-name>/` for comparison. The first probe is a cold start of about 6 minutes. It also warms Isaac's caches on the shared filesystem, so later probes and runs start much faster.
