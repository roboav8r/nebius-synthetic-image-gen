# One-time Nebius setup

You do this once per Nebius project. Afterwards, every run is launched from `nebius_synthetic_image_gen.ipynb` (or `python nebius_job.py --config ...`) on your machine.

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

## 3. Create the bucket
Create it in the console (**Storage → Object Storage → Create bucket**) or with `nebius storage bucket create --name <bucket>`. On the bucket's page, note:
- the **region**, such as `eu-north1` or `us-central1`
- the **endpoint**, `https://storage.<region>.nebius.cloud`

## 4. Create a service account and an S3 access key
The jobs and the notebook use an S3-style key belonging to a service account, not your personal login ([access keys](https://docs.nebius.com/iam/service-accounts/access-keys)).
1. Create a service account, e.g. `sdg-writer`, in **Administration → IAM → Service accounts**. Add it to a group with at least `editor` on the project, so it can write to the bucket. Note the service account ID.
2. Create an access key for it. Creating keys needs the `admin` role in the tenant.
   ```bash
   nebius iam v2 service-account list # Find your SDG service account's ID, e.g. sdk-writer
   
   nebius iam v2 access-key create --account-service-account-id <service-account-id> --description sdg # note the access key "resource_id" field, e.g. accesskey-...

   nebius iam v2 access-key list 
   ```
Find the access key you just created. Note the `aws_access_key_id` and `secret` entries.

## 5. Fill in `.env`
```bash
cp .env.example .env    # .env is gitignored
```
Fill in `SDG_BUCKET`, `SDG_S3_ENDPOINT`, `SDG_S3_REGION` from step 3, and `NEBIUS_S3_ACCESS_KEY_ID` and `NEBIUS_S3_SECRET_ACCESS_KEY` from step 4. Leave `SDG_SECRET=sdg-s3` unless you name the secret differently in step 6.

## 6. Store the keys in MysteryBox for the jobs
Jobs can't read your `.env`. They get the keys from a MysteryBox (SecretStash) secret, referenced by name with `--env-secret`. Create one secret with both keys as payload entries. The entry keys must equal the variable names. This reads the values from `.env` without printing them:
```bash
set -a; source .env; set +a
nebius mysterybox secret create --name "${SDG_SECRET}" --secret-version-payload "$(python3 -c '
import json, os
print(json.dumps([{"key": k, "string_value": os.environ[k]}
                  for k in ("NEBIUS_S3_ACCESS_KEY_ID", "NEBIUS_S3_SECRET_ACCESS_KEY")]))')"
```
See [creating secrets](https://docs.nebius.com/mysterybox/secrets/create). To rotate the key later, add a new version to the same secret. The jobs pick up the latest version.

## 7. Optional: a shared filesystem for Isaac's caches
Each job starts on a fresh VM, so Isaac Sim starts cold every time: shader compilation, asset downloads, roughly 5–6 min. A shared filesystem mounted at `/cache` keeps those caches between jobs. Locally, the same caches took startup from 336 s to 19.5 s.
1. Create a filesystem (**Compute → Shared filesystems**, about 50 GiB is plenty) in the same region.
2. Put its ID (`computefilesystem-…`) in `SDG_CACHE_VOLUME` in `.env`.

The launcher then adds `--volume <id>:/cache:rw` and `SDG_CACHE_ROOT=/cache`, and `run_sdg.py` links Isaac's cache directories into it.

## 8. Optional: subnet
If the project has more than one subnet, set `NEBIUS_SUBNET_ID` in `.env`. Otherwise jobs use the default subnet.

## Done
Open `nebius_synthetic_image_gen.ipynb`. Section 0 checks all of the above (the CLI, `.env`, and bucket access with your keys). Section 2 runs the probe job, which checks the job side: GPU, Vulkan, network, the MysteryBox keys, and a one-frame render.
