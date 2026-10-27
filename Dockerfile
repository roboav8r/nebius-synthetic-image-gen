# The image every Nebius job runs: stock Isaac Sim 6.1 plus this repo's code.
# The base already ships boto3, numpy and Pillow, so nothing is installed.
# `nebius_job.py` tags it with a hash of the code (`nj.build()`); configs are
# injected per job, so changing one does not need a rebuild.
FROM nvcr.io/nvidia/isaac-sim:6.1.0@sha256:af1d2b4e75d553bfa27beb5a401198654aa8d607f3b7a6749196e9ce253def20

COPY --chown=isaac-sim:isaac-sim run_sdg.py constants.py /opt/sdg/
COPY --chown=isaac-sim:isaac-sim sdg/generate_replicator.py sdg/check_validation.py /opt/sdg/sdg/
COPY --chown=isaac-sim:isaac-sim nebius/job_entry.sh /opt/sdg/nebius/

# The base image runs as the non-root `isaac-sim` user (uid 1234), but on the
# Nebius shared filesystem that user could not write even to a directory
# chowned to 1234 -- the probe failed with "Permission denied". Run as root,
# as the 4.5 image did; Kit renders the same either way.
USER root

ENTRYPOINT ["bash", "/opt/sdg/nebius/job_entry.sh"]
