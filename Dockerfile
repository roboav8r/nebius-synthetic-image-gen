# The image every job runs: Isaac Sim 6.1, pinned by digest, plus the sdg/
# package. The base already ships boto3, numpy and Pillow. `nebius_job.py`
# tags it with a hash of these files; the run config is injected per job.
FROM nvcr.io/nvidia/isaac-sim:6.1.0@sha256:af1d2b4e75d553bfa27beb5a401198654aa8d607f3b7a6749196e9ce253def20

WORKDIR /opt/sdg
COPY sdg/ sdg/
ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y PYTHONUNBUFFERED=1

# The base image's non-root user cannot write to a Nebius shared filesystem,
# even to a directory it owns, so run as root. Kit renders the same either way.
USER root

ENTRYPOINT ["/isaac-sim/python.sh", "-m", "sdg.run"]
