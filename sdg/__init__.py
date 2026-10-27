"""Synthetic warehouse images with Isaac Sim Replicator, run as Nebius serverless jobs.

Everything in this package is copied into the job image. `generate` and the
render half of `probe` run under Isaac Sim's own interpreter; the rest runs
under any Python with boto3, numpy and Pillow.
"""
