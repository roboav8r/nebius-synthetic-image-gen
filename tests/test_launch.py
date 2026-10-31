import nebius_job as nj

CFG = {
    "run_name": "naive_20",
    "job": {"registry": "cr.example/reg", "platform": "gpu", "preset": "1gpu", "disk_size": "100Gi",
            "timeout": "1h", "secret": "sdg-s3", "shared_fs": "fs-1", "subnet_id": ""},
}


def test_create_command(tmp_path):
    cmd = nj.create_command(CFG, "img:tag", "sdg-naive-20-x", tmp_path / "c.json")
    assert cmd.count("--inject-file") == 1
    assert f"{tmp_path / 'c.json'}:{nj.CONFIG_IN_JOB}" in cmd
    assert "AWS_ACCESS_KEY_ID=sdg-s3" in cmd and "AWS_SECRET_ACCESS_KEY=sdg-s3" in cmd
    assert "--async" in cmd
    assert "--subnet-id" not in cmd
    envs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--env"]
    assert not any("AWS_" in e for e in envs), "keys must only arrive as secrets"


def test_image_tag_hashes_the_code():
    files = {p.relative_to(nj.ROOT).as_posix() for p in nj.image_files()}
    assert {"Dockerfile", "sdg/run.py", "sdg/generate.py"} <= files
    assert nj.image_tag(CFG).startswith("cr.example/reg/sdg:")
