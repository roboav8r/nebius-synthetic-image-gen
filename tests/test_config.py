import json

import pytest

from sdg.config import load_config


def write(path, data):
    path.write_text(json.dumps(data))
    return path


def test_extends_deep_merges_child_over_parent(tmp_path):
    write(tmp_path / "base.json", {"a": 1, "nested": {"x": 1, "y": 2}})
    child = write(tmp_path / "child.json", {"extends": "base.json", "nested": {"y": 3}})
    assert load_config(child) == {"a": 1, "nested": {"x": 1, "y": 3}}


def test_env_references(tmp_path, monkeypatch):
    monkeypatch.setenv("BUCKET", "b")
    monkeypatch.delenv("SUBNET", raising=False)
    cfg = write(tmp_path / "c.json", {"bucket": "${BUCKET}", "subnet": "${SUBNET:-}",
                                      "list": ["s3://${BUCKET}/x"]})
    assert load_config(cfg) == {"bucket": "b", "subnet": "", "list": ["s3://b/x"]}


def test_unset_reference_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING", raising=False)
    with pytest.raises(KeyError, match="MISSING"):
        load_config(write(tmp_path / "c.json", {"bucket": "${MISSING}"}))


def test_repo_configs_load(monkeypatch):
    for var in ("SDG_BUCKET", "SDG_S3_ENDPOINT", "SDG_S3_REGION", "SDG_REGISTRY", "SDG_SHARED_FS"):
        monkeypatch.setenv(var, "x")
    cfg = load_config("configs/naive_20.json")
    assert cfg["generator"]["num_frames"] == 20
    assert cfg["generator"]["rt_subframes"] == 8  # from common.json
    assert cfg["upload"]["overwrite"] is True
