import sys
from types import SimpleNamespace

from sdg import storage

CFG = {"run_name": "r", "upload": {"bucket": "b", "endpoint_url": "e", "region": "x", "prefix": "/p/"}}


def fake_boto3(monkeypatch, calls, keys=()):
    s3 = SimpleNamespace(
        upload_file=lambda src, bucket, key: calls.append(("file", key)),
        put_object=lambda **kw: calls.append(("marker", kw["Key"])),
        get_paginator=lambda _: SimpleNamespace(
            paginate=lambda **kw: [{"Contents": [{"Key": f"{kw['Prefix']}{k}"} for k in keys]}]),
    )
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *a, **k: s3))


def test_run_prefix():
    assert storage.run_prefix(CFG) == "p/r"


def test_upload_writes_marker_last(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "1.png").write_text("x")
    (tmp_path / "summary.json").write_text("{}")
    calls = []
    fake_boto3(monkeypatch, calls)
    storage.upload_dir(CFG, tmp_path, "p/r", marker="{}")
    assert calls[-1] == ("marker", "p/r/_SUCCESS")
    assert sorted(calls[:-1]) == [("file", "p/r/a/1.png"), ("file", "p/r/summary.json")]


def test_is_done(monkeypatch):
    fake_boto3(monkeypatch, [], keys=["a/1.png", "_SUCCESS"])
    assert storage.is_done(CFG, "p/r")
    fake_boto3(monkeypatch, [], keys=["a/1.png"])
    assert not storage.is_done(CFG, "p/r")
