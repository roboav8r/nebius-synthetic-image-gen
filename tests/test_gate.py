import numpy as np
from PIL import Image

from sdg import gate
from sdg.scene import SEG_COLOURS

W, H = 960, 540


def make_run(root, boxes_per_frame, n_frames=4):
    """A one-regime run whose every box is fully covered by its class colour."""
    cam = root / "elevated_fixed" / "elevated_fixed"
    for sub in ("rgb", "object_detection", "semantic_segmentation"):
        (cam / sub).mkdir(parents=True)
    for i in range(n_frames):
        Image.new("RGB", (W, H), (90, 90, 90)).save(cam / "rgb" / f"{i}.png")
        mask = np.zeros((H, W, 4), np.uint8)
        rows = []
        for b in range(boxes_per_frame):
            x1, y1 = 20 + 40 * (b % 20), 20 + 40 * (b // 20)
            x2, y2 = x1 + 30, y1 + (20 if b % 2 else 30)
            mask[y1:y2, x1:x2] = SEG_COLOURS["pallet"]
            rows.append(f"pallet 0.00 0 0.00 {x1} {y1} {x2} {y2} 0 0 0 0 0 0 0")
        (cam / "object_detection" / f"{i}.txt").write_text("\n".join(rows))
        Image.fromarray(mask).save(cam / "semantic_segmentation" / f"{i}.png")
    return cam


def test_read_labels(tmp_path):
    f = tmp_path / "0.txt"
    f.write_text("pallet_truck 0.00 0 0.00 31 0 190 108 0.00 0.00 0.00 0.00 0.00 0.00 0.00\n\n")
    assert gate.read_labels(f) == [("pallet_truck", 31.0, 0.0, 190.0, 108.0)]


def test_dense_run_passes(tmp_path):
    make_run(tmp_path, boxes_per_frame=15)
    passed, report = gate.run_gate(tmp_path)
    assert passed, report
    assert (tmp_path / "contact_elevated_fixed.jpg").exists()


def test_sparse_run_fails_on_density(tmp_path):
    cam = make_run(tmp_path, boxes_per_frame=2)
    failed = [name for name, ok, _ in gate.checks(gate.regime_stats(cam)) if not ok]
    assert failed == ["median boxes/frame in [8, 30]"]


def test_fill_measures_hidden_boxes(tmp_path):
    cam = make_run(tmp_path, boxes_per_frame=1, n_frames=1)
    (cam / "object_detection" / "0.txt").write_text("pallet 0 0 0 500 300 600 400 0 0 0 0 0 0 0")
    [box] = gate.regime_stats(cam)["boxes"]
    assert box["fill"] == 0.0
