"""Render a warehouse scene from two camera regimes with Isaac Sim Replicator.

    /isaac-sim/python.sh -m sdg.generate --config generator.json --output-dir <dir>

`--config` is the run config's `generator` block; any key in `DEFAULTS` can
be overridden there. Each frame re-places every object, light and camera,
then KittiWriter writes RGB, 2D boxes and segmentation for both regimes.

How this differs from NVIDIA's Replicator examples, and why:
- Assets come from the DigitalTwin warehouse library, which has pallet trucks;
  the `Isaac/Props` library used by the examples has none.
- Objects are spawned once and moved each frame. Re-creating ~60 referenced
  USDs per frame, as the examples do, stalls the renderer.
- Placement is our own rejection sampling, so no two objects overlap and none
  stands where a camera does. Replicator's `scatter_2d` collision check raised
  errors and still left objects inside one another.
- Placement, lights and cameras draw from our own seeded RNGs, so a seed
  reproduces a run. Replicator's distributions tick once per rendered frame,
  including the frames Kit burns while loading, which varies run to run.
- A large pool of objects is scattered and the camera geometry decides what
  is visible, which gives the dense, small-object frames real warehouses have.
"""

import argparse
import json
import math
import random
from pathlib import Path

from isaacsim import SimulationApp

from sdg.scene import REGIMES, RESOLUTION, SEG_COLOURS, TARGET_CLASSES, WAREHOUSE_USD

DEFAULTS = {
    "num_frames": 20,
    "rt_subframes": 16,
    "seed": 0,
    # Kit settings applied before rendering, e.g. {"/rtx/post/aa/op": 4}.
    "rtx_settings": {},
    # Objects in the pool per class.
    "n_pallet": 18,
    "n_pallet_truck": 8,
    "n_forklift": 4,
    "n_cardbox": 20,
    "n_distractor": 10,
    # Scatter area on the warehouse floor, in metres.
    "scatter_center": [-8.0, 0.0],
    "scatter_extent": [18.0, 14.0],
    # Abort before rendering if any asset's longest side falls outside this (m).
    "size_sanity_m": [0.15, 6.0],
    # Gap kept between object footprints, and between objects and the spots a
    # ground camera can stand on.
    "collision_margin_m": 0.15,
    "camera_clearance_m": 0.7,
    "placement_tries": 400,
    # Where objects that do not fit this frame are parked, out of every view.
    "parking_spot": [200.0, 200.0, -50.0],
}


def parse_args() -> tuple[dict, Path]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, help="JSON overrides for DEFAULTS")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--num-frames", type=int, help="overrides the config, for quick tests")
    args, _ = ap.parse_known_args()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    if args.num_frames is not None:
        cfg["num_frames"] = args.num_frames
    return cfg, args.output_dir


CONFIG, OUTPUT_DIR = parse_args()
simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

# Isaac modules can only be imported once the app is running.
import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils import prims  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.semantics import (  # noqa: E402
    add_labels,
    get_labels,
    remove_labels,
    upgrade_prim_semantics_to_labels,
)
from isaacsim.core.utils.stage import get_current_stage, open_stage  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

OBJECT_ROOT = "/World/Sdg"
LIGHT_ROOT = "/World/SdgLights"
N_LIGHTS = 4


def log(message: str) -> None:
    # Kit ends the process on close() without flushing stdout.
    print(f"[sdg] {message}", flush=True)


def abort(message: str) -> None:
    carb.log_error(message)
    log(f"ABORT: {message}")
    simulation_app.close()
    raise SystemExit(1)


# --- assets ----------------------------------------------------------------

def asset_tables(isaac_root: str) -> dict[str, list[str]]:
    """Spawn group -> USD URLs. The first three groups are labelled classes;
    the clutter groups render but are never labelled, because a detector that
    only ever sees target classes learns a trivially separable world."""
    # `isaac_root` is `<...>/Assets/Isaac/<version>`; DigitalTwin and SimReady
    # are version-independent siblings of `Isaac/`.
    content = isaac_root.rsplit("/Isaac/", 1)[0]
    dt = f"{content}/DigitalTwin/Assets/Warehouse/Equipment"
    sr = f"{content}/simready_content/common_assets/props"
    wh = f"{isaac_root}/Isaac/Environments/Simple_Warehouse/Props"
    ip = f"{isaac_root}/Isaac/Props"
    return {
        "pallet": [
            f"{sr}/blockpallet_a01/blockpallet_a01.usd",
            f"{sr}/blockpallet_a02/blockpallet_a02.usd",
            f"{sr}/blockpallet_a03/blockpallet_a03.usd",
            f"{sr}/blockpallet_b01/blockpallet_b01.usd",
            f"{sr}/exportpallet_a01/exportpallet_a01.usd",
            f"{sr}/economyplasticpallet_a01/economyplasticpallet_a01.usd",
            f"{sr}/aluminumpallet_a01/aluminumpallet_a01.usd",
            f"{ip}/Pallet/pallet.usd",
        ],
        "pallet_truck": [
            f"{dt}/Pallet_Trucks/Heavy_Duty_A/HeavyDutyPalletTruck_A01_PR_NVD_01.usd",
            f"{dt}/Pallet_Trucks/Low_Profile_A/LowProfilePalletTruck_A01_PR_NVD_01.usd",
            f"{dt}/Pallet_Trucks/Scale_A/PalletTruckScale_A01_PR_NVD_01.usd",
        ],
        "forklift": [
            f"{dt}/Forklifts/Forklift_A/Forklift_A01_PR_V_NVD_01.usd",
            f"{dt}/Forklifts/Forklift_B/Forklift_B01_PR_V_NVD_01.usd",
            f"{dt}/Forklifts/Forklift_C/Forklift_C01_PR_V_NVD_01.usd",
            f"{ip}/Forklift/forklift.usd",
        ],
        "clutter_box": [f"{wh}/SM_CardBoxA_01.usd"],
        "clutter_misc": [
            f"{dt}/Hand_Trucks/Std_Alum_A/StandardAluminumHandTruck_A01_PR_NVD_01.usd",
            f"{dt}/Hand_Trucks/Convertible_Steel_A/ConvertibleSteelHandTruck_A01_PR_V_NVD_01.usd",
            f"{wh}/S_TrafficCone.usd",
            f"{wh}/SM_BarelPlastic_A_01.usd",
            f"{wh}/SM_BucketPlastic_B.usd",
        ],
    }


def pool_sizes(cfg: dict) -> dict[str, int]:
    return {
        "pallet": cfg["n_pallet"],
        "pallet_truck": cfg["n_pallet_truck"],
        "forklift": cfg["n_forklift"],
        "clutter_box": cfg["n_cardbox"],
        "clutter_misc": cfg["n_distractor"],
    }


# --- units and bounds ------------------------------------------------------

_unit_scales: dict[str, float] = {}


def unit_scale(url: str, stage_mpu: float) -> float:
    """Scale that brings a referenced asset to its real size in this stage.

    USD does not rescale across a reference. DigitalTwin assets are authored
    in centimetres (metersPerUnit 0.01) and this stage is in metres, so
    without this they arrive 100x too large.
    """
    if url not in _unit_scales:
        try:
            asset = Usd.Stage.Open(url)
            mpu = UsdGeom.GetStageMetersPerUnit(asset) if asset else stage_mpu
        except Exception as e:  # noqa: BLE001
            carb.log_warn(f"could not read metersPerUnit for {url}: {e}")
            mpu = stage_mpu
        _unit_scales[url] = float(mpu) / float(stage_mpu)
    return _unit_scales[url]


_bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def world_size(prim) -> list[float] | None:
    """World-space size of a prim, as the union of its boundable descendants.

    `ComputeWorldBound` on the root returns a partial union on these assets
    (a pallet truck measured smaller than its own body), so union by hand.
    """
    bounds = Gf.Range3d()
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Boundable):
            _bbox_cache.Clear()
            r = _bbox_cache.ComputeWorldBound(child).ComputeAlignedRange()
            if not r.IsEmpty():
                bounds = Gf.Range3d.GetUnion(bounds, r)
    return None if bounds.IsEmpty() else [float(v) for v in bounds.GetSize()]


# --- moving prims ----------------------------------------------------------

def pose_ops(prim, translate_and_orient_only: bool = False):
    """(translate op, orient op, quaternion type) for writing a prim's pose.

    `create_prim` authors an orient quaternion, which `XformCommonAPI` cannot
    drive, so poses are written to the ops directly. An op is added at the
    precision of any attribute already authored under its name; adding it at
    another precision raises, and Kit turns that into a segfault.

    `translate_and_orient_only` first drops any other ops (Replicator cameras
    come with rotateXYZ). Never use it on spawned objects: it drops their scale.
    """
    xf = UsdGeom.Xformable(prim)
    allowed = {UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient}
    if translate_and_orient_only and {op.GetOpType() for op in xf.GetOrderedXformOps()} - allowed:
        xf.ClearXformOpOrder()
    ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}

    def get_or_add(op_type, add, attr, other_type, other_precision):
        if op_type in ops:
            return ops[op_type]
        existing = prim.GetAttribute(attr)
        if existing and str(existing.GetTypeName()) == other_type:
            return add(other_precision)
        return add()

    translate = get_or_add(UsdGeom.XformOp.TypeTranslate, xf.AddTranslateOp, "xformOp:translate",
                           "float3", UsdGeom.XformOp.PrecisionFloat)
    orient = get_or_add(UsdGeom.XformOp.TypeOrient, xf.AddOrientOp, "xformOp:orient",
                        "quatd", UsdGeom.XformOp.PrecisionDouble)
    quat = Gf.Quatf if orient.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd
    return translate, orient, quat


def look_at_quat(eye, target, quat, up=(0.0, 0.0, 1.0)):
    """Orientation that aims a Replicator camera's Xform from `eye` at `target`.

    In Isaac Sim 6.1 the Xform follows the USD camera convention (looks down
    -Z, +Y up), so the rotation's rows are (right, up, -forward). This matches
    `rep.modify.pose(look_at=...)` exactly. Isaac Sim 4.5 needed a different
    basis; recheck after any Isaac upgrade.
    """
    fwd = Gf.Vec3d(*target) - Gf.Vec3d(*eye)
    fwd = fwd.GetNormalized() if fwd.GetLength() > 1e-9 else Gf.Vec3d(1.0, 0.0, 0.0)
    u = Gf.Vec3d(*up)
    if abs(Gf.Dot(fwd, u)) > 0.999:
        u = Gf.Vec3d(0.0, 1.0, 0.0)
    right = Gf.Cross(fwd, u).GetNormalized()
    upv = Gf.Cross(right, fwd)
    q = Gf.Matrix3d(*right, *upv, *(-fwd)).ExtractRotation().GetQuat()
    im = q.GetImaginary()
    return quat(float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))


# --- building the scene ----------------------------------------------------

def spawn(stage, label: str, urls: list[str], count: int) -> list[str]:
    """Create `count` prims cycling through `urls`, at their real size."""
    stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    paths = []
    for j in range(count):
        url = urls[j % len(urls)]
        path = f"{OBJECT_ROOT}/{label}/{label}_{j}"
        s = unit_scale(url, stage_mpu)
        prims.create_prim(prim_path=path, usd_path=url, position=(0.0, 0.0, 0.0),
                          orientation=euler_angles_to_quat([0.0, 0.0, 0.0]),
                          scale=(s, s, s), semantic_label=None)
        paths.append(path)
    return paths


def apply_semantics(stage, spawned: dict[str, list[str]]) -> None:
    """Replace whatever labels our objects brought with them by our own.

    Referenced assets carry baked semantics (SimReady pallets arrive as
    `pallet,prop_general`), so this has to run after spawning. Assets using
    the older SemanticsAPI are upgraded first so the removal catches them.
    """
    for label, paths in spawned.items():
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            upgrade_prim_semantics_to_labels(prim, include_descendants=True)
            remove_labels(prim, include_descendants=True)
            if label in TARGET_CLASSES:
                add_labels(prim, labels=[label], instance_name="class")


def stage_vocabulary(stage) -> set[str]:
    vocab = set()
    for prim in stage.Traverse():
        vocab.update(get_labels(prim).get("class", []))
    return vocab


def scale_errors(stage, spawned: dict[str, list[str]], lo: float, hi: float) -> list[str]:
    """Objects whose longest side is outside [lo, hi] m, from a sample per group."""
    errors = []
    for label, paths in spawned.items():
        sizes = [s for s in (world_size(stage.GetPrimAtPath(p)) for p in paths[:3]) if s]
        log(f"size {label:14} " + "  ".join(f"({x:.2f} x {y:.2f} x {z:.2f})" for x, y, z in sizes))
        errors += [f"{label}: longest side {max(s):.2f} m outside [{lo}, {hi}]"
                   for s in sizes if not lo <= max(s) <= hi]
    return errors


# --- per-frame randomisation -----------------------------------------------

class Area:
    """The scatter area on the floor."""

    def __init__(self, center, extent):
        (self.cx, self.cy), (self.ex, self.ey) = center, extent

    def random_xy(self, rng):
        return (self.cx + rng.uniform(-self.ex / 2, self.ex / 2),
                self.cy + rng.uniform(-self.ey / 2, self.ey / 2))


def place_objects(rng, objects: dict[str, dict], reserved, area: Area, cfg: dict) -> int:
    """Put every object on the floor with no two footprints overlapping.

    `objects` maps a prim path to its footprint `radius` and its `pose_ops`.
    Each footprint is a circle; `reserved` circles (camera standing spots) are
    kept clear too. Objects go in biggest first; one that finds no spot in
    `placement_tries` is parked out of view, so the pool size is constant.
    Returns how many objects were placed in the scene.
    """
    margin = cfg["collision_margin_m"]
    placed = list(reserved)
    for obj in sorted(objects.values(), key=lambda o: -o["radius"]):
        r = obj["radius"]
        translate, orient, quat = obj["pose_ops"]
        for _ in range(cfg["placement_tries"]):
            x, y = area.random_xy(rng)
            if all((x - px) ** 2 + (y - py) ** 2 >= (r + pr + margin) ** 2 for px, py, pr in placed):
                placed.append((x, y, r))
                translate.Set(Gf.Vec3d(x, y, 0.0))
                w, qx, qy, qz = euler_angles_to_quat([0.0, 0.0, rng.uniform(0.0, 2 * math.pi)])
                orient.Set(quat(float(w), float(qx), float(qy), float(qz)))
                break
        else:
            translate.Set(Gf.Vec3d(*cfg["parking_spot"]))
    return len(placed) - len(reserved)


def pose_lights(rng, lights: list, area: Area) -> None:
    """Four sphere lights above the scatter area, random position, power and tint."""
    for prim in lights:
        translate, orient, quat = pose_ops(prim, translate_and_orient_only=True)
        x = rng.uniform(area.cx - area.ex / 2, area.cx + area.ex / 2)
        y = rng.uniform(area.cy - area.ey / 2, area.cy + area.ey / 2)
        translate.Set(Gf.Vec3d(x, y, rng.uniform(5.0, 9.0)))
        orient.Set(quat(1.0, 0.0, 0.0, 0.0))
        light = UsdLux.SphereLight(prim)
        light.CreateIntensityAttr().Set(rng.uniform(2000.0, 12000.0))
        light.CreateRadiusAttr().Set(rng.uniform(3.0, 9.0))
        light.CreateColorAttr().Set(Gf.Vec3f(rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0),
                                             rng.uniform(0.55, 1.0)))


def camera_rigs(area: Area) -> dict[str, dict]:
    """Per regime: lens, near clip, the eye positions it may use, and the
    half-width of the box around the scene centre it aims into.

    Focal lengths assume Replicator's default 20.955 mm aperture.
    - elevated_fixed (24 mm, about 47 deg): CCTV mounts at 5.5-8 m on the
      corners and edge midpoints of the scatter area, looking across it.
    - ground_handheld (16 mm, about 66 deg): eye height 1.2-1.7 m on rings
      2.5-6 m from the centre, the working distance of photos taken in aisles.
    """
    cx, cy, ex, ey = area.cx, area.cy, area.ex, area.ey
    mounts = [(cx + sx * ex / 2, cy + sy * ey / 2, z)
              for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0))
              for z in (5.5, 7.0, 8.0)]
    ring = [(cx + r * math.cos(a), cy + r * math.sin(a), z)
            for r in (2.5, 3.5, 4.5, 6.0)
            for a in [i * math.pi / 6 for i in range(12)]
            for z in (1.2, 1.45, 1.7)]
    return {
        "elevated_fixed": {"focal_mm": 24.0, "near_clip": 0.1, "eyes": mounts, "aim_half_m": 5.0},
        "ground_handheld": {"focal_mm": 16.0, "near_clip": 0.05, "eyes": ring, "aim_half_m": 2.5},
    }


def pose_cameras(rng, rigs: dict, area: Area) -> dict[str, dict]:
    """Place each camera at one of its eyes, aimed at a random point near the
    centre. Returns the drawn eye and aim point per regime."""
    drawn = {}
    for regime in REGIMES:
        rig = rigs[regime]
        h = rig["aim_half_m"]
        eye = rig["eyes"][rng.randrange(len(rig["eyes"]))]
        lo, hi = (area.cx - h, area.cy - h, 0.0), (area.cx + h, area.cy + h, 1.2)
        target = tuple(rng.uniform(a, b) for a, b in zip(lo, hi))
        translate, orient, quat = pose_ops(rig["prim"], translate_and_orient_only=True)
        translate.Set(Gf.Vec3d(*eye))
        orient.Set(look_at_quat(eye, target, quat))
        drawn[regime] = {"look_at": [float(v) for v in target]}
    return drawn


def camera_pose(prim) -> dict:
    m = omni.usd.get_world_transform_matrix(prim)
    t = m.ExtractTranslation()
    r = m.ExtractRotation().Decompose((1, 0, 0), (0, 1, 0), (0, 0, 1))
    return {"position": [t[0], t[1], t[2]], "rotation_xyz_deg": [r[0], r[1], r[2]]}


def create_writer(render_products, output_dir: Path):
    """KittiWriter, labelling only TARGET_CLASSES.

    The semantic filter has to be passed to each annotator as well: since Kit
    109 an annotator created without one gets `*:*`, which overrides the
    writer's predicate and labels `floor`, `rack`, `wall` and everything else
    the warehouse ships with.
    """
    predicate = "class: " + " | ".join(TARGET_CLASSES)
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(
        output_dir=str(output_dir),
        omit_semantic_type=True,
        semantic_filter_predicate=predicate,
        # The default of 25 px drops small boxes; a third of real warehouse
        # pallets are that small.
        bbox_height_threshold=1,
        partly_occluded_threshold=0.5,
        fully_visible_threshold=0.95,
        colorize_instance_segmentation=False,
        use_kitti_dir_names=False,  # rgb/ + object_detection/
        mapping_dict={cls: list(colour) for cls, colour in SEG_COLOURS.items()},
    )
    init_params = {
        "bounding_box_2d_tight_fast": {},
        "bounding_box_2d_loose_fast": {},
        # A private method, but the only way to get the writer's own mapping.
        "semantic_segmentation": {"mapping": writer._get_anno_semantic_mapping()},
        "instance_segmentation_fast": {"colorize": False},
    }
    annotators = []
    for annotator in writer.annotators:
        name = annotator if isinstance(annotator, str) else annotator.get_name()
        if name in init_params:
            annotator = rep.AnnotatorRegistry.get_annotator(
                name, init_params={**init_params[name], "semanticFilter": predicate})
        annotators.append(annotator)
    writer.annotators = annotators
    writer.attach(render_products)
    return writer


# --- main ------------------------------------------------------------------


def main() -> None:
    cfg = CONFIG
    settings = carb.settings.get_settings()
    # 6.1's metrics assembler also corrects units, but only on some assets and
    # on top of `unit_scale`; one forklift came out 0.02 m long. Turn it off.
    settings.set("/metricsAssembler/operationMode", 0)
    for key, value in cfg["rtx_settings"].items():
        settings.set(key, value)
        log(f"{key} = {settings.get(key)}")
    rep.set_global_seed(cfg["seed"])

    isaac_root = get_assets_root_path()
    if isaac_root is None:
        abort("no Isaac assets root")
    log(f"loading {WAREHOUSE_USD}")
    if not open_stage(isaac_root + WAREHOUSE_USD):
        abort("could not open the warehouse stage")
    rep.orchestrator.set_capture_on_play(False)
    stage = get_current_stage()

    # The warehouse's own semantics are left alone: it labels its 158 racked
    # pallets correctly, and the writer's filter drops `floor`, `wall` and the
    # rest. Clearing them would also fail, since USD cannot author over
    # instance proxies.
    assets, sizes = asset_tables(isaac_root), pool_sizes(cfg)
    spawned = {}
    for label, urls in assets.items():
        spawned[label] = spawn(stage, label, urls, sizes[label])
        log(f"spawned {len(spawned[label])} x {label}")
    for _ in range(120):  # let referenced layers load before measuring them
        simulation_app.update()
    apply_semantics(stage, spawned)

    vocab = stage_vocabulary(stage)
    log(f"stage semantic vocabulary ({len(vocab)}): {sorted(vocab)}")
    if set(TARGET_CLASSES) - vocab:
        abort(f"target classes missing from the stage: {sorted(set(TARGET_CLASSES) - vocab)}")
    errors = scale_errors(stage, spawned, *cfg["size_sanity_m"])
    if errors:
        abort("scale check failed: " + "; ".join(errors))

    # Footprint: a circle around the object's measured XY extent.
    objects = {}
    for path in (p for paths in spawned.values() for p in paths):
        prim = stage.GetPrimAtPath(path)
        size = world_size(prim)
        objects[path] = {"radius": 0.5 * math.hypot(size[0], size[1]) if size else 0.5,
                         "pose_ops": pose_ops(prim)}

    area = Area(cfg["scatter_center"], cfg["scatter_extent"])
    rigs = camera_rigs(area)
    # Keep objects off every spot a ground camera can stand on.
    stands = sorted({(round(x, 3), round(y, 3)) for x, y, _ in rigs["ground_handheld"]["eyes"]})
    reserved = [(x, y, cfg["camera_clearance_m"]) for x, y in stands]

    lights = []
    for i in range(N_LIGHTS):
        path = f"{LIGHT_ROOT}/Sphere_{i}"
        prims.create_prim(prim_path=path, prim_type="SphereLight")
        lights.append(stage.GetPrimAtPath(path))

    render_products = []
    for regime in REGIMES:
        rig = rigs[regime]
        camera = rep.create.camera(name=regime, focal_length=rig["focal_mm"],
                                   clipping_range=(rig["near_clip"], 1e6))
        rig["prim"] = stage.GetPrimAtPath(f"/Replicator/{regime}_Xform")
        render_products.append(rep.create.render_product(camera, RESOLUTION, name=regime))
    # Don't render while the writer is being set up.
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = create_writer(render_products, OUTPUT_DIR)
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(True)

    n_frames, rt_subframes, seed = cfg["num_frames"], cfg["rt_subframes"], cfg["seed"]
    place_rng = random.Random(seed)
    camera_rng = random.Random(seed + 1_000_003)
    light_rng = random.Random(seed + 2_000_003)
    log(f"rendering {n_frames} frames x {len(REGIMES)} regimes @ {RESOLUTION}, "
        f"rt_subframes={rt_subframes}, seed={seed}")
    frames, n_placed = {}, []
    for i in range(n_frames):
        n_placed.append(place_objects(place_rng, objects, reserved, area, cfg))
        drawn = pose_cameras(camera_rng, rigs, area)
        pose_lights(light_rng, lights, area)
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)
        frames[str(i)] = {regime: {**camera_pose(rigs[regime]["prim"]), **drawn[regime]}
                          for regime in REGIMES}
        frames[str(i)]["n_placed"] = n_placed[-1]
        if i % 50 == 0:
            log(f"  frame {i}/{n_frames} ({n_placed[-1]}/{len(objects)} placed)")
    log(f"placed {min(n_placed)}-{max(n_placed)} of {len(objects)} objects per frame")

    rep.orchestrator.wait_until_complete()
    (OUTPUT_DIR / "sim_params.json").write_text(json.dumps({"config": cfg, "frames": frames}, indent=1))
    writer.detach()
    for rp in render_products:
        rp.destroy()
    log("done")
    simulation_app.close()


main()
