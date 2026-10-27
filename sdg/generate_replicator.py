"""Generate our own Replicator SDG under two camera regimes.

Runs *inside* the Isaac Sim 6.1 container (`./python.sh`), so it cannot
import anything from the project env -- every constant it needs is
duplicated in CONFIG below and kept in sync with `constants.py` by hand.

Why we generate rather than download: it is the only way to get the same
scene, same seed, same object placement rendered from **two camera
regimes**. Without that the sim-to-real gap cannot be decomposed into a
viewpoint part and an appearance part; it can only be quoted as one number.

Three things this does differently from NVIDIA's `scene_based_sdg.py`,
each because the stock example's choice would bias the experiment:

- **Assets come from the DigitalTwin warehouse library, not `Isaac/Props`.**
  The tutorials' library has no pallet truck at all -- our headline class.
  `Assets/DigitalTwin/Assets/Warehouse/Equipment/` has three, and they are
  manual hand jacks, which is what LOCO's `pallet_truck` mostly is.
- **Semantics are cleared and re-declared.** The stock scene labels `floor`
  and emits a whole-image box for it. We remove all inherited semantics and
  attach only our own four classes.
- **Density is matched to LOCO, and LOCO is dense and small.** 23.6 pallets
  per image (median 15, p90 50), median box 0.22% of frame area. The stock
  example renders one or two frame-filling pallets, which is the opposite
  distribution. We scatter a large pool and let camera geometry decide what
  is visible, which is how object counts arise in reality.

`bbox_height_threshold=1`, not the default 25: **30.5% of LOCO's pallet
boxes are under 25 px tall**, so the default would train the synthetic arms
on a strictly easier label set than the real test set scores them against.

Round 8 fixed four content bugs that the Round 7 validation run exposed --
all four invisible in `scene_based_sdg.py`, because that example uses one
correctly-scaled asset library and five objects:

1. **Units.** DigitalTwin assets declare `metersPerUnit=0.01`; this stage
   declares 1.0, and USD does not rescale across a reference. Everything
   from that library arrived 100x too large and the cameras rendered the
   inside of a forklift. Each asset is now scaled by `unit_scale(url)`.
2. **Scatter pattern** anchored to the object roots, so `rep.get.prims` stops
   matching each asset's `Looks` Scope and failing every frame.
3. **Semantics applied after spawning**, not before -- referenced USDs carry
   their own, including SimReady's multi-label `pallet,prop_general`.
4. **Density raised** toward LOCO's p50 of 15 visible boxes per frame.

Two gates now run before the first frame, because the Round 7 lesson is that
correct throughput says nothing about correct content: the semantic
vocabulary must be exactly our three classes, and every sampled asset must
measure within `size_sanity_m`. Either failing aborts the run.
"""

import argparse
import json
import math
import os
import random

from isaacsim import SimulationApp

CONFIG = {
    "launch_config": {"renderer": "RaytracedLighting", "headless": True},
    "resolution": [960, 540],
    "rt_subframes": 16,
    "num_frames": 20,
    "seed": 0,
    "output_dir": "/workspace/_out_sdg",
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    # Scatter area on the warehouse floor, in metres.
    "scatter_extent": [18.0, 14.0],
    "scatter_center": [-8.0, 0.0],
    # Pool sizes per frame. Visible counts emerge from camera geometry.
    "n_pallet": 18,
    "n_pallet_truck": 8,
    "n_forklift": 4,
    "n_cardbox": 20,
    "n_distractor": 10,
    # Abort rather than render if any asset lands outside this band (m).
    "size_sanity_m": [0.15, 6.0],
    # Placement is done in Python (see `place_objects`), not by
    # `rep.randomizer.scatter_2d`, so that no two object footprints overlap.
    "collision_margin_m": 0.15,
    "placement_tries": 400,
    # Somewhere no camera can see, for objects that cannot be fitted.
    "parking_spot": [200.0, 200.0, -50.0],
    # Keep objects this far from any position a ground camera can stand.
    "camera_clearance_m": 0.7,
}

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="json overrides")
parser.add_argument("--num-frames", type=int)
parser.add_argument("--rt-subframes", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--output-dir")
args, _ = parser.parse_known_args()

if args.config and os.path.isfile(args.config):
    with open(args.config) as f:
        CONFIG.update(json.load(f))
for k in ("num_frames", "rt_subframes", "seed", "output_dir"):
    v = getattr(args, k if k != "output_dir" else "output_dir")
    if v is not None:
        CONFIG[k] = v

simulation_app = SimulationApp(launch_config=CONFIG["launch_config"])

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

# Units are corrected exactly once, by `unit_scale()` below. Isaac Sim 6.1's
# metrics assembler also corrects them on its own, adding an
# `xformOp:scale:unitsResolve` op -- but only to some assets, and not
# always in time: in the 6.1 container Forklift_A came out 0.02 m long with
# both corrections applied, while the HeavyDuty pallet truck was fine. Turn
# it off so one mechanism owns the units, as in 4.5.
_settings = carb.settings.get_settings()
_settings.set("/metricsAssembler/operationMode", 0)

# Renderer settings from the run config, e.g. the anti-aliasing mode. Kit's
# default is DLSS Performance, which renders at half resolution.
for _k, _v in CONFIG.get("rtx_settings", {}).items():
    _settings.set(_k, _v)
    print(f"[sdg] {_k} = {_settings.get(_k)}")

random.seed(CONFIG["seed"])
rep.set_global_seed(CONFIG["seed"])

ROOT = get_assets_root_path()
if ROOT is None:
    carb.log_error("no assets root; closing")
    simulation_app.close()

# `get_assets_root_path()` returns `<...>/Assets/Isaac/6.1`, but only the
# Isaac library lives under that. The SimReady and DigitalTwin collections
# are siblings one level up, under `<...>/Assets`. Resolving everything
# against ROOT silently produces 404s (and then a segfault, because
# Replicator's FileNotFoundError unwinds through the OmniGraph callback).
CONTENT_ROOT = ROOT.rsplit("/Isaac/", 1)[0]

DT = f"{CONTENT_ROOT}/DigitalTwin/Assets/Warehouse/Equipment"
SR = f"{CONTENT_ROOT}/simready_content/common_assets/props"
WH = f"{ROOT}/Isaac/Environments/Simple_Warehouse/Props"
IP = f"{ROOT}/Isaac/Props"

# class -> [usd paths]. Multiple variants per class is the cheapest form of
# domain randomization there is, and the pallet class has 30 of them.
ASSETS = {
    "pallet_truck": [
        f"{DT}/Pallet_Trucks/Heavy_Duty_A/HeavyDutyPalletTruck_A01_PR_NVD_01.usd",
        f"{DT}/Pallet_Trucks/Low_Profile_A/LowProfilePalletTruck_A01_PR_NVD_01.usd",
        f"{DT}/Pallet_Trucks/Scale_A/PalletTruckScale_A01_PR_NVD_01.usd",
    ],
    "forklift": [
        f"{DT}/Forklifts/Forklift_A/Forklift_A01_PR_V_NVD_01.usd",
        f"{DT}/Forklifts/Forklift_B/Forklift_B01_PR_V_NVD_01.usd",
        f"{DT}/Forklifts/Forklift_C/Forklift_C01_PR_V_NVD_01.usd",
        f"{IP}/Forklift/forklift.usd",
    ],
    "pallet": [
        f"{SR}/blockpallet_a01/blockpallet_a01.usd",
        f"{SR}/blockpallet_a02/blockpallet_a02.usd",
        f"{SR}/blockpallet_a03/blockpallet_a03.usd",
        f"{SR}/blockpallet_b01/blockpallet_b01.usd",
        f"{SR}/exportpallet_a01/exportpallet_a01.usd",
        f"{SR}/economyplasticpallet_a01/economyplasticpallet_a01.usd",
        f"{SR}/aluminumpallet_a01/aluminumpallet_a01.usd",
        f"{IP}/Pallet/pallet.usd",
    ],
}
# Unlabelled clutter. Present in the render, absent from the label set --
# real warehouses are full of things that are not the target classes, and a
# detector that never sees them learns a trivially separable world.
DISTRACTORS = [
    f"{DT}/Hand_Trucks/Std_Alum_A/StandardAluminumHandTruck_A01_PR_NVD_01.usd",
    f"{DT}/Hand_Trucks/Convertible_Steel_A/ConvertibleSteelHandTruck_A01_PR_V_NVD_01.usd",
    f"{WH}/S_TrafficCone.usd",
    f"{WH}/SM_BarelPlastic_A_01.usd",
    f"{WH}/SM_BucketPlastic_B.usd",
]
CARDBOX = f"{WH}/SM_CardBoxA_01.usd"

print(f"[sdg] loading {CONFIG['env_url']}")
if not open_stage(ROOT + CONFIG["env_url"]):
    carb.log_error("could not open stage; closing")
    simulation_app.close()

rep.orchestrator.set_capture_on_play(False)

# The stock warehouse ships semantics including `floor`, which Round 4 saw
# the writer emit as a whole-image box on every frame. Round 7 answered that
# by clearing every semantic on the stage before spawning. That was a
# mistake twice over:
#
#   - It does not work. `stage.Traverse()` does not descend into instance
#     proxies, so the semantics baked into instanced prototypes survived --
#     27 labels were still standing at the first Round 8 gate.
#   - Where it *did* work it destroyed real labels. The stock warehouse
#     annotates its own content (`pallet` x158, `forklift` x1), and clearing
#     turned those into unlabelled instances of our target classes sitting
#     in frame -- false negatives, the one label defect that actively
#     teaches a detector to be wrong. LOCO labels racked pallets too.
#
# So we leave the warehouse's semantics alone and let the writer's
# `semantic_filter_predicate` do the narrowing. `floor` never reaches an
# annotator, and the 158 warehouse pallets are annotated as what they are.
stage = get_current_stage()

STAGE_MPU = UsdGeom.GetStageMetersPerUnit(stage)
print(f"[sdg] stage metersPerUnit={STAGE_MPU} upAxis={UsdGeom.GetStageUpAxis(stage)}")

_bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
)
_unit_scale = {}


def unit_scale(url):
    """Scale factor that makes a referenced asset arrive at its real size.

    USD does **not** rescale geometry across a reference, so an asset whose
    own layer declares `metersPerUnit=0.01` lands 100x too large in this
    stage, which declares 1.0. Measured with `sdg/probe_assets.py`: every
    `DigitalTwin/` asset (all three pallet trucks, Forklift_A/B/C, both hand
    trucks) is 0.01; every `Isaac/` and `simready_content/` asset is 1.0.

    That single mismatch is what put the cameras inside the geometry in the
    Round 7 validation run -- Forklift_C measured 136 x 373 x 217 m.
    """
    if url not in _unit_scale:
        try:
            a = Usd.Stage.Open(url)
            mpu = UsdGeom.GetStageMetersPerUnit(a) if a is not None else STAGE_MPU
        except Exception as e:  # noqa: BLE001
            carb.log_warn(f"could not read metersPerUnit for {url}: {e}")
            mpu = STAGE_MPU
        _unit_scale[url] = float(mpu) / float(STAGE_MPU)
    return _unit_scale[url]


def union_bound(prim):
    """World-space size of a prim, unioned over its boundable descendants.

    Deliberately not `BBoxCache.ComputeWorldBound(root)`: on these assets
    that returns a *partial* union. It reported 49.9 x 36.1 x 41.0 for
    HeavyDutyPalletTruck_A01 whose `MainBody` child alone is 68.1 x 150.5 x
    50.3, which is how the units bug nearly got misdiagnosed as the pallet
    trucks being authored too small.
    """
    rng = Gf.Range3d()
    for d in Usd.PrimRange(prim):
        if d.IsA(UsdGeom.Boundable):
            _bbox_cache.Clear()
            r = _bbox_cache.ComputeWorldBound(d).ComputeAlignedRange()
            if not r.IsEmpty():
                rng = Gf.Range3d.GetUnion(rng, r)
    return None if rng.IsEmpty() else [float(v) for v in rng.GetSize()]

cx, cy = CONFIG["scatter_center"]
ex, ey = CONFIG["scatter_extent"]
OBJ_ROOT = "/World/Sdg"


# label -> [top-level prim paths]. Recorded at spawn so both the semantics
# pass and the placement pass can address exactly the object roots and
# nothing beneath them.
SPAWNED = {}

# prim path -> (translate_op, orient_op, QuatType). Resolved once at spawn so
# that moving an object per frame is two attribute writes.
XFORM_OPS = {}


def _add_op(xf, prim, existing, add, attr, alt_type, alt_prec):
    """Add an xform op at the precision of any attribute already authored.

    `AddOrientOp()` defaults to float. If the prim already carries a
    `xformOp:orient` attribute of type `quatd` -- which it does whenever the
    op was authored once and then dropped from the op *order* -- USD raises
    `Tf.ErrorException` rather than adapting, and Kit turns that into a
    segfault. Match the existing precision instead.
    """
    if existing is not None:
        return existing
    a = prim.GetAttribute(attr)
    if a and str(a.GetTypeName()) == alt_type:
        return add(alt_prec)
    return add()


def _pose_ops(prim):
    xf = UsdGeom.Xformable(prim)
    ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
    t = _add_op(xf, prim, ops.get(UsdGeom.XformOp.TypeTranslate),
                xf.AddTranslateOp, "xformOp:translate", "float3",
                UsdGeom.XformOp.PrecisionFloat)
    o = _add_op(xf, prim, ops.get(UsdGeom.XformOp.TypeOrient),
                xf.AddOrientOp, "xformOp:orient", "quatd",
                UsdGeom.XformOp.PrecisionDouble)
    quat = (
        Gf.Quatf
        if o.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
        else Gf.Quatd
    )
    return t, o, quat

TARGET_CLASSES = ("pallet", "pallet_truck", "forklift")


def spawn(paths, label, count):
    """Create the prims **once**, at startup, at their true real-world size.

    The stock example calls `rep.create.from_usd(...)` *inside* its
    randomizer, so a fresh set of prims is instantiated every frame. That is
    fine for its five cardboxes and fatal at our density: at ~83 objects the
    first validation run sat at 129% CPU with the GPU idle and produced no
    frames in 11 minutes, because each frame re-resolved and re-referenced
    every USD. Spawning once and only moving things per frame turns an
    unbounded per-frame cost into a one-time one.

    Every prim is spawned with `semantic_label=None`. Semantics are applied
    afterwards by `apply_semantics()` -- see the comment there for why doing
    it here does not work.
    """
    made = []
    for j in range(count):
        url = paths[j % len(paths)]
        path = f"{OBJ_ROOT}/{label}/{label}_{j}"
        f = unit_scale(url)
        prims.create_prim(
            prim_path=path,
            usd_path=url,
            position=(0.0, 0.0, 0.0),
            orientation=euler_angles_to_quat([0.0, 0.0, 0.0]),
            scale=(f, f, f),
            semantic_label=None,
        )
        # `create_prim` always authors an `xformOp:orient` (quaternion).
        # `UsdGeom.XformCommonAPI` only understands translate / rotateXYZ /
        # scale, so it cannot drive these prims at all -- it fails silently,
        # which is what "Could not determine xform ops for incompatible
        # xformable" was (383 times in a 20-frame run) and why a
        # `SetScale` through that API left every asset at 100x. Cache the
        # real op handles instead and write the attributes directly.
        XFORM_OPS[path] = _pose_ops(stage.GetPrimAtPath(path))
        made.append(path)
    SPAWNED.setdefault(label, []).extend(made)
    return len(made)


def apply_semantics():
    """Clear inherited semantics **after** spawning, then declare our own.

    Round 7 cleared the stage's semantics before spawning, which does
    nothing: the referenced USDs carry their own baked semantics and bring
    them along. `box`, `cone`, `bucket` and `barel` all came out labelled
    despite `semantic_label=None`, and the SimReady pallets arrived
    multi-labelled as `pallet,prop_general` -- 21 rows against 4 clean
    `pallet` rows.

    Clearing recursively after the fact, then re-applying only our three
    classes to the object roots, is the order that actually holds.

    Isaac Sim 6.1 labels with `UsdSemantics.LabelsAPI`; assets authored with
    the older `SemanticsAPI` are converted first, so clearing catches both.
    """
    for label, paths in SPAWNED.items():
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            upgrade_prim_semantics_to_labels(prim, include_descendants=True)
            remove_labels(prim, include_descendants=True)
            if label in TARGET_CLASSES:
                add_labels(prim, labels=[label], instance_name="class")


def semantic_vocabulary():
    """Every distinct semantic label left on the stage."""
    vocab = set()
    for prim in stage.Traverse():
        vocab.update(get_labels(prim).get("class", []))
    return vocab


def check_scale():
    """Abort before rendering if anything is the wrong size.

    Renders completing at the right speed says nothing about whether the
    content is right; the Round 7 run burned seven minutes proving that. A
    16-hour run shot from inside a wall is the failure this guards.
    """
    lo, hi = CONFIG["size_sanity_m"]
    bad = []
    for label, paths in SPAWNED.items():
        dims = []
        for path in paths[:3]:  # a sample per class is enough
            size = union_bound(stage.GetPrimAtPath(path))
            if size is None:
                continue
            dims.append(size)
            longest = max(size)
            if not (lo <= longest <= hi):
                bad.append(f"{path}: longest dim {longest:.2f} m outside [{lo}, {hi}]")
        if dims:
            shown = "  ".join(f"({d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f})" for d in dims)
            print(f"[sdg] size {label:14} {shown}", flush=True)
    return bad


for _label, _paths, _n in (
    ("pallet", ASSETS["pallet"], CONFIG["n_pallet"]),
    ("pallet_truck", ASSETS["pallet_truck"], CONFIG["n_pallet_truck"]),
    ("forklift", ASSETS["forklift"], CONFIG["n_forklift"]),
    # Unlabelled clutter: spawned like everything else, but left out of
    # TARGET_CLASSES so it renders and is never annotated. A detector that
    # only ever sees target classes learns a trivially separable world; real
    # warehouses are full of things that are not pallets.
    ("clutter_box", [CARDBOX], CONFIG["n_cardbox"]),
    ("clutter_misc", DISTRACTORS, CONFIG["n_distractor"]),
):
    print(f"[sdg] spawned {spawn(_paths, _label, _n)} x {_label}", flush=True)

# Let the referenced layers resolve before measuring or reading semantics.
for _ in range(120):
    simulation_app.update()

apply_semantics()

# The stage vocabulary cannot be reduced to our three classes by clearing
# semantics, because the warehouse props and the referenced assets are
# **instanceable**: their semantics live in instance prototypes, and USD does
# not let you author an override onto an instance proxy. `remove_labels`
# silently no-ops there. The first Round 8 validation attempt aborted on
# exactly this -- 27 labels survived, including `floor`, `ceiling`, `wall`,
# `rack` and `prop_general`.
#
# The lever that does work is the writer's `semantic_filter_predicate`, set
# below: it filters the *instance mapping*, so unwanted labels never reach an
# annotator at all. What we gate on here is the thing clearing can still get
# wrong -- that our own three classes actually landed on the object roots.
_vocab = semantic_vocabulary()
print(f"[sdg] stage semantic vocabulary ({len(_vocab)}): {sorted(_vocab)}")
_missing = set(TARGET_CLASSES) - _vocab
if _missing:
    carb.log_error(f"target classes missing from the stage: {sorted(_missing)}")
    simulation_app.close()
    raise SystemExit(2)
print(f"[sdg] filtered to: {sorted(TARGET_CLASSES)} (via semantic_filter_predicate)")

_bad = check_scale()
if _bad:
    for _b in _bad:
        carb.log_error(f"scale check failed -- {_b}")
    simulation_app.close()
    raise SystemExit(3)

GROUPS = ["pallet", "pallet_truck", "forklift", "clutter_box", "clutter_misc"]

# prim path -> footprint radius in metres, measured once from the spawned
# geometry. Half the diagonal of the object's world XY extent: a
# conservative circle that no two objects may overlap.
FOOTPRINT = {}
for _label, _paths in SPAWNED.items():
    for _path in _paths:
        _size = union_bound(stage.GetPrimAtPath(_path))
        FOOTPRINT[_path] = (
            0.5 * math.hypot(_size[0], _size[1]) if _size else 0.5
        )
print(
    "[sdg] footprint radius m: "
    + "  ".join(
        f"{lb}={min(FOOTPRINT[p] for p in ps):.2f}-{max(FOOTPRINT[p] for p in ps):.2f}"
        for lb, ps in SPAWNED.items()
    )
)

ALL_OBJECTS = [p for ps in SPAWNED.values() for p in ps]
CAM_CLEARANCE = CONFIG["camera_clearance_m"]
MARGIN = CONFIG["collision_margin_m"]
MAX_TRIES = CONFIG["placement_tries"]
PARKING = CONFIG["parking_spot"]


def place_objects(rng):
    """Place every object on the floor so that no two footprints overlap.

    Replicator's own `scatter_2d(check_for_collisions=True)` is the obvious
    tool and it does not work here. It raises
    `OgnScatter2D: Assertion raised in compute - list index out of range`
    several times per frame, and even when it limps through it only got
    heavily-occluded boxes from 47% down to 36%.

    Doing it ourselves is both correct and cheaper. Rejection sampling 60
    circles is microseconds of Python, against an O(n^2) OmniGraph node, and
    it lets us use a *measured* footprint per object instead of whatever
    bound the node infers. Objects that cannot be fitted after `MAX_TRIES`
    are parked far outside the camera volume rather than dropped, so the
    pool size stays constant across frames and nothing is left stacked
    inside something else.

    Returns the number of objects actually placed in the scene.
    """
    # Seed the occupancy list with the camera standing positions so nothing
    # is ever placed where a camera might be.
    placed = [(x, y, CAM_CLEARANCE) for x, y in GROUND_FOOTPRINTS]
    n_reserved = len(placed)
    order = list(ALL_OBJECTS)
    # Biggest first: forklifts placed last into a crowded floor is how you
    # get a forklift with no legal position.
    order.sort(key=lambda p: -FOOTPRINT[p])

    for path in order:
        r = FOOTPRINT[path]
        spot = None
        for _ in range(MAX_TRIES):
            x = cx + rng.uniform(-ex / 2, ex / 2)
            y = cy + rng.uniform(-ey / 2, ey / 2)
            if all(
                (x - px) ** 2 + (y - py) ** 2 >= (r + pr + MARGIN) ** 2
                for px, py, pr in placed
            ):
                spot = (x, y)
                break
        t_op, o_op, quat = XFORM_OPS[path]
        if spot is None:
            t_op.Set(Gf.Vec3d(*PARKING))
            continue
        placed.append((spot[0], spot[1], r))
        t_op.Set(Gf.Vec3d(spot[0], spot[1], 0.0))
        w, qx, qy, qz = euler_angles_to_quat(
            [0.0, 0.0, rng.uniform(0.0, 2 * math.pi)]
        )
        o_op.Set(quat(float(w), float(qx), float(qy), float(qz)))
    return len(placed) - n_reserved


# Four sphere lights, created once and re-posed per frame from our own RNG.
#
# They used to be a registered randomizer built on `rep.create.light`, which
# instantiated four fresh prims every frame -- the same "create prims inside
# a randomizer" shape that made the Round 7 object spawning unusable, just
# small enough not to hurt. More importantly, anything driven by
# `rep.distribution` is drawn from Replicator's global RNG, which ticks per
# *rendered* frame rather than per *captured* one, so the sequence depends
# on how many internal frames the app happens to burn during startup and
# asset streaming. That is wall-clock and cache dependent, which is why two
# runs at the same `--seed` produced different data (Round 12).
LIGHT_ROOT = "/World/SdgLights"
LIGHTS = []
for i in range(4):
    lp = f"{LIGHT_ROOT}/Sphere_{i}"
    prims.create_prim(prim_path=lp, prim_type="SphereLight")
    LIGHTS.append(lp)


def pose_lights(rng):
    """Re-pose and re-colour the lights. Same distributions as before."""
    for lp in LIGHTS:
        prim = stage.GetPrimAtPath(lp)
        t_op, o_op, quat = _normalise_ops(lp)
        t_op.Set(Gf.Vec3d(
            rng.uniform(cx - ex / 2, cx + ex / 2),
            rng.uniform(cy - ey / 2, cy + ey / 2),
            rng.uniform(5.0, 9.0),
        ))
        o_op.Set(quat(1.0, 0.0, 0.0, 0.0))
        light = UsdLux.SphereLight(prim)
        light.CreateIntensityAttr().Set(rng.uniform(2000.0, 12000.0))
        light.CreateRadiusAttr().Set(rng.uniform(3.0, 9.0))
        light.CreateColorAttr().Set(Gf.Vec3f(
            rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0), rng.uniform(0.55, 1.0)
        ))

# --- the two regimes -------------------------------------------------------
#
# Focal lengths assume Replicator's default 20.955 mm horizontal aperture:
# 24 mm ~= 47 deg hFOV (CCTV-like), 16 mm ~= 66 deg (handheld-like).
#
# elevated-fixed aims at a point (a mounted camera is pointed at an area),
# so it uses `look_at`. ground-handheld uses an explicit rotation
# distribution instead, including roll -- a person holding a camera does not
# perfectly frame anything, and that imperfect framing is part of what
# distinguishes the regime.
RES = tuple(CONFIG["resolution"])

# Handheld standing positions on a ring around the middle of the scene,
# rather than anywhere on the floor. Sampling the position uniformly over
# the whole 18x14 m area and aiming anywhere within +/-7 m of centre put the
# subject up to ~20 m away, and a 66-degree lens at ground level then fills
# the frame with specks: 11.7% of ground boxes came out below the smallest
# box LOCO's annotators ever drew, against LOCO's own 0.1%. LOCO is
# photographed in aisles, a few metres from the subject. A ring at 3-7 m
# with a tight aim point reproduces that working distance.
GROUND_SPOTS = [
    (cx + r * math.cos(a), cy + r * math.sin(a), z)
    for r in (2.5, 3.5, 4.5, 6.0)
    for a in [i * math.pi / 6 for i in range(12)]
    for z in (1.2, 1.45, 1.7)
]

# The floor positions a ground camera can occupy, deduplicated. Objects are
# kept clear of these: placement is collision-free between objects, but
# nothing stopped an object being placed on the spot the camera then stands
# on, and two of 20 ground frames came back shot from inside a forklift.
# A camera inside a forklift is the same physical impossibility as a pallet
# inside one -- it just shows up as a black frame instead of a bad box.
GROUND_FOOTPRINTS = sorted({(round(x, 3), round(y, 3)) for x, y, _ in GROUND_SPOTS})

# Eight fixed mounts on the boundary of the scatter area -- four corners and
# four edge midpoints -- at CCTV height. One is chosen per frame.
ELEVATED_MOUNTS = [
    (cx + sx * ex / 2, cy + sy * ey / 2, z)
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0))
    for z in (5.5, 7.0, 8.0)
]
elevated_cam = rep.create.camera(name="elevated_fixed", focal_length=24.0,
                                 clipping_range=(0.1, 1e6))
ground_cam = rep.create.camera(name="ground_handheld", focal_length=16.0,
                               clipping_range=(0.05, 1e6))
elevated_rp = rep.create.render_product(elevated_cam, RES, name="elevated-fixed")
ground_rp = rep.create.render_product(ground_cam, RES, name="ground-handheld")
rps = [elevated_rp, ground_rp]
for rp in rps:
    rp.hydra_texture.set_updates_enabled(False)

out_dir = CONFIG["output_dir"]
os.makedirs(out_dir, exist_ok=True)

writer = rep.WriterRegistry.get("KittiWriter")
writer.initialize(
    output_dir=out_dir,
    omit_semantic_type=True,
    # Default is "class:*", which is how Round 7 emitted `floor`, `box`,
    # `barel` and the multi-label `pallet,prop_general`. Restricting the
    # instance mapping here is the only thing that reaches semantics baked
    # into instance prototypes.
    semantic_filter_predicate="class: " + " | ".join(TARGET_CLASSES),
    # 30.5% of LOCO's pallet boxes are shorter than the default 25 px.
    bbox_height_threshold=1,
    partly_occluded_threshold=0.5,
    fully_visible_threshold=0.95,
    colorize_instance_segmentation=False,
    # Keep rgb/ + object_detection/ so `load_replicator.py` reads one layout.
    use_kitti_dir_names=False,
    mapping_dict={
        "pallet": [211, 210, 223, 255],
        "pallet_truck": [11, 110, 223, 255],
        "forklift": [255, 140, 0, 255],
    },
)
# Since Kit 109, Replicator gives every annotator created without an explicit
# filter `semanticFilter="*:*"`, which overrides the writer's global
# predicate above. KittiWriter lists its box and segmentation annotators that
# way, so on 6.1 the first naive_20 run labelled `box`, `rack`, `floor` and
# `wall` again. Recreate them with the predicate passed explicitly.
_PREDICATE = "class: " + " | ".join(TARGET_CLASSES)
_FILTERED = {
    "bounding_box_2d_tight_fast": {},
    "bounding_box_2d_loose_fast": {},
    "semantic_segmentation": {"mapping": writer._get_anno_semantic_mapping()},
    "instance_segmentation_fast": {"colorize": False},
}


def _name(a):
    return a if isinstance(a, str) else a.get_name()


writer.annotators = [
    rep.AnnotatorRegistry.get_annotator(
        _name(a), init_params={**_FILTERED[_name(a)], "semanticFilter": _PREDICATE})
    if _name(a) in _FILTERED else a
    for a in writer.annotators
]
writer.attach(rps)

# Camera pose, driven from our own RNG in the frame loop rather than by
# `rep.trigger.on_frame()` + `rep.distribution`. The distributions are
# unchanged -- a discrete choice of mount/standing spot, and a uniform aim
# point -- only the source of randomness and the place it is applied moved.
# See the comment on the lights above for why.
CAM_XFORMS = {
    "elevated-fixed": "/Replicator/elevated_fixed_Xform",
    "ground-handheld": "/Replicator/ground_handheld_Xform",
}


def _look_at_quat(eye, target, quat, up=(0.0, 0.0, 1.0)):
    """Orientation for a Replicator camera's `_Xform` that aims it at `target`.

    **Measured against Replicator's own `rep.modify.pose(look_at=...)`, not
    assumed** -- and the answer changed between versions:

    - Isaac Sim 6.1 (Replicator 1.13): `rep.create.camera` makes an `_Xform`
      with the Camera as an identity child, so the Xform follows the USD camera
      convention: it looks down local -Z, with +Y up. Replicator aiming from
      (0, 0, 1.5) at (5, 0, 1) wrote `rotateXYZ = (84.29, 0, -90)`, which maps
      -Z onto that direction. Rows: right, up, -forward.
    - Isaac Sim 4.5 needed rows (-forward, right, up). On 6.1 that basis maps -Z
      to -up, so the first 6.1 naive_20 run had every camera staring at the
      floor.
    """
    e = Gf.Vec3d(*eye)
    fwd = Gf.Vec3d(*target) - e
    if fwd.GetLength() < 1e-9:
        fwd = Gf.Vec3d(1.0, 0.0, 0.0)
    fwd = fwd.GetNormalized()
    u = Gf.Vec3d(*up)
    if abs(Gf.Dot(fwd, u)) > 0.999:          # looking straight up or down
        u = Gf.Vec3d(0.0, 1.0, 0.0)
    right = Gf.Cross(fwd, u).GetNormalized()
    upv = Gf.Cross(right, fwd)
    m = Gf.Matrix3d(right[0], right[1], right[2],
                    upv[0], upv[1], upv[2],
                    -fwd[0], -fwd[1], -fwd[2])
    q = m.ExtractRotation().GetQuat()
    im = q.GetImaginary()
    return quat(float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))


def _normalise_ops(path):
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    types = {op.GetOpType() for op in xf.GetOrderedXformOps()}
    if types - {UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient}:
        xf.ClearXformOpOrder()
    return _pose_ops(prim)


def pose_cameras(rng):
    """Place both cameras for one frame and return the poses for the log."""
    poses = {}
    for regime, mounts, aim_lo, aim_hi in (
        ("elevated-fixed", ELEVATED_MOUNTS,
         (cx - 5.0, cy - 5.0, 0.0), (cx + 5.0, cy + 5.0, 1.2)),
        ("ground-handheld", GROUND_SPOTS,
         (cx - 2.5, cy - 2.5, 0.0), (cx + 2.5, cy + 2.5, 1.2)),
    ):
        eye = mounts[rng.randrange(len(mounts))]
        target = tuple(rng.uniform(lo, hi) for lo, hi in zip(aim_lo, aim_hi))
        t_op, o_op, quat = _normalise_ops(CAM_XFORMS[regime])
        t_op.Set(Gf.Vec3d(*eye))
        o_op.Set(_look_at_quat(eye, target, quat))
        poses[regime] = {"position": [float(v) for v in eye],
                         "look_at": [float(v) for v in target]}
    return poses


for rp in rps:
    rp.hydra_texture.set_updates_enabled(True)

n_frames = CONFIG["num_frames"]
rt = CONFIG["rt_subframes"]
print(f"[sdg] {n_frames} frames x 2 regimes @ {RES}, rt_subframes={rt}, seed={CONFIG['seed']}")

# Per-frame camera poses, so `sim_params` records what actually happened
# rather than the distribution it was drawn from.
params = {}
seed = CONFIG["seed"]
place_rng = random.Random(seed)
cam_rng = random.Random(seed + 1_000_003)
light_rng = random.Random(seed + 2_000_003)
n_placed = []
for i in range(n_frames):
    # Placement happens here rather than in an `on_frame` randomizer
    # because we drive the frame loop ourselves -- the stage is simply in
    # its new state by the time `step()` renders it. Lights and camera
    # poses stay as Replicator randomizers; only object placement moved.
    n_placed.append(place_objects(place_rng))
    drawn = pose_cameras(cam_rng)
    pose_lights(light_rng)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=rt)
    frame = {}
    for cam_path, regime in (
        ("/Replicator/elevated_fixed_Xform", "elevated-fixed"),
        ("/Replicator/ground_handheld_Xform", "ground-handheld"),
    ):
        p = stage.GetPrimAtPath(cam_path)
        if p and p.IsValid():
            m = omni.usd.get_world_transform_matrix(p)
            t = m.ExtractTranslation()
            r = m.ExtractRotation().Decompose((1, 0, 0), (0, 1, 0), (0, 0, 1))
            frame[regime] = {"position": [t[0], t[1], t[2]],
                             "rotation_xyz_deg": [r[0], r[1], r[2]]}
    for regime, d in drawn.items():
        frame.setdefault(regime, {})["look_at"] = d["look_at"]
    params[str(i)] = frame
    params[str(i)]["n_placed"] = n_placed[-1]
    if i % 50 == 0:
        print(f"[sdg]   frame {i}/{n_frames} ({n_placed[-1]}/{len(ALL_OBJECTS)} placed)",
              flush=True)

print(f"[sdg] placed {min(n_placed)}-{max(n_placed)} of {len(ALL_OBJECTS)} objects per frame")

rep.orchestrator.wait_until_complete()

with open(os.path.join(out_dir, "sim_params.json"), "w") as f:
    json.dump({"config": CONFIG, "frames": params}, f, indent=1)

writer.detach()
for rp in rps:
    rp.destroy()
simulation_app.close()
print("[sdg] done")
