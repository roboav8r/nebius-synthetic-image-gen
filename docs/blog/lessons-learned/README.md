# Lessons-learned image archive

Frames kept for the blog post. Each folder is one lesson from [`docs/LESSONS.md`](../../LESSONS.md). The originals were overwritten by later runs, so these copies are the only ones left.

`edge` is the mean absolute difference between horizontally adjacent pixels (greyscale): about 1–2 for a clean frame, 14–19 for the speckled ones.

## 1. Speckled frames: Isaac Sim 4.5 on an RTX PRO 6000 Blackwell (`1-speckle-isaac45/`)
| File | Source | edge |
|---|---|---|
| `isaac45_elevated_frame0.png` | `naive_20`, elevated camera, frame 0 | 14.6 |
| `isaac45_ground_frame16.png` | `naive_20`, ground camera, frame 16 | 19.2 |

- **Source:** Nebius job `aijob-u00t7gcjngv8ftgajy` (2026-10-21), stock `nvcr.io/nvidia/isaac-sim` 4.5 image, GPU driver 580.173.02, Kit's default DLSS Performance (renders at 480×270, upscales to 960×540), 8 subframes.
- **Symptom:** every frame was covered in fine speckle and vertical streaks.
- **A misleading comparison:** the same scene rendered cleanly on a local RTX 4070 SUPER, but that machine ran Isaac 5.1. The comparison changed two things at once (GPU and Isaac version), so it couldn't isolate the cause.
- **Fix:** move to Isaac Sim 6.1 (section 2). On 6.1 every anti-aliasing mode is clean on the same GPU. The modes were never compared on 4.5, so "4.5 on Blackwell" is the culprit, but which part of 4.5 is unknown.

## 2. Clean frames: Isaac Sim 6.1, same GPU (`2-clean-isaac61/`)
| File | Anti-aliasing | edge |
|---|---|---|
| `isaac61_nebius_dlss_performance.png` | DLSS Performance (Kit default, half resolution) | 0.8 |
| `isaac61_nebius_dlaa.png` | DLAA (full resolution; now the project default) | 1.2 |

- **Source:** Nebius probe job `aijob-u00szctypv7acyfzjk` (2026-10-26), our image built on `isaac-sim:6.1.0` and running as root, on the same Blackwell GPU and 580 driver.
- `probe_result.txt` has the scores for all four anti-aliasing modes; every one was clean.
- **Reading the scores:** DLAA scores slightly *higher* because it is sharper (wall brick texture, floor scuffs), not noisier.

## 3. Cameras pointed at the floor: Isaac Sim 6.1 before the camera fix (`3-camera-pointed-down/`)
- `contact_ground_handheld.jpg`: the "handheld" camera should look across the warehouse at eye level. Every frame is a close-up of the floor.
- `contact_elevated_fixed.jpg`: the CCTV-style camera looks almost straight down, often rolled about 90°.
- `ground_frame0.png` and `elevated_frame0.png`: single full-resolution frames.
- `validation.txt`: the quality gate's verdict.
  - It **passed** the label vocabulary (the label filter fix worked).
  - It failed on 70% empty elevated frames and 95% empty ground frames. That failure came from the cameras, not from the sparse scene: the gate reported the symptom, not the cause.
- **Source:** Nebius job `aijob-u00heaqzjmwcpawa7y` (2026-10-26), `naive_20`.
- **Cause:**
  - The generator aims cameras with its own look-at quaternion. Its basis, rows (−forward, right, up), had been measured against Isaac 4.5's Replicator.
  - In 6.1, `rep.create.camera` builds an Xform that follows the USD camera convention: look down −Z, +Y up.
  - Under that convention the old basis maps the view direction to straight down.
- **Fix:** rows (right, up, −forward). This matches Replicator's own `look_at` rotation exactly: `rotateXYZ (84.29, 0, −90)` for an eye at (0, 0, 1.5) aimed at (5, 0, 1).

## 4. After the camera fix (`4-camera-fixed/`)
`local_3frames_ground_top_elevated_bottom.jpg` is a 3-frame local render in the 6.1 container (RTX 4070 SUPER).
- **Top row (ground):** eye-level views with the horizon, racks and walls.
- **Bottom row (elevated):** upright views looking down at the scatter area.
- **Labels:** `pallet` 17, `forklift` 2, `pallet_truck` 1.

The lessons themselves, with the rest from the migration, are in [`docs/LESSONS.md`](../../LESSONS.md).
