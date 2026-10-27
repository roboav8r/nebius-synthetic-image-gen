# Lessons learned

Each of these cost at least one failed job or a misleading result. The frames they refer to are in [`blog/lessons-learned/`](blog/lessons-learned/); the originals were overwritten by later runs.

## Nebius platform

- **Injected files are small.** All `--inject-file` contents of one job are encrypted together and must total 64 KiB or less. That is why the code ships in the image and only the resolved run config is injected.
- **The shared filesystem refuses non-root writers.** A non-root container user cannot write to it, even to a directory chowned to that user. The image runs as root, which also makes `HOME=/root`.
- **Logs look back 1 h by default.** `nebius ai job logs` hides older jobs' output unless given `--since`; `nj.logs()` passes `--since 72h`.
- **CLI login expires.** The next CLI call, including the `docker push` credential helper, then waits for a browser sign-in, and a background task can hang there silently. Run `nebius iam whoami` before long background work.
- **Capacity can run out.** `gpu-rtx6000` has been unavailable for 20+ minutes (`NotEnoughResources`, "VM schedule timeout"). The job ends in ERROR without running, and nothing is billed.
- **Pick a GPU with RT cores.** The job GPU is an RTX PRO 6000 Blackwell on driver 580.173.02. Isaac Sim 6.1 is tested by NVIDIA on 595 but works on 580. H100, H200 and B200 have no RT cores and cannot render.

## Isaac Sim and Kit

- **Isaac Sim 4.5 rendered speckled frames on Blackwell.** Every frame had fine speckle and vertical streaks (edge scores 14–19). On 6.1 every anti-aliasing mode is clean on the same GPU (edge 0.8–1.2). See [`1-speckle-isaac45/`](blog/lessons-learned/1-speckle-isaac45/) and [`2-clean-isaac61/`](blog/lessons-learned/2-clean-isaac61/).
- **Kit's default anti-aliasing is DLSS Performance,** which renders at half resolution and upscales. `configs/common.json` sets DLAA (`"/rtx/post/aa/op": 4`), which is sharper. DLAA's edge score is slightly higher because it is sharper, not noisier.
- **`SimulationApp.close()` ends the process.** It skips Python's normal exit, so buffered stdout is lost and exit codes after it are ignored. Print with `flush=True` and write results to a file before closing. When stdout went to a file, the probe's PASS line was lost and a good render reported FAIL.
- **Units are corrected twice in 6.1.** DigitalTwin assets are in centimetres, and USD does not rescale across a reference, so the generator scales each asset itself. 6.1's metrics assembler also adds a correction, but only to some assets, and one forklift came out 0.02 m long. The generator turns the assembler off (`/metricsAssembler/operationMode = 0`).
- **The semantics API moved in 6.1.** `add_update_semantics`, `get_semantics` and `remove_all_semantics` are gone. Use `add_labels`, `get_labels` and `remove_labels` from `isaacsim.core.utils.semantics`, calling `upgrade_prim_semantics_to_labels` first.
- **Annotators ignore the writer's label filter since Kit 109.** An annotator created without a filter gets `semanticFilter="*:*"`, which overrides KittiWriter's `semantic_filter_predicate`, so `box`, `rack`, `floor` and `wall` were labelled. The generator recreates the writer's box and segmentation annotators with the filter passed explicitly.
- **Camera aiming changed in 6.1.** `rep.create.camera` makes an Xform that follows the USD camera convention (looks down −Z, +Y up), so the look-at rotation's rows are (right, up, −forward). 4.5 needed (−forward, right, up); on 6.1 that basis pointed every camera at the floor, and the gate reported only the symptom, 70–95% empty frames. See [`3-camera-pointed-down/`](blog/lessons-learned/3-camera-pointed-down/) and the fix in [`4-camera-fixed/`](blog/lessons-learned/4-camera-fixed/). Check a contact sheet after any Isaac upgrade.
- **Asset paths are versioned.** Isaac assets resolve under `…/Assets/Isaac/6.1/`; `DigitalTwin/` and `simready_content/` are version-independent siblings of `Isaac/`.
- **KittiWriter's defaults are unchanged from 4.5:** `bbox_height_threshold=25`, occlusion thresholds 0.5 and 0.95. The generator lowers the height threshold to 1 px.

## Scene content

- **Spawn once, move per frame.** Creating prims inside a Replicator randomizer re-references every USD each frame; at ~80 objects the renderer sat idle and produced no frames in 11 minutes.
- **Clear an asset's baked labels after spawning, not before.** Referenced assets bring their own semantics (SimReady pallets arrive as `pallet,prop_general`).
- **Leave the warehouse's own labels alone.** It labels its 158 racked pallets correctly, and clearing them would leave unlabelled pallets in frame. USD cannot author over instance proxies anyway; the writer's filter is what keeps `floor` and `wall` out.
- **Use our own RNGs.** Replicator's distributions tick once per rendered frame, including the frames Kit burns while loading, so the same seed gave different data from run to run.
- **Keep objects off the camera's standing spots.** Without that, two of 20 ground frames were shot from inside a forklift.

## Diagnosing render quality

- **Score a frame by its mean absolute horizontal pixel difference** (`edge`). Clean frames score about 1–2; the speckled 4.5 frames scored 14–19. The probe prints this for each anti-aliasing mode.
- **Change one variable at a time.** A clean local render on an RTX 4070 SUPER suggested the GPU was at fault, but that machine ran Isaac 5.1, so the comparison changed two things at once. A local render on a different Isaac version proves nothing about the job's GPU.
