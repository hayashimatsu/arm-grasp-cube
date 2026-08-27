# nhc12-grasp-demo

A robot arm in Isaac Sim locates a red cube on the table with its wrist-mounted
camera, works out which way the cube is turned, rotates the gripper to match,
grips it, and lifts it.

[日本語版 README](README.ja.md)

| | |
|---|---|
| Robot | Yaskawa Motoman NHC12 + Robotiq 2F-85 (one 12-DOF articulation) |
| Camera | Intel RealSense D455, mounted on the wrist |
| Scene | `scenes/arm310d_d455_ik_demo_r8.usd` |
| Target | Red 50 mm cube (`/World/object`) |

---

## 1. Quick start

### 1.1 Launch

1. Open `scenes/arm310d_d455_ik_demo_r8.usd` in Isaac Sim.
2. Press **Play**. Physics must be running.
3. Open `scripts/demo_start.py` in the **Script Editor** and press `Ctrl+Enter`.

`{'status': 'ready', ...}` in the console means the demo is loaded. IK following
is now active: dragging `/World/IKTarget` in the viewport moves the arm.

### 1.2 Run one grasp

Place the cube anywhere on the table, at any rotation. The only requirement is
that it appears in the left camera image.

```python
demo_run(lift_mm=100)
```

Progress for each stage is printed to the console. The run is finished when this
line appears:

```
demo_run: DONE  outcome=held  object_rise_mm=100.45  elapsed_s=41.2
```

Wait for that line before doing anything else. For the full merged result:

```python
demo_status()
```

### 1.3 Release and repeat

```python
demo_release()
```

The cube falls back to the table. Move it somewhere else, turn it to any angle,
and call `demo_run()` again.

---

## 2. Architecture

```
                    ┌─────────────────────┐
   D455 camera ────▶│  capture_*.py       │──▶ object position, size,
                    │  capture_approach.py│    and grasp axis
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    demo_api.py      │  ← the functions you call
                    └─────────────────────┘
                       │              │
              target pose             gripper angle
                       ▼              ▼
          ┌──────────────────┐  ┌───────────────────────┐
          │ /World/IKTarget  │  │ gripper_controller.py │
          │  position + yaw  │  └───────────────────────┘
          └──────────────────┘              │
                       │                    ▼
                       ▼            finger_joint (index 6)
          ┌──────────────────┐
          │ ik_controller.py │
          └──────────────────┘
                       │
                       ▼
              arm joints, index 0-5
```

Joint angles are never commanded directly. The demo moves a coordinate frame
called `/World/IKTarget`, and `ik_controller.py` drives the arm so that the tool
centre point follows it in both position and orientation. The gripper is on a
separate path and does not go through IK.

| File | Role |
|---|---|
| `scripts/demo_start.py` | Entry point. Loads every module and starts IK following |
| `scripts/demo_api.py` | Public API — every function below is defined here |
| `scripts/capture_d455.py` | Capture orchestration: render, save, assemble one run |
| `scripts/capture_approach.py` | Survey, half-step approach, precise measurement |
| `scripts/capture_segment.py` | Plane fitting and connected-region segmentation |
| `scripts/capture_geometry.py` | Camera intrinsics, projection, pose synthesis |
| `scripts/capture_annotate.py` | Annotated image rendering |
| `scripts/capture_result.py` | `result.json` / `diagnostics.json` assembly |
| `scripts/capture_run.py` | Run directories, hashing, quality gate |
| `scripts/yaw_estimate.py` | Grasp axis and shape class from the top face |
| `scripts/pose_math.py` | Quaternion composition and symmetry folding |
| `scripts/ik_controller.py` | IK following control (arm, index 0-5) |
| `scripts/gripper_controller.py` | Gripper angle command and observation (index 6) |
| `scripts/grasp_demo.py` | Grasp sequence and control-law constants |
| `outputs/captures/<run_id>/` | Images and measurements from every capture |

`yaw_estimate.py` and `pose_math.py` import nothing from Isaac Sim. They are pure
geometry and can be unit-tested with a plain `python3` interpreter.

---

## 3. How it works

### 3.1 Locating the object

The camera is on the wrist, so the visible area of the table depends on where the
arm is. `demo_capture()` runs three phases:

1. **Survey.** The arm moves to a high overview pose and takes one photo. At this
   phase a candidate is accepted on `height_above_plane_m` alone (0.040–0.060 m).
   Height above the table is unaffected by rotation and by partial cropping, so it
   remains usable when only part of the object is inside the frame.
2. **Half-step approach.** If the candidate is off-centre in the image, the camera
   moves half of the distance needed to centre it and takes another photo. This
   repeats up to `MAX_APPROACH_STEPS` times.
3. **Precise measurement.** With the object centred, the full criteria are applied:
   height, the horizontal-extent-to-height ratio (0.85–1.50), `touches_border`, and
   `center_reliability`.

The approach loop carries two safety properties. After each move, the object's
pixel distance from the image centre must decrease; if it does not decrease, or if
the candidate disappears, the arm returns to the last pose at which the object was
detected and the loop stops.

Depth pixels are back-projected into world coordinates, the tabletop plane is
fitted with RANSAC (the table sits at world z ≈ 0.595 m), pixels within 0.18 m of
the camera are excluded as the robot's own body, and the remaining points above the
plane are segmented into connected regions.

### 3.2 Working out which way the object is turned

A parallel-jaw gripper closes across an object. It must therefore align with the
object's **narrowest direction**, not with "the object's angle".

The top face of each region is projected onto the table plane, a convex hull is
taken, and a **minimum-area rectangle** is fitted with rotating calipers. The
short side of that rectangle is the direction the fingers must span:

```
grip_yaw_deg     = direction of the rectangle's short side
grip_width_mm    = length of the short side
symmetry_period  = 90° (square-like) / 180° (elongated) / none (round)
```

The symmetry period is derived from the measured rectangle, never assumed. A
square can be gripped from any of four equivalent directions; a rectangle has only
one correct family; a circle or disc has no preferred direction at all.

Principal component analysis is deliberately **not** used: a square's covariance
matrix is isotropic, so its eigenvector direction is numerical noise rather than
signal.

> **Theory.** The full derivation — plane filtering, convex hull, the rotating
> calipers theorem, the fill-ratio shape test, symmetry folding, and worked
> examples — is in [`docs/README-yaw-estimate.html`](docs/README-yaw-estimate.html)
> ([中文版](docs/README-yaw-estimate.zh.html) · [日本語版](docs/README-yaw-estimate.ja.html)).

### 3.3 Moving the arm

`demo_trace()` computes the target from the capture result:

```
IK_target = surface_centre + [0, 0, -(object_height / 2) - 0.0021]
```

`surface_centre` is the centre of the object's top face, so the target descends
half the object height to reach the centre of mass, then a further 2.1 mm. That
2.1 mm is a measured gripper geometry offset.

Orientation is the downward-facing quaternion rotated about the world Z axis by
`grip_yaw_deg`, folded into the object's own symmetry period so the wrist always
takes the shorter turn. Folding a 135° reading under 90° symmetry, for example,
gives −44.7° rather than a 135° rotation.

The yaw is applied only here, at the final move. All three capture phases run at a
fixed orientation, because the camera is mounted on the gripper: turning the wrist
during capture would turn the image with it and invalidate the half-step geometry.

### 3.4 Gripping

The Robotiq 2F-85 has six joints. Only `finger_joint` (articulation index 6) has a
drive; the other five carry PhysX mimic constraints with gearing ±1.0 that all
reference `finger_joint`. Commanding one joint moves the whole linkage.

- **0° is fully open, 47° is closed.** A larger angle means a smaller opening.
- The drive is a position drive with `stiffness = 3.0` and `max_force = 26 N·m`.

`demo_grasp()` commands the joint all the way to 47°. The cube stops the fingers at
roughly 21.5°, and the gap between the commanded and achieved angle builds motor
torque up to the force limit. That torque is the grip force. A rising
`tracking_error_rad` during closing indicates grip force rather than a fault.

---

## 4. Function reference

| Function | Behaviour |
|---|---|
| `demo_run(lift_mm=100.0)` | Runs capture, trace and grasp as one scheduled task. Prints progress per stage and a final `DONE` / `FAILED` line |
| `demo_capture()` | Survey, approach, measure. Result via `demo_status()` |
| `demo_trace(surface_center_xyz=None, thickness_m=None)` | Opens the gripper, rotates the wrist to the grasp axis, and moves the IK target to the object centre |
| `demo_grasp(lift_mm=100.0)` | Closes the gripper in steps, then lifts |
| `demo_release()` | Opens the gripper to 0° and clears the recorded trace target |
| `demo_status()` | `stage`, `running`, `ok`, `result`, `error` |
| `demo_gripper_angle(deg)` | Sets the finger joint directly, 0–47° |
| `demo_gripper_status()` | Current gripper angle and tracking error |

`demo_capture()`, `demo_trace()`, `demo_grasp()` and `demo_run()` schedule an async
task and return immediately. Isaac Sim's asyncio loop is driven by Kit's update
event, so blocking inside the Script Editor re-enters that loop and aborts the
scheduling round. `demo_run()` awaits each stage inside a single task, which is why
one call is enough.

### 4.1 Wrist alignment fields

`demo_trace()` reports how the wrist was rotated:

| Field | Meaning |
|---|---|
| `yaw_applied_deg` | The rotation actually applied, after symmetry folding |
| `yaw_source` | `capture` / `indeterminate` / `none` — see below |

| `yaw_source` | When | `yaw_applied_deg` |
|---|---|---|
| `capture` | Normal case, read from the capture result | The folded grasp axis |
| `indeterminate` | The object is round, so yaw is meaningless | `0.0` |
| `none` | The field could not be read at all | `0.0` |

`none` is always printed to the console. It is never a silent fallback.

### 4.2 Grasp outcomes

`demo_grasp()` reports an `outcome` field:

| Value | Meaning |
|---|---|
| `held` | The object rose with the arm, within 5 % of the commanded lift |
| `ejected` | The object rose more than the arm. The fingers pushed it out |
| `partial_slip` | The object rose 5–15 % less than the commanded lift |
| `dropped` | The object rose more than 15 % less than the commanded lift |
| `no_contact` | No contact was detected. The lift is skipped |

`slip_mm` is retained for backward compatibility. It uses an absolute value and
therefore cannot distinguish `ejected` from `dropped`; read `outcome` instead.

---

## 5. Errors

These are raised synchronously and appear as a traceback in the Script Editor.

| Message begins with | Cause | Action |
|---|---|---|
| `gripper not fully open` | The gripper still holds something | Call `demo_release()` |
| `most recent capture failed` | The last capture did not succeed | Call `demo_capture()` again |
| `no demo_trace() target recorded` | `demo_trace()` has not run, or `demo_release()` cleared it | Call `demo_trace()` |
| `IK target drifted from the traced pose` | The IK target moved after `demo_trace()` | Call `demo_trace()` again |
| `no cube candidate among N objects` | No region matched the criteria | Open the annotated image named in the message |

The drift message states whether the position or the orientation drifted, and by
how much. Failure messages name the annotated image and list every candidate with
the rule it failed. They do not contain coordinate ranges.

---

## 6. Captured images

Every `demo_capture()` creates `outputs/captures/<run_id>/`:

| File | Contents |
|---|---|
| `rgb_left.png`, `rgb_right.png`, `rgb_color.png` | Raw frames from the three D455 sensors |
| `rgb_left_annotated.png` | Detected regions, crosshair, surface centre |
| `region_overlay.png` | Segmentation overlay |
| `depth_preview.png`, `depth_left_annotated.png` | Depth visualisations |
| `depth_axial_left.npy`, `depth_radial_left.npy` | Raw float32 depth arrays |
| `result.json`, `diagnostics.json` | Measurements and diagnostic fields |

Each object in `result.json` carries `grip_yaw_deg`, `grip_width_mm`,
`object_length_mm`, `symmetry_period_deg`, `shape_class` and `fill_ratio`.
The older `yaw_deg_estimated` field is kept for compatibility; it is taken mod 90°
and cannot describe a non-square object, so read `grip_yaw_deg` instead.

`outputs/` is listed in `.gitignore`, so these files do not appear in `git status`
and are not present in a fresh clone. Open the directory directly. Set
`D455_DEMO_OUTPUT_DIR` to write elsewhere.

---

## 7. Scope and limits

**Supported.** A red 50 mm cube anywhere on the table that the left camera can
see, at any rotation about the vertical axis. The gripper aligns itself to the
cube before closing.

**Measured.** Eight blind trials, cube placed and rotated freely by the operator
without telling the system anything:

| | |
|---|---|
| Identified | 8 / 8 |
| Held | 8 / 8 |
| Wrist rotation applied | 8 / 8 |
| Worst grasp-axis error | 0.31° |
| Worst position error | 3.83 mm |

**Shape generality.** The grasp-axis algorithm is not specific to squares. It was
verified offline against synthetic squares, rectangles, circles, ellipses and an
irregular polygon, at six rotations each — 36 cases, including an 80 × 30 mm
rectangle where it correctly reports a grip width of 30 mm rather than 80 mm.
**Only the 50 mm cube has been tested on the robot**, because it is the only
object in the scene.

**Not supported.** Selecting among several objects, deriving the finger angle from
object width, placing the object down again, and objects of other sizes.

**Operating notes.**

Call `demo_release()` before the next `demo_run()`. A cube still held by the
gripper is not on the table, and the next capture will not find it.
