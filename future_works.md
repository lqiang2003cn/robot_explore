# Future work: X3Plus VLA — simulation to real deployment

This document captures the planned path from **MuJoCo + SmolVLA** (`vla_x3plus/`) to **real X3Plus hardware**, using **`ros2_stack/`** as the ROS 2 control and integration layer. It is not a training tutorial for LeRobot; see [Hugging Face LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla) for `lerobot-train` and dataset format.

---

## Current state (reference)

| Stack | Role today |
|-------|------------|
| **`vla_x3plus/`** | MuJoCo scene (`models/x3plus_mujoco.xml`), `X3PlusPickCubeEnv`, LeRobot policy inference (`src/pick_and_place.py`). Observation: `observation.images.front_cam` (256×256), `observation.state` (6 floats: 5 arm + gripper). Action: 6D normalized joint-position deltas into `env.step()`. |
| **`ros2_stack/`** | Franka Panda pick-and-place with MoveIt 2, `topic_based_ros2_control`, and Isaac Sim–style bridging via `/joint_command` and `/joint_states`. **No X3Plus packages or drivers** in-tree. |
| **`simulation/`** | Isaac Sim (Franka/Carter), separate from MuJoCo x3plus. |

Fine-tuning SmolVLA on your embodiment is required: `lerobot/smolvla_base` does not generalize to X3Plus out of the box.

---

## What *not* to use `ros2_stack` for

**Scripted demonstration collection in MuJoCo** should stay in **`vla_x3plus`** (pure Python + MuJoCo): same Python env as LeRobot datasets, no Panda kinematics, no Isaac bridge, no Jazzy/conda mixing.

Use **`ros2_stack`** when you are ready to **command the real arm**, add **MoveIt** for the real kinematic chain, and run **teleoperation / safety / trajectory execution** on the robot.

---

## Deployment strategy: data for the real robot

Choose one primary path (they can be combined):

### A — Real demonstrations only (most reliable)

- Teleoperate the physical X3Plus; record episodes in **LeRobot dataset format** (same feature shapes as sim: 6D state, 6D action, 256×256 RGB, task string).
- **Order of magnitude:** tens to low hundreds of successful episodes for a single tabletop pick task; more if you need lighting/background robustness.
- **ROS fit:** teleop nodes, joint state publishers, and camera publishers live naturally in `ros2_stack` (new packages); dataset writing can be a separate process (subprocess from conda env per repo rules).

### B — Sim pre-train + small real fine-tune

- Train or fine-tune on **heavily domain-randomized** MuJoCo (lighting, textures, camera pose noise, friction, cube variation, distractors).
- Then **fine-tune on 20–50 real episodes** to close the remaining gap.

### C — Mixed training

- Combine sim and real in one dataset (e.g. majority sim, minority real) if you need volume and distribution accuracy.

---

## Real deployment checklist (robotics engineering)

1. **Camera / vision alignment**  
   Match training views to deployment: same approximate mount, FOV, and resolution (256×256 crops/scales if needed). Recalibrate intrinsics/extrinsics if the real camera differs from `front_cam` in `x3plus_mujoco.xml`.

2. **Action interface**  
   The policy outputs the same semantic action as in `vla_x3plus/src/env.py`: normalized deltas applied to position actuators with per-joint limits. Map these to **real servo commands** (limits, gear ratios, units) in a thin adapter; retrain if the mapping changes discretely.

3. **Control rate / latency**  
   Sim uses `control_dt` (e.g. 20 Hz). SmolVLA inference may require **10–20 Hz** or buffered actions on edge GPUs; validate timing on target hardware.

4. **Safety**  
   Joint/workspace limits, velocity/torque caps, e-stop, and conservative clipping before sending commands to hardware.

---

## Using `ros2_stack` for real X3Plus: concrete additions

The existing stack demonstrates the **pattern** you want to replicate for X3Plus:

- **`topic_based_ros2_control`**: `ros2_control` talks to the plant via **`/joint_command`** (out) and **`/joint_states`** (in). See `ros2_stack/ws/src/panda_pick_place/config/panda.ros2_control.xacro` and upstream docs for `topic_based_ros2_control`.
- **MoveIt 2 + trajectory execution**: `pick_place_node` uses MoveGroup actions; controllers are wired in `controllers.yaml` and MoveIt’s simple controller manager.

**Planned new work under `ros2_stack/ws/src/`** (names illustrative):

1. **`x3plus_description` (or reuse/extend `vla_x3plus/urdf/`)**  
   URDF + SRDF for MoveIt; ensure collision meshes and joint limits match the real arm.

2. **`x3plus_hardware` / driver package**  
   Node or `ros2_control` hardware interface that talks to the real servos (serial/USB/CAN per your hardware). Publishes `sensor_msgs/JointState` (or equivalent) and subscribes to the same command topic shape expected by `topic_based_ros2_control` **or** a dedicated hardware plugin.

3. **`x3plus_moveit_config`**  
   Kinematics, OMPL (or other) pipeline, joint limits, planning groups for the 5+1 DOF arm and gripper. No dependency on `moveit_resources_panda_moveit_config`.

4. **`x3plus_vla_bridge` (optional but useful)**  
   - Subscribes to camera (or uses shared memory / compressed image topics).  
   - Runs inference in a **subprocess** with `source activate_env.sh vla_x3plus` (per repo cross-component guidance) **or** exports ONNX/TorchScript if you later standardize on a C++ node.  
   - Publishes **trajectories or low-level joint targets** consistent with your training action space; ideally through `ros2_control` and the same safety limits as teleop.

5. **Launch files**  
   Mirror `panda_pick_place/launch/pick_place.launch.py`: robot state publisher, controller manager, MoveIt, then application nodes — but parameterized for X3Plus.

6. **Teleoperation package** (for Strategy A)  
   Gamepad, spacemouse, or leader arm → joint or Cartesian increments → record to LeRobot format via a small recorder node or external script.

**Reference parameters** for tabletop poses (not wired to ROS nodes today) live in `ros2_stack/config.yml` for the **Panda** demo; treat that file as a template only until you add X3Plus-specific YAML and code that reads it.

---

## Integration with `vla_x3plus` training artifacts

- **Checkpoint**: point `vla_x3plus/config.yml` `pretrained_path` (or `--policy`) at your fine-tuned Hub repo or local directory after `lerobot-train`.
- **Observation contract**: keep `observation.images.*` keys and `observation.state` layout aligned between dataset, training config, and the ROS bridge node.
- **Language**: use the same task strings at train and deploy time (e.g. `model.task` in `vla_x3plus/config.yml`).

---

## Summary

| Phase | Where it lives |
|-------|----------------|
| MuJoCo scripted / randomized demos, LeRobot dataset, SmolVLA fine-tune | `vla_x3plus/` (+ Hugging Face Hub for datasets/checkpoints) |
| Real drivers, MoveIt, teleop, safe execution, topic bridge to policies | `ros2_stack/ws/src/` (new X3Plus packages; reuse `topic_based_ros2_control` pattern where appropriate) |
| Cross-env inference | Prefer **subprocess** from ROS launch/scripts so conda (`vla_x3plus`) and ROS 2 Jazzy stay isolated |

Update this document when the first X3Plus package lands in `ros2_stack` so onboarding stays accurate.
