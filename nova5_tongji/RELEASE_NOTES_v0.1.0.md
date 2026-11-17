# Nova5 Tongji Real-Time Tracking v0.1.0

## First real-time tracking implementation

This release is the first working implementation of real-time visual tracking
for Nova5 using the Tongji controller stack. It connects the original Tongji
EKF and `auto_aim::Aimer` to an official Dobot Python SDK command loop and
tracks a 100 mm ArUco marker at a nominal 0.40 m camera distance.

## Included

- Original Tongji C++ EKF and original `auto_aim::Aimer` integration.
- Single-marker center tracking and optional marker-plane normal alignment.
- Nova5 six-axis Jacobian control with joint velocity, acceleration, and
  workspace limits.
- Fixed-rate 50 Hz direct Dobot `ServoJ` control.
- Continuous joint-command trajectory with encoder-noise attenuation.
- Near-zero dither suppression and a bounded target hold hysteresis.
- Immediate stop on stale vision, stale feedback, servo rejection, joint
  limit, workspace limit, or Jacobian singularity.
- Correct separation of Dobot joint feedback in degrees and radians.
- Offline configuration check and timestamped JSONL control logs.

## Validation

- 46 automated tests pass.
- Measured direct `ServoJ` scheduler: approximately 50.0 Hz with no missed
  control deadlines in the recorded live runs.
- The complete tracking loop has been exercised on the physical Nova5 and
  D435i setup.
- Sprint trajectories and correction probes exercised center recovery and
  marker-plane alignment.

## Known limitations

- Intermittent joint vibration can still occur during aggressive marker
  motion. Additional command and target filtering remains under validation.
- A 0.20 s recenter time has not been established as a stable acceptance
  result.
- Fast marker motion or a marker leaving the camera frame can cause a vision
  timeout and an intentional safe stop.
- The implementation does not provide scene collision checking. The operator
  must maintain an independent hardware emergency stop.
- The direct SDK path exclusively owns the Dobot dashboard and feedback ports;
  it must not run together with `dobot_bringup_v4`.

## Start

```bash
cd /home/clover/Desktop/ur5-pbvs-mujoco-simulation
source /opt/ros/humble/setup.bash
source /home/clover/ur_ws/install/setup.bash

bash nova5_tongji/scripts/run_tongji_controller.sh \
  --execute \
  --duration 60 \
  --output-log /home/clover/ur_ws/ibvs_rm65/runtime/tongji_servoj_release_v0.1.0.jsonl
```
