# Nova5 Tongji Jitter-Free Real-Time Tracking v0.1.1

## Operator-confirmed jitter-free tracking build

This release stabilizes the first Nova5/Tongji real-time tracking implementation.
The operator confirmed that the latest tracking run no longer exhibits the
previously audible joint jitter.

## Changes since v0.1.0

- Use a continuous joint-command trajectory instead of rebuilding every
  `ServoJ` target from noisy encoder feedback.
- Lock a fixed joint target while inside the visual hold hysteresis.
- Suppress near-zero joint-command dither and require confirmation before
  accepting a small velocity-direction reversal.
- Remove unnecessary marker roll following while retaining center and
  marker-plane normal alignment.
- Low-pass filter the estimated marker-plane normal.
- Use native Dobot feedback fields for diagnosis: controller timestamp/runtime,
  `QTarget`, `QDTarget`, `QDDTarget`, `QDActual`, target TCP, and TCP speeds.
- Tolerate occasional ROS pose-arrival gaps up to 250 ms only while warming the
  EKF; runtime vision remains limited to 150 ms.
- Add the complete three-terminal camera, tracker, and controller startup flow.

## Validation

- 46 automated tests pass.
- Direct `ServoJ` control remained approximately 50 Hz with no missed control
  deadlines in recorded runs.
- Physical Nova5 and D435i tracking was exercised.
- The operator confirmed the latest build was no longer audibly jittering.

## Known limitations

- A 0.20 s recenter time has not been established as a stable acceptance
  result.
- Fast marker motion or a marker leaving the camera frame causes an intentional
  stale-vision stop.
- No scene collision checker is included; an independent hardware emergency
  stop remains mandatory.
- The direct SDK path exclusively owns the Dobot dashboard and feedback ports
  and must not run with `dobot_bringup_v4`.

## Start

See the complete three-terminal workflow in
[`README.md`](README.md#完整启动流程), or start the controller after the camera
and tracker are already running:

```bash
cd /home/clover/Desktop/ur5-pbvs-mujoco-simulation
source /opt/ros/humble/setup.bash
source /home/clover/ur_ws/install/setup.bash

bash nova5_tongji/scripts/run_tongji_controller.sh \
  --execute \
  --duration 60 \
  --output-log /home/clover/ur_ws/ibvs_rm65/runtime/tongji_servoj_release_v0.1.1.jsonl
```
