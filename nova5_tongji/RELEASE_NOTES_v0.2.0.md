# Nova5 Tongji Stable Dual-Mode Conveyor Tracking v0.2.0

## Release summary

This release adds full-pose 12D marker tracking with stationary/moving control
modes. The stationary mode suppresses residual searching and audible joint
impact, while the moving mode automatically raises speed and acceleration limits
when the EKF detects conveyor motion.

## Changes since v0.1.1

- Add the 12D full-pose ESKF tracking mode with four-axis-marker orientation
  tracking.
- Low-pass filter the complete marker orientation so a single-frame orientation
  jump is not amplified into flange translation through the camera lever arm.
- Tune the ESKF position/orientation uncertainty and process noise for a
  stationary 100 mm marker.
- Correct image-to-feedback timing compensation to 25 ms.
- Add hysteresis that decelerates before entering the stationary hold state.
- Add an 0.8 s zero-slope startup ramp.
- Add stationary and moving-target speed/acceleration limits:
  - stationary: `0.15 m/s`, `0.6 rad/s`, `2 rad/s²`;
  - moving: `0.30 m/s`, `1.1 rad/s`, `6 rad/s²`.
- Enter the moving mode when target speed reaches `0.025 m/s` or angular speed
  reaches `0.12 rad/s`; transition in over 0.35 s and return to the stationary
  mode automatically.
- Log `motion_boost_scale` and the effective limits in every control row.
- Add `restore_tongji_pose.py` for bounded, acceleration-continuous ServoJ
  return to a captured initial joint pose.
- Add `run_tongji_repeat_trials.sh` for fixed-start repeated testing, automatic
  post-trial restore, and retries after startup vision loss.

## Validation

- 60 automated tests pass.
- A three-trial fixed-start campaign completed with all valid trials returning
  to the captured baseline; the final two seconds of each completed trial had
  no joint target increment or commanded velocity.
- Post-trial restore errors were below `0.0002 rad` in recorded runs.
- A 10 s stationary dual-mode test kept `motion_boost_scale` at zero, confirming
  that visual estimation noise did not activate the moving mode.
- The updated smooth controller reduced a recorded first-500-ms TCP peak from
  approximately `350 mm/s` to `70 mm/s` before the moving-mode feature.

## Known limitations

- The moving conveyor mode is unit-tested and instrumented, but quantitative
  long-duration conveyor throughput remains to be validated on the physical
  setup.
- Marker loss, stale feedback, stale vision, servo rejection, joint limits,
  workspace limits, and Jacobian singularity retain fail-stop behavior.
- The direct SDK path exclusively owns the Dobot dashboard and feedback ports
  and must not run with `dobot_bringup_v4`.
- An independent hardware emergency stop remains mandatory.

## Start

See [`README.md`](README.md#完整启动流程), or start after the camera and tracker
are already running:

```bash
cd /home/clover/Desktop/ur5-pbvs-mujoco-simulation
source /opt/ros/humble/setup.bash
source /home/clover/ur_ws/install/setup.bash

bash nova5_tongji/scripts/run_tongji_controller.sh \
  --ekf 12d \
  --upright \
  --execute \
  --duration 60 \
  --observation-log /home/clover/ur_ws/ibvs_rm65/runtime/tongji_12d_observations.jsonl \
  --output-log /home/clover/ur_ws/ibvs_rm65/runtime/tongji_12d_realtime.jsonl
```
