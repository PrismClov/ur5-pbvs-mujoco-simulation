# Nova5 Tongji Vision Robustness v0.2.3

## Release summary

This release reduces false target-loss events caused by overly coarse ArUco
threshold scanning and short multi-frame detection gaps.

## Changes since v0.2.2

- Change the fast ArUco adaptive-threshold scan step from `10` to `4`.
  - Fine scan remains `2`.
  - This keeps the fast mode below the 30 Hz camera period while recovering
    weak or motion-blurred marker edges more reliably.
- Increase the visual-loss grace period before controlled deceleration from
  `0.06 s` to `0.12 s`.
  - At a 30 Hz camera rate this tolerates approximately three consecutive
    missed frames without declaring the target lost.
- Preserve the controlled deceleration, measured-velocity stop confirmation,
  zero-speed hold, return trajectory, and safe reacquisition-zone logic from
  `v0.2.2`.

## Validation

- 74 automated tests pass.
- Python compilation, JSON configuration validation, and patch syntax checks
  pass.
- The tracker was restarted from the installed workspace and confirmed to
  load `NOVA5_UPRIGHT_FAST_DETECT=1` with the new step-4 detector.

## Known limitations

- A marker that is not visible or is outside the camera field of view still
  counts as a real target loss.
- Very low exposure, severe motion blur, or incorrect camera calibration can
  still reduce detection reliability.
- The direct SDK path exclusively owns the Dobot dashboard and feedback ports.
- An independent hardware emergency stop remains mandatory.

## Start

See [`README.md`](README.md), or start after the camera and tracker are running:

```bash
cd /home/clover/Desktop/ur5-pbvs-mujoco-simulation
source /opt/ros/humble/setup.bash
source /home/clover/ur_ws/install/setup.bash

bash nova5_tongji/scripts/run_tongji_controller.sh \
  --ekf 12d \
  --upright \
  --execute \
  --duration 0 \
  --observation-log /home/clover/ur_ws/ibvs_rm65/runtime/tongji_12d_observations.jsonl \
  --output-log /home/clover/ur_ws/ibvs_rm65/runtime/tongji_12d_realtime.jsonl
```
