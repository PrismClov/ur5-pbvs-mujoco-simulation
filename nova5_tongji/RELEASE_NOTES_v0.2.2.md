# Nova5 Tongji Controlled-Stop Recovery v0.2.2

## Release summary

This release restores the conservative `v0.2.1` base and adds an integrated
stop stage before returning to the captured startup pose. It also prevents
reacquisition of a target far outside the original tracking region.

## Changes since v0.2.1

- Keep the 50 Hz ServoJ stream active when vision becomes stale or an EKF
  callback reports a target-tracking failure.
- Enter a raised-cosine deceleration after a `0.06 s` visual-loss grace period.
- Wait until commanded and measured joint speeds are below `0.02 rad/s`.
- Hold the stopped position for `0.5 s` before starting the return trajectory.
- Use a maximum stop deceleration of `1.2 rad/s²` with zero acceleration at
  the beginning and end of the stop profile.
- Apply the same controlled deceleration for manual `Ctrl+C`/SIGTERM exit.
- Blend angular feedforward continuously between `0.25` and `0.45 rad/s`
  instead of switching it at a hard threshold.
- Limit tracking jerk to `12 rad/s³` in the stationary mode and `30 rad/s³`
  in the moving mode.
- Begin continuous speed reduction when the minimum Jacobian singular value
  falls below `0.18`, reaching zero at the hard `0.03` threshold.
- Use a slower hold-entry deceleration and require measured joint speed to
  settle before latching `motion_hold`.
- After automatic recovery, require a fresh target lock inside `0.15 m` of the
  initial target position before resuming motion.
- Preserve the existing smooth return to the captured startup joint pose.

## Validation

- 74 automated tests pass.
- Tests cover target-loss deceleration, stop-state confirmation, smooth
  hold-entry behavior, angular-feedforward continuity, Jacobian slowdown, and
  closed-loop reacquisition gating.
- Previous physical tests exercised repeated target loss, automatic return,
  EKF reset, and automatic reacquisition.

## Known limitations

- High-speed target loss can still produce audible servo or mechanism
  transients. The stop profile and speed limits reduce but do not fully remove
  this hardware behavior.
- A fault occurring while already close to a kinematic singularity may still
  shorten the available braking distance.
- The return trajectory currently completes or times out after it starts;
  interrupting it mid-trajectory remains a separate follow-up item.
- Servo rejection and stale joint feedback remain fail-stop conditions.
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
