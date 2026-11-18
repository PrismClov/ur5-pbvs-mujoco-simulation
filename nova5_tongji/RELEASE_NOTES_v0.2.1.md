# Nova5 Tongji Closed-Loop Recovery v0.2.1

## Release summary

This release keeps a conveyor-tracking controller alive across target loss and
recoverable kinematics failures. After a safe fault, the arm returns to the
joint pose captured at process startup, resets the target estimator, waits for a
fresh target lock, and resumes tracking without restarting the controller.

## Changes since v0.2.0

- Capture the startup joint pose before tracking begins.
- Return to that pose after target loss, stale vision, estimator prediction
  failure, Jacobian singularity, joint/workspace protection, or target-speed
  estimation failure.
- Use a quintic ServoJ return trajectory with zero endpoint slope:
  - maximum return speed `0.2 rad/s`;
  - maximum return acceleration `0.6 rad/s²`;
  - final position tolerance `0.5°`.
- Do not attempt a return after stale joint feedback or a rejected ServoJ
  command.
- Reset the 12D ESKF after return and require eight fresh accepted poses before
  starting another tracking cycle.
- Reset stale partial 12D initialization when a warmup sample produces an
  invalid speed estimate.
- Add equivalent native-EKF reset support for the 8D Tongji tracker.
- Add a raw safe-target `send_joint_target()` entry so a pre-limited return
  trajectory is not throttled by the moving-target speed mode.
- Add `closed_loop` and `target_loss_return` configuration sections.
- Record recovery events, cycle counts, trigger reasons, and return results in
  the final JSON report.

## Validation

- 66 automated tests pass.
- A timed no-target run entered continuous wait mode and exited successfully at
  the configured duration instead of reporting a startup target failure.
- A live staged-target test exercised two target interruptions:
  - `cycles_started=3`;
  - `recovery_events=2`;
  - both returns succeeded;
  - tracking restarted after fresh EKF warmup;
  - final process report had `error=null`.
- The previously observed warmup failure
  `estimated target speed outside device configuration` is now discarded and
  retried rather than terminating the closed loop.

## Long-running start

Use `--duration 0` while the conveyor remains in operation. The process exits
only on `Ctrl+C`, SIGTERM, or an unrecoverable servo/feedback failure.

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

## Known limitations

- With a continuously moving single marker, the controller waits indefinitely
  after returning until a fresh target lock is available.
- A target remaining visible but continuously outside the reachable/safe
  kinematic region can cause repeated recovery cycles.
- Servo rejection and stale joint feedback remain fail-stop conditions and do
  not trigger automated motion.
- An independent hardware emergency stop remains mandatory.
