# Direct Dobot Python SDK

The vendor SDK under `dobot/sdk` can now run without the ROS2 Dobot driver and
without importing the optional LeRobot wrapper dependencies.

Install:

```bash
python3 -m pip install -r dobot/requirements.txt
```

Read-only feedback benchmark:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m dobot.sdk.feedback_benchmark \
  --ip 192.168.5.2 --read-hz 60 --duration 5
```

Fixed-rate ServoJ hold benchmark. This sends the current joint position and
does not command an intentional pose change:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m dobot.sdk.direct_benchmark \
  --execute --ip 192.168.5.2 \
  --feedback-fps 60 --control-hz 50 --duration 5 \
  --servo-t 0.02 --aheadtime 50 --gain 500
```

On 2026-11-17 the direct SDK measured:

- feedback: `59.995 Hz`
- control scheduler: `49.998 Hz`
- ServoJ calls: `250 / 5 s`
- command acknowledgements: `249 / 250`
- empty acknowledgements: `1` (initial command)
- acknowledgement order violations: `0`
- missed deadlines: `0`

`GetFeed` reads at `update_fps` but cannot create packets faster than the
controller source. `set_joints()` owns no timer; the fixed-rate scheduler in
`dobot/sdk/realtime.py` invokes it. `servo_t` remains the robot motion-time
parameter and is not the Python loop period.

## Direct visual tracking

`direct_tracking.py` connects `/tag_tracker/pose_camera` to the existing
ESKF/PBVS controller and sends its 50 Hz joint target through `set_joints()`.
It does not start `dobot_bringup_v4`.

```bash
PYTHONPATH="$PWD" .venv/bin/python -m dobot.direct_tracking \
  --execute --ip 192.168.5.2 \
  --control-hz 50 --feedback-fps 60 \
  --speed 20 --servo-t 0.02 --aheadtime 50 --gain 500 \
  --duration 30 \
  --output dobot/validation/direct_tracking_run.json
```

The D435i camera and `real_tag_tracker.py` must already be running. The node
waits until the ESKF is ready, then starts the fixed-rate loop. A missing or
stale target, joint/workspace violation, tracker fault, or loop exception
calls `Stop()` before disconnecting.
