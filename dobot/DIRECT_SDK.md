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

The visual PBVS callback is not yet connected to this direct backend. The
benchmark above establishes the communication and scheduling layer first.
