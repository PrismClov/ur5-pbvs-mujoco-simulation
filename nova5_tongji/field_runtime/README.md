# Verified Nova5 field-runtime patches

These files capture the changes made to the separate, field-verified
`ur_ws/src/ibvs_ur5` runtime used for the Nova5 + D435i ArUco experiments.
They are kept separately because `tcp_bridge.py` intentionally hashes and
loads that external runtime.

## Contents

- `patches/real_tag_tracker.patch`: unique IDs for the four marker corners
- `patches/detector_utils.patch`: accepts the larger near-field marker image
- `patches/nova5_low_speed_executor.patch`: ArUco support, adaptive gain,
  target feedforward use, static recenter sprint, direct-pose ServoP, and
  planned-joint-speed limiting
- `config/nova5_fast_tracking.yaml`: 100 mm marker at a 400 mm standoff
- `config/nova5_aruco_center_040m.json`: centered desired image features
- `tests/test_adaptive_gain.py`: adaptive and joint-speed limit tests

## Apply

Apply from the runtime workspace root:

```bash
cd /home/clover/ur_ws
patch -p1 < /path/to/nova5_tongji/field_runtime/patches/real_tag_tracker.patch
patch -p1 < /path/to/nova5_tongji/field_runtime/patches/detector_utils.patch
patch -p1 < /path/to/nova5_tongji/field_runtime/patches/nova5_low_speed_executor.patch

cp /path/to/nova5_tongji/field_runtime/config/nova5_aruco_center_040m.json \
  /home/clover/ur_ws/ibvs_rm65/runtime/
cp /path/to/nova5_tongji/field_runtime/config/nova5_fast_tracking.yaml \
  /home/clover/ur_ws/ibvs_rm65/config/
```

Validate:

```bash
source /opt/ros/humble/setup.bash
source /home/clover/dobot_ws/install/setup.bash

cd /home/clover/ur_ws
python3 -m py_compile \
  src/ibvs_ur5/mujoco/pbvs/real_tag_tracker.py \
  src/ibvs_ur5/mujoco/pbvs/detector_utils.py \
  src/ibvs_ur5/mujoco/pbvs/nova5_low_speed_executor.py
python3 -m unittest discover \
  -s src/ibvs_ur5/tests -p 'test_adaptive_gain.py' -v
```

The marker size is exactly `0.100 m`. The detector's previous
`maxMarkerPerimeterRate=0.5` rejected the 100 mm marker when it filled more of
the 424x240 image after approaching the 400 mm standoff; the patch raises the
limit to `0.9`.

The field runs measured 50 Hz joint/TCP feedback, 30 Hz camera/detections, and
about 25 Hz active ServoP commands. The static 200 ms recenter target remains
unverified and should not be claimed as achieved.
