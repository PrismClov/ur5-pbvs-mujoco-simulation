"""Dobot CR5 simulation: no real Dobot TCP connection.

Joints are pure command-loopback; TCP pose comes from URDF FK. Inherits
configurable joint/pose/pose_delta action handling from the helpers in
:mod:`.dobot_cr5` so that real and simulated robots stay in sync without
code duplication.

``render=true`` opens a MuJoCo window that draws the CR5 URDF; it never owns
joint state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from .._observation_cameras import read_observation_cameras
from leapcollect.common.rot6d_pose import (
    apply_pose_delta,
    extract_pose_action,
    extract_pose_delta_action,
    format_pose_action,
    format_pose_delta_action,
    is_identity_pose_delta,
    pose_delta_between,
    pose_delta_keys,
    pose_keys,
)

from .dobot_cr5 import (
    DobotCR5RobotConfig,
    detect_action_kind,
    extract_joint_action,
    format_joint_action,
    forward_kinematics,
    inverse_kinematics,
)


@DobotCR5RobotConfig.register_subclass("dobot_cr5_sim")
@dataclass(kw_only=True)
class DobotCR5SimRobotConfig(DobotCR5RobotConfig):
    """字段与 :class:`DobotCR5RobotConfig` 相同；``ip`` 等网络项在仿真中忽略。

    ``gripper_type`` 为 ``damiao`` 时通过 DM-USB2FDCAN（motor-drive-layer v1.0）
    驱动真实达妙电机夹爪；其余类型仍为内存回环。需安装可选依赖 ``damiao-gripper``。
    Damiao 参数见基类 :class:`DobotCR5RobotConfig`。

    ``render=true`` 用 MuJoCo 画出 ``urdf_path``（运动学显示，无物理步进）。
    ``home_joints_deg`` is ``joint1``…``joint6`` in degrees, applied on connect.
    """

    render: bool = False
    home_joints_deg: list[float] = field(default_factory=lambda: [0.0] * 6)

    def __post_init__(self) -> None:
        super().__post_init__()
        vals = [float(x) for x in self.home_joints_deg]
        if len(vals) != 6:
            raise ValueError(f"home_joints_deg must have 6 values, got {len(vals)}")
        self.home_joints_deg = vals


class DobotCR5SimRobot(Robot):
    """Same action contract as :class:`DobotCR5Robot` but without real hardware."""

    config_class = DobotCR5SimRobotConfig
    name = "dobot_cr5_sim"
    _JOINT_COUNT = 6

    def __init__(self, config: DobotCR5SimRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._joints_rad: list[float] = list(np.deg2rad(config.home_joints_deg))
        self._gripper_pos: float = 0.0
        self._kin = None
        self._gripper = None
        self._viewer = None

    def _has_gripper(self) -> bool:
        t = self.config.gripper_type
        return t is not None and bool(str(t).strip())

    @property
    def _motors_ft(self) -> dict[str, type]:
        if self.config.observation_format == "joint":
            ft: dict[str, type] = {f"joint_{i}.pos": float for i in range(1, self._JOINT_COUNT + 1)}
        elif self.config.observation_format == "pose":
            ft = {k: float for k in pose_keys()}
        else:
            raise ValueError(f"Unknown observation_format: {self.config.observation_format}")
        if self._has_gripper():
            ft["gripper.pos"] = float
        return ft

    @property
    def _action_ft(self) -> dict[str, type]:
        fmt = self.config.action_format
        if fmt == "joint":
            ft: dict[str, type] = {
                f"joint_{i}.pos": float for i in range(1, self._JOINT_COUNT + 1)
            }
        elif fmt == "pose":
            ft = {k: float for k in pose_keys()}
        elif fmt == "pose_delta":
            ft = {k: float for k in pose_delta_keys()}
        else:
            raise ValueError(f"Unknown action_format: {fmt}")
        if self._has_gripper():
            ft["gripper.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._action_ft

    @property
    def extra_dataset_features(self) -> dict:
        """Additional named vector features saved alongside observation.state and action.

        Mirrors :meth:`DobotCR5Robot.extra_dataset_features` so that sim and real
        datasets share the same schema.  Requires urdf_path / RobotKinematics for
        the pose-based features.
        """
        joint_names = [f"joint_{i}.pos" for i in range(1, self._JOINT_COUNT + 1)]
        pose_names = list(pose_keys())
        pose_delta_names = list(pose_delta_keys())
        extra = {
            "observation.joint_pos": {
                "dtype": "float32",
                "shape": (self._JOINT_COUNT,),
                "names": joint_names,
            },
            "observation.pose": {
                "dtype": "float32",
                "shape": (len(pose_names),),
                "names": pose_names,
            },
            "action.joint_pos": {
                "dtype": "float32",
                "shape": (self._JOINT_COUNT,),
                "names": joint_names,
            },
            "action.pose": {
                "dtype": "float32",
                "shape": (len(pose_names),),
                "names": pose_names,
            },
            "action.pose_delta": {
                "dtype": "float32",
                "shape": (len(pose_delta_names),),
                "names": pose_delta_names,
            },
        }
        if self._has_gripper():
            extra["observation.gripper_pos"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper.pos"],
            }
            extra["action.gripper_pos"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper.pos"],
            }
        return extra

    @property
    def is_connected(self) -> bool:
        return all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True, *, gripper_dobot=None) -> None:
        if self.config.urdf_path and str(self.config.urdf_path).strip():
            from leapcollect.common.kinematics import RobotKinematics

            self._kin = RobotKinematics(
                urdf_path=str(self.config.urdf_path),
                target_frame_name=self.config.kin_target_frame,
                joint_names=self.config.kin_joint_names,
            )
        else:
            self._kin = None

        if self.config.render:
            urdf = str(self.config.urdf_path or "").strip()
            if not urdf:
                raise ValueError("dobot_cr5_sim render=true requires urdf_path")
            from leapcollect.robots.dobot.mujoco_view import DobotMujocoViewer

            self._viewer = DobotMujocoViewer(urdf, self._joints_rad)

        if gripper_dobot is not None:
            self._gripper = gripper_dobot
            self._gripper_pos = float(self._gripper.get_normalized_position())
        elif self.config._gripper_type_lower() == "damiao":
            from leapcollect.robots.dobot.sdk.damiao_gripper_dobot import DamiaoGripperDobot

            self._gripper = DamiaoGripperDobot.from_config(self.config.resolved_damiao_gripper())
            self._gripper_pos = float(self._gripper.get_normalized_position())

        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def disconnect(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._gripper is not None:
            if getattr(self._gripper, "_owns_bus", True):
                self._gripper.close()
            self._gripper = None
        self._kin = None
        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ---- state readers ----

    def _current_joints_rad(self) -> np.ndarray:
        return np.asarray(self._joints_rad, dtype=np.float64)

    def _current_pose(self) -> np.ndarray:
        if self._kin is None:
            raise RuntimeError(
                "dobot_cr5_sim requires urdf_path/RobotKinematics to compute TCP pose."
            )
        return forward_kinematics(self._kin, self._joints_rad)

    def _current_gripper_pos(self) -> float | None:
        if not self._has_gripper():
            return None
        if self._gripper is not None:
            return float(self._gripper.get_normalized_position())
        return float(self._gripper_pos)

    def _set_gripper_pos(self, gripper_pos: float) -> None:
        if self.config._gripper_type_lower() == "damiao" and self._gripper is None:
            raise ConnectionError(f"{self} external gripper is not connected.")
        if self._gripper is not None:
            hw = int(round(float(gripper_pos) * 1000.0))
            self._gripper.set_pos(hw)
            self._gripper_pos = float(self._gripper.get_normalized_position())
        else:
            self._gripper_pos = float(gripper_pos)

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Always include both joint and pose scalars so that the extra dataset
        # features (observation.joint_pos / observation.pose) can be assembled
        # by build_dataset_frame.  _current_pose() uses FK and requires _kin.
        joint_scalars = {f"joint_{i + 1}.pos": float(self._joints_rad[i]) for i in range(self._JOINT_COUNT)}
        tcp = self._current_pose()
        pose_scalars = {k: float(tcp[i]) for i, k in enumerate(pose_keys())}

        obs: dict = {**joint_scalars, **pose_scalars}
        gripper_pos = self._current_gripper_pos()
        if gripper_pos is not None:
            obs["gripper.pos"] = float(gripper_pos)

        obs.update(read_observation_cameras(self.cameras))

        return obs

    # ---- IK / FK delegating to module helpers ----

    def _inverse_kinematics(self, target_pose, seed_joints_rad) -> np.ndarray:
        return inverse_kinematics(
            self._kin,
            target_pose,
            seed_joints_rad,
            position_weight=self.config.kin_ik_position_weight,
            orientation_weight=self.config.kin_ik_orientation_weight,
            joint_count=self._JOINT_COUNT,
        )

    def _forward_kinematics(self, joints_rad) -> np.ndarray:
        return forward_kinematics(self._kin, joints_rad)

    # ---- send_action ----

    def _format_return_action(
        self,
        *,
        pre_action_pose: np.ndarray,
        sent_joints_rad: np.ndarray,
        sent_pose: np.ndarray,
        sent_gripper_pos: float | None = None,
    ) -> RobotAction:
        """Pure formatter: build the returned action dict from pre-computed values.

        Mirrors :meth:`DobotCR5Robot._format_return_action` so that sim and real
        robots return the same key schema.
        """
        d_xyz, d_orientation = pose_delta_between(pre_action_pose, sent_pose)
        out: dict = {
            **format_joint_action(sent_joints_rad, joint_count=self._JOINT_COUNT, gripper_pos=None),
            **format_pose_action(sent_pose, gripper_pos=None),
            **format_pose_delta_action(d_xyz, d_orientation, gripper_pos=None),
        }
        if sent_gripper_pos is not None:
            out["gripper.pos"] = float(sent_gripper_pos)
        return out

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        sent_gripper_pos = (
            float(action["gripper.pos"])
            if self._has_gripper() and "gripper.pos" in action
            else None
        )
        if sent_gripper_pos is not None:
            self._set_gripper_pos(sent_gripper_pos)

        kind = detect_action_kind(action, self._JOINT_COUNT)

        pre_action_pose: np.ndarray = self._current_pose()
        pre_action_joints = self._current_joints_rad()
        target_joints: np.ndarray

        if kind == "joint":
            target_joints = extract_joint_action(action, self._JOINT_COUNT)

        elif kind == "pose":
            target_pose = extract_pose_action(action)
            target_joints = self._inverse_kinematics(target_pose, pre_action_joints)

        elif kind == "pose_delta":
            delta_xyz, delta_orientation = extract_pose_delta_action(action)
            if is_identity_pose_delta(delta_xyz, delta_orientation):
                # No movement: sent joints == current joints, sent_pose == pre_action_pose.
                return self._format_return_action(
                    pre_action_pose=pre_action_pose,
                    sent_joints_rad=pre_action_joints,
                    sent_pose=pre_action_pose,
                    sent_gripper_pos=sent_gripper_pos,
                )
            target_pose = apply_pose_delta(pre_action_pose, delta_xyz, delta_orientation)
            target_joints = self._inverse_kinematics(target_pose, pre_action_joints)

        else:
            raise ValueError(
                f"Unsupported Dobot action kind: {kind!r}; expected one of "
                "'joint', 'pose', 'pose_delta'."
            )

        # Sim execution: directly write the planned joints into state.
        self._joints_rad = list(
            np.asarray(target_joints, dtype=np.float64).reshape(-1)[: self._JOINT_COUNT]
        )
        sent_arr = np.asarray(self._joints_rad, dtype=np.float64)
        sent_pose = self._forward_kinematics(sent_arr)
        if self._viewer is not None:
            self._viewer.set_joints(sent_arr)

        return self._format_return_action(
            pre_action_pose=pre_action_pose,
            sent_joints_rad=sent_arr,
            sent_pose=sent_pose,
            sent_gripper_pos=sent_gripper_pos,
        )
