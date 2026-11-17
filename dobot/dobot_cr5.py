"""Dobot CR5 LeRobot wrapper.

Owns robot kinematics (URDF / placo IK + FK), parses joint / joint_delta / pose /
pose_delta input actions, executes them via :class:`DobotInterface.set_joints`,
and returns the executed action in the configured representation
(joint / pose / pose_delta).
The SDK layer (:mod:`.sdk.interface`) is intentionally
left as a thin hardware adapter and does not know about LeRobot action schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal

import numpy as np
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import Robot, RobotConfig
from lerobot.types import RobotAction, RobotObservation

from leapcollect.common.euler_pose import euler_pose_to_rot6d_pose, pose_euler_to_quat
from leapcollect.common.rot6d_pose import (
    POSE_ORIENTATION_WIDTH,
    apply_pose_delta,
    extract_pose_action,
    extract_pose_delta_action,
    format_pose_action,
    format_pose_delta_action,
    matrix_to_pose,
    pose_delta_between,
    pose_delta_keys,
    pose_keys,
    pose_to_matrix,
)

from .._observation_cameras import read_observation_cameras
from ..gripper.damiao import DamiaoGripperConfig
from .sdk.interface import DobotInterface


ActionKind = Literal["joint", "joint_delta", "pose", "pose_delta"]
_ACTION_RETURN_FORMATS: tuple[str, ...] = ("joint", "pose", "pose_delta")
_OBSERVATION_RETURN_FORMATS: tuple[str, ...] = ("joint", "pose")


# ---- IK/FK (module-level for reuse by sim) ----


def inverse_kinematics(
    kin,
    target_pose,
    seed_joints_rad,
    *,
    position_weight: float,
    orientation_weight: float,
    joint_count: int,
) -> np.ndarray:
    """Solve IK for ``target_pose`` using ``kin`` (placo). Returns joints in radians."""
    if kin is None:
        raise RuntimeError("Cartesian control requires RobotKinematics (urdf_path).")
    n_j = len(kin.joint_names)
    seed_rad = np.asarray(seed_joints_rad, dtype=np.float64).reshape(-1).copy()
    if seed_rad.size < n_j:
        padded = np.zeros(n_j, dtype=np.float64)
        padded[: seed_rad.size] = seed_rad
        seed_rad = padded
    else:
        seed_rad = seed_rad[:n_j]

    T_tgt = pose_to_matrix(target_pose)
    joint_deg = kin.inverse_kinematics(
        np.rad2deg(seed_rad),
        T_tgt,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
    )
    jdeg = np.asarray(joint_deg, dtype=np.float64).reshape(-1)
    if jdeg.size < joint_count:
        raise ValueError(f"IK joint count {jdeg.size} < {joint_count}")
    return np.deg2rad(jdeg[:joint_count])


def forward_kinematics(kin, joints_rad) -> np.ndarray:
    """Forward kinematics: joints (rad) -> ``[x,y,z,rot6d(6)]``."""
    if kin is None:
        raise RuntimeError("Forward kinematics requires RobotKinematics (urdf_path).")
    n = len(kin.joint_names)
    deg = np.rad2deg(np.asarray(joints_rad, dtype=np.float64).reshape(-1))
    if deg.size < n:
        padded = np.zeros(n, dtype=np.float64)
        padded[: deg.size] = deg
        deg = padded
    else:
        deg = deg[:n]
    T = kin.forward_kinematics(deg)
    return matrix_to_pose(T)


# ---- Pure action parsing / formatting helpers ----


def _has_keys(action: RobotAction, keys) -> bool:
    return all(k in action for k in keys)


def detect_action_kind(action: RobotAction, joint_count: int) -> ActionKind:
    """Detect which action family is present; raises if zero or multiple are present."""
    joint_keys = tuple(f"joint_{i}.pos" for i in range(1, joint_count + 1))
    joint_delta_keys = tuple(f"joint_{i}.delta" for i in range(1, joint_count + 1))
    has_joint = _has_keys(action, joint_keys)
    has_joint_delta = _has_keys(action, joint_delta_keys)
    has_pose = _has_keys(action, pose_keys())
    has_pose_delta = _has_keys(action, pose_delta_keys())
    present = [
        name
        for name, ok in (
            ("joint", has_joint),
            ("joint_delta", has_joint_delta),
            ("pose", has_pose),
            ("pose_delta", has_pose_delta),
        )
        if ok
    ]
    if len(present) != 1:
        raise ValueError(
            "Dobot action must contain exactly one of "
            "joint/joint_delta/pose/pose_delta key sets, "
            f"got {present or 'none'}."
        )
    return present[0]  # type: ignore[return-value]


def extract_joint_action(action: RobotAction, joint_count: int) -> np.ndarray:
    return np.array(
        [float(action[f"joint_{i}.pos"]) for i in range(1, joint_count + 1)],
        dtype=np.float64,
    )


def extract_joint_delta_action(action: RobotAction, joint_count: int) -> np.ndarray:
    return np.array(
        [float(action[f"joint_{i}.delta"]) for i in range(1, joint_count + 1)],
        dtype=np.float64,
    )


def format_joint_action(
    sent_joints_rad,
    *,
    joint_count: int,
    gripper_pos: float | None = None,
) -> RobotAction:
    out: dict = {
        f"joint_{i + 1}.pos": float(sent_joints_rad[i]) for i in range(joint_count)
    }
    if gripper_pos is not None:
        out["gripper.pos"] = float(gripper_pos)
    return out


def _default_cameras() -> dict[str, CameraConfig]:
    return {
        "cam_1": OpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=480,
        ),
    }


@RobotConfig.register_subclass("dobot_cr5")
@dataclass(kw_only=True)
class DobotCR5RobotConfig(RobotConfig):
    """与 :class:`DobotInterface` 对齐的网络与伺服参数；夹爪与运动学可选。

    ``gripper_type`` 为 ``damiao`` 时通过 DM-USB2FDCAN（motor-drive-layer v1.0）
    驱动达妙电机夹爪，不经过 Dobot Modbus 夹爪接口；其余类型（``ag95`` / ``pge`` /
    ``do`` 等）由 :class:`DobotInterface` 处理。
    """

    ip: str = "192.168.5.1"
    dashboard_port: int = 29999
    feed_port_four: int = 30004
    speed: int = 20
    servo_t: float = 0.2
    aheadtime: int = 50
    gain: int = 500
    feedback_fps: float = 60.0
    gripper_type: str | None = None
    gripper_force: int = 80
    gripper_velocity: int = 500
    gripper_slave_id: int = 1
    gripper_parity: str = "N"
    gripper_do_index: int = 1
    tool_idx: int = 0
    urdf_path: str | None = None
    """URDF 路径；启用 placo IK/FK 后才能消费 ``pose`` / ``pose_delta`` 输入或输出。"""
    kin_target_frame: str = "eef"
    kin_joint_names: list[str] | None = None
    kin_ik_position_weight: float = 1.0
    kin_ik_orientation_weight: float = 0.01
    # NOTE: draccus (as used by LeRobot) can't always decode typing.Literal[...] from CLI.
    # Keep this as a plain str and validate explicitly.
    action_format: str = "joint"
    """``send_action()`` 返回值/数据集动作 schema：``joint``（默认）、``pose`` 或 ``pose_delta``。"""
    observation_format: str = "joint"
    """``get_observation()`` 电机状态 schema：``joint``（默认，关节弧度）或 ``pose``。"""
    execute_action: bool = True
    """为 True（默认）时正常下发关节指令；为 False 时跳过所有写操作（ServoJ / 夹爪），
    仅读取当前状态并返回——适用于 GELLO 调试或 dry-run 场景。"""
    max_joint_delta_deg: float = 20.0
    """单步最大允许关节变化量（度）。目标关节与当前关节之差超过此阈值的任意一轴时，
    跳过本次 ServoJ 并保持原位，防止标定偏差或通信异常导致的大幅跳变。
    设为 0 或负值可禁用此保护。"""
    safety_engaged_threshold_deg: float = 5.0
    """安全保护恢复内阈值（度）。从 HELD 状态回到 max_joint_delta_deg 以内后，
    进入 ENGAGING 阶段逐步逼近目标；当所有关节偏差降到此阈值以内时切换为
    NORMAL 直通模式。应小于 max_joint_delta_deg。"""
    safety_engage_max_step_deg: float = 3.0
    """ENGAGING 阶段每步最大关节步长（度）。控制从 HELD 恢复时的加速节奏，
    防止瞬间从零速跳到最大 ServoJ 速度。30 fps 下 3° ≈ 90°/s，
    低于 SDK 层 joint_command_max_velocity (默认 2 rad/s ≈ 114°/s) 以确保
    此参数为实际限速瓶颈。"""
    safety_cmd_gap_held_s: float = 1.0
    """距上次成功下发关节指令超过此秒数时，下一次指令从 HELD 起跳（再经 ENGAGING），
    避免遥操作长时间跳过 ``send_action`` 后突然直通大步长。≤0 关闭此逻辑。"""
    joint_command_max_velocity: float = 2.0
    """ServoJ 速度限幅（rad/s）；超速时对目标关节进行插值截断。"""
    joint_command_max_dt: float = 0.05
    """ServoJ 时间步长上限（秒）；超时后重置平滑器以防累积漂移。"""
    joint_command_smoothing_factor: float = 0.1
    """ServoJ 指令平滑系数（0–1）；越大越平滑（1 = 完全跟随上一帧位置），越小响应越快（0 = 直接跟踪目标）。"""
    home_pose: list[float] | None = None
    """固定 home 位姿 ``[x,y,z,rx,ry,rz]``（米 + sxyz 欧拉弧度）；用于 :meth:`go_home`。"""
    home_gripper_pos: float | None = None
    """可选 home 夹爪开合 ``[0,1]``；仅在存在夹爪时生效。"""
    home_timeout_s: float = 10.0
    """home 位姿到位超时时间（秒）。"""
    cameras: dict[str, CameraConfig] = field(default_factory=_default_cameras)
    damiao_gripper: DamiaoGripperConfig | None = None
    """Nested config when ``gripper_type: damiao`` (Damiao USB2CANFD / motor-drive-layer)."""

    def _gripper_type_lower(self) -> str | None:
        if self.gripper_type is None:
            return None
        s = str(self.gripper_type).strip()
        return s.lower() if s else None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_format not in _ACTION_RETURN_FORMATS:
            raise ValueError(
                f"robot.action_format must be one of {_ACTION_RETURN_FORMATS}, "
                f"got {self.action_format!r}."
            )
        if self.observation_format not in _OBSERVATION_RETURN_FORMATS:
            raise ValueError(
                f"robot.observation_format must be one of {_OBSERVATION_RETURN_FORMATS}, "
                f"got {self.observation_format!r}."
            )
        if self.home_pose is not None and len(self.home_pose) != 6:
            raise ValueError(
                "robot.home_pose must be a 6-value Euler pose [x,y,z,rx,ry,rz] in meters/radians."
            )
        if self.home_gripper_pos is not None and not (0.0 <= float(self.home_gripper_pos) <= 1.0):
            raise ValueError("robot.home_gripper_pos must be in [0,1].")
        if self._gripper_type_lower() == "damiao":
            self.resolved_damiao_gripper()

    def resolved_damiao_gripper(self) -> DamiaoGripperConfig:
        if self.damiao_gripper is not None:
            dg = self.damiao_gripper
            if isinstance(dg, dict):
                return DamiaoGripperConfig(**dg)
            return dg
        return DamiaoGripperConfig()


class DobotCR5Robot(Robot):
    """6 轴关节（弧度）+ 可选夹爪（gripper.pos 为 0–1）+ 相机；支持 joint/joint_delta/pose/pose_delta 动作。"""

    config_class = DobotCR5RobotConfig
    name = "dobot_cr5"
    _JOINT_COUNT = 6
    _SAFETY_HOLD_LOG_INTERVAL_S = 1.0

    def __init__(self, config: DobotCR5RobotConfig):
        super().__init__(config)
        self.config = config
        self._dobot: DobotInterface | None = None
        self._kin = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._cached_tcp_pose: np.ndarray | None = None
        self._cached_tcp_pose_ts: float | None = None
        self._tcp_pose_cache_ttl_s: float = 1.0
        self._cached_joints_rad: np.ndarray | None = None
        self._cached_joints_ts: float | None = None
        self._joints_cache_ttl_s: float = 1.0
        self._last_safety_hold_log_ts: float | None = None
        self._last_joint_cmd_ts: float | None = None
        self._safety_state: Literal["normal", "held", "engaging"] = "normal"

    def _has_gripper(self) -> bool:
        t = self.config.gripper_type
        return t is not None and bool(str(t).strip())

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor/TCP features used for observation features; format set by ``observation_format``."""
        fmt = self.config.observation_format
        if fmt == "joint":
            ft: dict[str, type] = {f"joint_{i}.pos": float for i in range(1, self._JOINT_COUNT + 1)}
        elif fmt == "pose":
            ft = {k: float for k in pose_keys()}
        else:
            raise ValueError(f"Unknown observation_format: {fmt}")
        if self._has_gripper():
            ft["gripper.pos"] = float
        return ft

    @property
    def _action_ft(self) -> dict[str, type]:
        """Action features depend on configured ``action_format``."""
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

        These are assembled by build_dataset_frame using the same scalar key names
        that get_observation() and send_action() always return, regardless of the
        configured observation_format / action_format.

        Requires forward kinematics (_kin / urdf_path) for the action.pose and
        action.pose_delta features.
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
        if self._dobot is None:
            return False
        return all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True, *, gripper_dobot=None) -> None:
        if self._dobot is not None:
            return

        damiao_cfg = (
            self.config.resolved_damiao_gripper()
            if self.config._gripper_type_lower() == "damiao" and gripper_dobot is None
            else None
        )
        self._dobot = DobotInterface(
            dashboardPort=self.config.dashboard_port,
            feedPortFour=self.config.feed_port_four,
            ip=self.config.ip,
            speed=self.config.speed,
            t=self.config.servo_t,
            aheadtime=self.config.aheadtime,
            gain=self.config.gain,
            update_fps=self.config.feedback_fps,
            gripper_type=self.config.gripper_type,
            gripper_force=self.config.gripper_force,
            gripper_velocity=self.config.gripper_velocity,
            gripper_slave_id=self.config.gripper_slave_id,
            gripper_parity=self.config.gripper_parity,
            gripper_do_index=self.config.gripper_do_index,
            tool_idx=self.config.tool_idx,
            damiao_gripper=damiao_cfg,
            gripper_dobot=gripper_dobot,
            joint_command_max_velocity=self.config.joint_command_max_velocity,
            joint_command_max_dt=self.config.joint_command_max_dt,
            joint_command_smoothing_factor=self.config.joint_command_smoothing_factor,
        )
        # Reset pose / joint caches on (re)connect.
        self._cached_tcp_pose = None
        self._cached_tcp_pose_ts = None
        self._cached_joints_rad = None
        self._cached_joints_ts = None

        # Robot owns kinematics; SDK layer no longer loads URDF.
        if self.config.urdf_path and str(self.config.urdf_path).strip():
            from leapcollect.common.kinematics import RobotKinematics

            self._kin = RobotKinematics(
                urdf_path=str(self.config.urdf_path),
                target_frame_name=self.config.kin_target_frame,
                joint_names=self.config.kin_joint_names,
            )
        else:
            self._kin = None

        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def disconnect(self) -> None:
        if self._dobot is not None:
            self._dobot.disconnect()
            self._dobot = None
        self._kin = None
        self._cached_tcp_pose = None
        self._cached_tcp_pose_ts = None
        self._cached_joints_rad = None
        self._cached_joints_ts = None
        self._last_safety_hold_log_ts = None
        self._last_joint_cmd_ts = None
        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected or self._dobot is None:
            raise ConnectionError(f"{self} is not connected.")

        fmt = self.config.observation_format
        obs: dict = {}

        # Always read both joint and pose scalars regardless of observation_format
        # so extra dataset features (observation.joint_pos / observation.pose) are available.
        joints = self._dobot.get_joint_position()
        if joints is None or len(joints) < self._JOINT_COUNT:
            raise RuntimeError(f"Dobot 关节反馈不可用或长度不是 {self._JOINT_COUNT}。")
        tcp_euler = self._dobot.get_end_pose()
        if tcp_euler is None or len(tcp_euler) < 6:
            raise RuntimeError("Dobot TCP feedback unavailable.")
        tcp = euler_pose_to_rot6d_pose(tcp_euler[:6])

        joint_scalars = {f"joint_{i + 1}.pos": float(joints[i]) for i in range(self._JOINT_COUNT)}
        pose_scalars = {k: float(tcp[i]) for i, k in enumerate(pose_keys())}

        if fmt == "joint":
            obs = {**joint_scalars, **pose_scalars}
        elif fmt == "pose":
            # pose keys are the primary state; joint keys come second so they
            # don't shadow the primary keys in the merged dict.
            obs = {**pose_scalars, **joint_scalars}
        else:
            raise ValueError(f"Unknown observation_format: {fmt}")

        gripper_pos = self._current_gripper_pos()
        if gripper_pos is not None:
            obs["gripper.pos"] = float(gripper_pos)

        obs.update(read_observation_cameras(self.cameras))

        return obs

    # ---- robot state readers ----

    def _current_joints_rad(self) -> np.ndarray:
        if self._dobot is None:
            raise ConnectionError(f"{self} is not connected.")
        now = time.monotonic()
        if (
            self._cached_joints_rad is not None
            and self._cached_joints_ts is not None
            and (now - self._cached_joints_ts) <= self._joints_cache_ttl_s
        ):
            return self._cached_joints_rad.copy()

        curr = self._dobot.get_joint_position()
        if curr is None or len(curr) < self._JOINT_COUNT:
            raise RuntimeError("Dobot joint feedback unavailable.")
        joints = np.asarray(curr[: self._JOINT_COUNT], dtype=np.float64)
        self._cached_joints_rad = joints
        self._cached_joints_ts = now
        return joints.copy()

    def _current_pose(self) -> np.ndarray:
        if self._dobot is None:
            raise ConnectionError(f"{self} is not connected.")
        now = time.monotonic()
        if (
            self._cached_tcp_pose is not None
            and self._cached_tcp_pose_ts is not None
            and (now - self._cached_tcp_pose_ts) <= self._tcp_pose_cache_ttl_s
        ):
            return self._cached_tcp_pose.copy()

        tcp_euler = self._dobot.get_end_pose()
        if tcp_euler is None or len(tcp_euler) < 6:
            raise RuntimeError("Dobot TCP feedback unavailable.")
        pose = euler_pose_to_rot6d_pose(tcp_euler[:6])
        self._cached_tcp_pose = pose
        self._cached_tcp_pose_ts = now
        return pose.copy()

    def _current_gripper_pos(self) -> float | None:
        if not self._has_gripper() or self._dobot is None:
            return None
        gp = self._dobot.get_gripper_position()
        return float(gp) if gp is not None else 0.0

    def get_ee_force(self) -> np.ndarray | None:
        """TCP wrench from Dobot feed: ``[fx, fy, fz, frx, fry, frz]`` (N, N·m).

        Returns ``None`` if disconnected or force feedback is unavailable.
        """
        if not self.is_connected or self._dobot is None:
            return None
        force = self._dobot.get_force()
        if force is None:
            return None
        return np.asarray(force, dtype=np.float64)

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

    # ---- Safety gate (three-state: NORMAL / HELD / ENGAGING) ----

    def _note_joint_cmd_sent(self) -> None:
        self._last_joint_cmd_ts = time.monotonic()

    def _set_joints_tracked(self, target_joints: np.ndarray) -> np.ndarray:
        sent = np.asarray(
            self._dobot.set_joints(target_joints), dtype=np.float64
        ).reshape(-1)[: self._JOINT_COUNT]
        self._note_joint_cmd_sent()
        return sent

    def _safety_gate_joints(
        self,
        target_arr: np.ndarray,
        current_arr: np.ndarray,
        target_joints: np.ndarray,
    ) -> np.ndarray:
        """Decide what to send to the servo based on a three-state safety machine.

        States
        ------
        NORMAL    – pass target through to ``set_joints`` unmodified.
        HELD      – target exceeds ``max_joint_delta_deg``; hold current position.
                    Also entered when no joint command was sent for longer than
                    ``safety_cmd_gap_held_s`` (resume after teleop idle / skip).
        ENGAGING  – recovering from HELD; clamp per-step motion to
                    ``safety_engage_max_step_deg`` until all deltas drop below
                    ``safety_engaged_threshold_deg``, then transition to NORMAL.

        Returns the joint array (rad) that was actually sent / held.
        """
        if np.allclose(target_arr, current_arr, rtol=0.0, atol=1e-9):
            return current_arr

        max_delta = self.config.max_joint_delta_deg
        if max_delta <= 0:
            # Safety guard disabled — always pass through.
            return self._set_joints_tracked(target_joints)

        now = time.monotonic()
        gap_s = float(self.config.safety_cmd_gap_held_s)
        if gap_s > 0 and (
            self._last_joint_cmd_ts is None
            or now - self._last_joint_cmd_ts >= gap_s
        ):
            # Long idle (e.g. LeapGence stream OFF skipped send_action): resume via HELD.
            if self._safety_state == "normal":
                self._safety_state = "held"
                if (
                    self._last_safety_hold_log_ts is None
                    or now - self._last_safety_hold_log_ts
                    >= self._SAFETY_HOLD_LOG_INTERVAL_S
                ):
                    idle_s = (
                        now - self._last_joint_cmd_ts
                        if self._last_joint_cmd_ts is not None
                        else float("inf")
                    )
                    print(
                        f"[DobotCR5] Safety hold: no joint command for "
                        f"{idle_s:.2f}s (>= {gap_s}s). Resuming via ENGAGING."
                    )
                    self._last_safety_hold_log_ts = now

        delta_deg = np.abs(np.rad2deg(target_arr - current_arr))
        over_limit = delta_deg > max_delta

        # --- transition: any state → HELD ---
        if over_limit.any():
            self._safety_state = "held"
            if (
                self._last_safety_hold_log_ts is None
                or now - self._last_safety_hold_log_ts
                >= self._SAFETY_HOLD_LOG_INTERVAL_S
            ):
                print(
                    f"[DobotCR5] Safety hold: target exceeds max_joint_delta_deg "
                    f"({max_delta}°) on joint(s) "
                    f"{np.where(over_limit)[0] + 1} — "
                    f"delta (deg): {np.round(delta_deg, 2).tolist()}. Skipping ServoJ."
                )
                self._last_safety_hold_log_ts = now
            return current_arr

        # --- transition: HELD → ENGAGING ---
        if self._safety_state == "held":
            self._safety_state = "engaging"
            self._last_safety_hold_log_ts = None

        # --- ENGAGING: clamped step toward target ---
        if self._safety_state == "engaging":
            if delta_deg.max() <= self.config.safety_engaged_threshold_deg:
                self._safety_state = "normal"
                # fall through to NORMAL below
            else:
                max_step_rad = np.deg2rad(self.config.safety_engage_max_step_deg)
                diff = target_arr - current_arr
                clamped = np.clip(diff, -max_step_rad, max_step_rad)
                clamped_target = current_arr + clamped
                return self._set_joints_tracked(clamped_target)

        # --- NORMAL: raw pass-through ---
        return self._set_joints_tracked(target_joints)

    # ---- send_action: parse -> plan -> execute -> format ----

    def _format_return_action(
        self,
        *,
        pre_action_pose: np.ndarray,
        sent_joints_rad: np.ndarray,
        sent_pose: np.ndarray,
        sent_gripper_pos: float | None = None,
    ) -> RobotAction:
        """Pure formatter: build the returned action dict from pre-computed values.

        All three key families are always included so that the extra dataset
        features (action.joint_pos, action.pose, action.pose_delta) can be
        assembled by build_dataset_frame regardless of the configured
        action_format.  The primary action_format determines which key family
        appears in the top-level ``action`` vector.
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
        if not self.is_connected or self._dobot is None:
            raise ConnectionError(f"{self} is not connected.")

        sent_gripper_pos = (
            float(action["gripper.pos"])
            if self._has_gripper() and "gripper.pos" in action
            else None
        )
        if sent_gripper_pos is not None and self.config.execute_action:
            self._dobot.set_gripper(sent_gripper_pos)

        kind = detect_action_kind(action, self._JOINT_COUNT)

        # Capture source state (cached for up to TTL) so pose_delta / joint_delta
        # apply consistently without an extra network read every tick.
        pre_action_pose: np.ndarray | None = self._current_pose()
        pre_action_joints: np.ndarray = self._current_joints_rad()
        target_joints: np.ndarray

        if kind == "joint":
            target_joints = extract_joint_action(action, self._JOINT_COUNT)

        elif kind == "joint_delta":
            delta = extract_joint_delta_action(action, self._JOINT_COUNT)
            target_joints = pre_action_joints.reshape(-1)[: self._JOINT_COUNT] + delta

        elif kind == "pose":
            target_pose = extract_pose_action(action)
            target_joints = self._inverse_kinematics(target_pose, pre_action_joints)

        elif kind == "pose_delta":
            delta_xyz, delta_orientation = extract_pose_delta_action(action)
            target_pose = apply_pose_delta(pre_action_pose, delta_xyz, delta_orientation)
            target_joints = self._inverse_kinematics(target_pose, pre_action_joints)

        else:
            raise ValueError(
                f"Unsupported Dobot action kind: {kind!r}; expected one of "
                "'joint', 'joint_delta', 'pose', 'pose_delta'."
            )

        if not self.config.execute_action:
            # Dry-run: skip all write operations; report current joints as "sent".
            dry_joints = pre_action_joints.reshape(-1)[: self._JOINT_COUNT]
            return self._format_return_action(
                pre_action_pose=pre_action_pose,
                sent_joints_rad=dry_joints,
                sent_pose=self._forward_kinematics(dry_joints),
                sent_gripper_pos=sent_gripper_pos,
            )

        target_arr = np.asarray(target_joints, dtype=np.float64).reshape(-1)[: self._JOINT_COUNT]
        current_arr = pre_action_joints.reshape(-1)[: self._JOINT_COUNT]
        sent_joints_rad = self._safety_gate_joints(target_arr, current_arr, target_joints)

        # Cache sent joints + FK so the next _current_* calls are delay-free.
        # Readers re-anchor from hardware when TTL expires.
        now = time.monotonic()
        sent_pose = self._forward_kinematics(sent_joints_rad)
        self._cached_joints_rad = np.asarray(
            sent_joints_rad, dtype=np.float64
        ).reshape(-1)[: self._JOINT_COUNT].copy()
        self._cached_joints_ts = now
        self._cached_tcp_pose = sent_pose.reshape(-1)[: 3 + POSE_ORIENTATION_WIDTH]
        self._cached_tcp_pose_ts = now

        return self._format_return_action(
            pre_action_pose=pre_action_pose,
            sent_joints_rad=sent_joints_rad,
            sent_pose=sent_pose,
            sent_gripper_pos=sent_gripper_pos,
        )

    def held_action_snapshot(self) -> RobotAction:
        """Return current absolute state formatted like ``send_action``, without commanding.

        Used by dual-arm teleop when only one stream is ON: the idle side stays put
        but still contributes ``pose`` / ``joint`` / ``pose_delta`` (=0) keys so
        dataset ``build_dataset_frame`` has a complete action feature set.
        """
        if not self.is_connected or self._dobot is None:
            raise ConnectionError(f"{self} is not connected.")
        joints = self._current_joints_rad()
        pose = self._current_pose()
        return self._format_return_action(
            pre_action_pose=pose,
            sent_joints_rad=joints,
            sent_pose=pose,
            sent_gripper_pos=self._current_gripper_pos(),
        )

    def go_home(self) -> None:
        """Reset gripper (if configured), then move to fixed home pose via blocking SDK motion."""
        if not self.is_connected or self._dobot is None:
            raise ConnectionError(f"{self} is not connected.")
        if self.config.home_pose is None:
            raise ValueError(
                "robot.home_pose is not configured. Set [x,y,z,rx,ry,rz] in robot config first."
            )
        if not self.config.execute_action:
            return

        if (
            self._has_gripper()
            and self._dobot.gripper is not None
            and self.config.home_gripper_pos is not None
        ):
            self._dobot.set_gripper(float(self.config.home_gripper_pos))

        pose = np.asarray(self.config.home_pose, dtype=np.float64).reshape(-1)
        if pose.size != 6:
            raise ValueError("robot.home_pose must contain exactly 6 values.")
        pose_quat = pose_euler_to_quat(pose)
        reached = self._dobot.set_pose(pose_quat, timeout=self.config.home_timeout_s)
        if not reached:
            raise TimeoutError(
                f"Dobot go_home timed out after {self.config.home_timeout_s}s at pose {pose.tolist()}."
            )

        # Re-anchor pose cache at home; invalidate joints so next read hits hardware.
        self._cached_tcp_pose = euler_pose_to_rot6d_pose(pose[:6])
        self._cached_tcp_pose_ts = time.monotonic()
        self._cached_joints_rad = None
        self._cached_joints_ts = None
        self._safety_state = "normal"
        self._last_joint_cmd_ts = time.monotonic()
