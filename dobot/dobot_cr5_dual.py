"""双臂 Dobot CR5：两台独立 CR5（不同 IP），LeRobot 键名与 ``bi_so_follower`` 一致（``left_`` / ``right_`` 前缀）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.types import RobotAction, RobotObservation

from .._observation_cameras import read_observation_cameras
from .dobot_cr5 import DobotCR5Robot, DobotCR5RobotConfig
from ..gripper.damiao import DamiaoDualGripperConfig, DamiaoGripperConfig


@dataclass(kw_only=True)
class DobotCR5ArmSubConfig:
    """单臂网络/伺服/夹爪/运动学参数（**不是** :class:`RobotConfig`）。

    仅用于 :class:`DobotCR5DualArmRobotConfig` 的 ``left_arm_config`` / ``right_arm_config`` 嵌套。
    若此处使用已注册的 :class:`DobotCR5RobotConfig`，draccus 会再展开 ``CameraConfig`` 等选择器，
    与顶层 ``cameras`` 叠加后易在解析阶段触发递归过深（``RecursionError`` in ``canonicalize_union``）。

    运行时由 :func:`_arm_sub_to_robot_config` 转为 :class:`DobotCR5RobotConfig`（``cameras`` 恒为空）。
    """

    id: str | None = None
    calibration_dir: Path | None = None

    ip: str = "192.168.5.1"
    dashboard_port: int = 29999
    feed_port_four: int = 30004
    speed: int = 20
    servo_t: float = 0.2
    aheadtime: int = 50
    gain: int = 500
    gripper_type: str | None = None
    gripper_force: int = 80
    gripper_velocity: int = 500
    gripper_slave_id: int = 1
    gripper_parity: str = "N"
    gripper_do_index: int = 1
    tool_idx: int = 0
    urdf_path: str | None = None
    kin_target_frame: str = "eef"
    kin_joint_names: list[str] | None = None
    kin_ik_position_weight: float = 1.0
    kin_ik_orientation_weight: float = 0.01
    # draccus CLI decoding doesn't reliably support typing.Literal[...]
    action_format: str = "joint"
    observation_format: str = "joint"
    execute_action: bool = True
    """为 True 时下发 ServoJ / 夹爪；为 False 时仅读状态（与单臂 DobotCR5 一致）。"""
    max_joint_delta_deg: float = 20.0
    """单步最大关节变化（度）；超阈跳过 ServoJ。≤0 关闭保护。"""
    safety_engaged_threshold_deg: float = 8.0
    """安全保护恢复内阈值（度）；ENGAGING → NORMAL 的切换门槛。"""
    safety_engage_max_step_deg: float = 3.0
    """ENGAGING 阶段每步最大关节步长（度）。30 fps 下 3° ≈ 90°/s。"""
    safety_cmd_gap_held_s: float = 1.0
    """距上次关节指令超过此秒数时，恢复跟踪从 HELD 起跳。≤0 关闭。"""
    joint_command_max_velocity: float = 2.0
    joint_command_max_dt: float = 0.05
    joint_command_smoothing_factor: float = 0.1
    home_pose: list[float] | None = None
    """可选固定 home 位姿 ``[x,y,z,rx,ry,rz]``（米 + sxyz 欧拉弧度）。"""
    home_gripper_pos: float | None = None
    """可选 home 夹爪开合 ``[0,1]``。"""
    home_timeout_s: float = 10.0
    damiao_gripper: DamiaoGripperConfig | None = None
    """Nested config when ``gripper_type: damiao`` (Damiao USB2CANFD / motor-drive-layer)."""

    def __post_init__(self) -> None:
        if self.action_format not in ("joint", "pose", "pose_delta"):
            raise ValueError(
                "left/right_arm_config.action_format must be one of "
                "('joint', 'pose', 'pose_delta'), "
                f"got {self.action_format!r}."
            )
        if self.observation_format not in ("joint", "pose"):
            raise ValueError(
                "left/right_arm_config.observation_format must be one of "
                "('joint', 'pose'), "
                f"got {self.observation_format!r}."
            )
        if self.home_pose is not None and len(self.home_pose) != 6:
            raise ValueError(
                "left/right_arm_config.home_pose must be a 6-value Euler pose [x,y,z,rx,ry,rz]."
            )
        if self.home_gripper_pos is not None and not (0.0 <= float(self.home_gripper_pos) <= 1.0):
            raise ValueError("left/right_arm_config.home_gripper_pos must be in [0,1].")


def _arm_sub_to_robot_config(arm: DobotCR5ArmSubConfig) -> DobotCR5RobotConfig:
    kwargs = asdict(arm)
    # asdict() recurses into nested dataclasses; keep DamiaoGripperConfig so .build() works.
    if arm.damiao_gripper is not None:
        kwargs["damiao_gripper"] = arm.damiao_gripper
    return DobotCR5RobotConfig(**kwargs, cameras={})


@RobotConfig.register_subclass("dobot_cr5_dual")
@dataclass(kw_only=True)
class DobotCR5DualArmRobotConfig(RobotConfig):
    """左右各一台 CR5；**相机仅在顶层** ``cameras`` 配置，不由 ``left_arm_config`` / ``right_arm_config`` 继承。

    两臂子配置为 :class:`DobotCR5ArmSubConfig`（非 ``type: dobot_cr5`` 的 Choice）；例如：

    .. code-block:: yaml

        cameras:
          wrist_left: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
          wrist_right: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
          external: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}
        left_arm_config:
          ip: 192.168.5.1
        right_arm_config:
          ip: 192.168.5.2
    """

    left_arm_config: DobotCR5ArmSubConfig
    right_arm_config: DobotCR5ArmSubConfig
    damiao_dual_gripper: DamiaoDualGripperConfig | None = None
    """When set, both arms share one DM-USB2FDCAN bus (``gripper_type: damiao`` on each arm)."""
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    """整机相机：键名为观测中的图像键（如 ``wrist_left`` / ``wrist_right`` / ``external``）。"""


class DobotCR5DualArmRobot(Robot):
    """两台 CR5 关节伺服 + 可选夹爪；图像仅来自顶层 ``config.cameras``（与单臂 ``DobotCR5Robot`` 的用法一致，键名自定义）。

    遥操作动作需带 ``left_`` / ``right_`` 前缀（例如 :class:`~leapcollect.teleoperators.gello.gello_dual.GelloDualTeleop`
    或 ``vr_quest_dual``），与本类的 :meth:`send_action` 去前缀分发逻辑一致。
    """

    config_class = DobotCR5DualArmRobotConfig
    name = "dobot_cr5_dual"

    def __init__(self, config: DobotCR5DualArmRobotConfig):
        super().__init__(config)
        self.config = config
        # 相机只在整机层打开；子臂转为 DobotCR5RobotConfig 时 cameras 为空
        self.left_arm = DobotCR5Robot(_arm_sub_to_robot_config(config.left_arm_config))
        self.right_arm = DobotCR5Robot(_arm_sub_to_robot_config(config.right_arm_config))
        self.cameras = make_cameras_from_configs(config.cameras)
        self._damiao_dual_gripper = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm._motors_ft.items()},
            **{f"right_{k}": v for k, v in self.right_arm._motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            name: (self.cameras[name].height, self.cameras[name].width, 3) for name in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @property
    def extra_dataset_features(self) -> dict:
        """Prefixed extra features for both arms.

        The dual arm's get_observation() / send_action() prefix all per-arm scalar
        keys with ``left_`` / ``right_``, so the names lists in each feature spec
        must match those prefixed keys.  This is derived automatically from each
        arm's own extra_dataset_features by splitting the feature key at the first
        dot and prepending the arm prefix to both the key's short name and every
        entry in names.

        Example transformation for the left arm:
            feat key  ``observation.joint_pos``  → ``observation.left_joint_pos``
            names     ``["joint_1.pos", ...]``   → ``["left_joint_1.pos", ...]``
        """
        result: dict = {}
        for arm_prefix, arm in (("left", self.left_arm), ("right", self.right_arm)):
            for feat_key, spec in arm.extra_dataset_features.items():
                prefix, short = feat_key.split(".", 1)          # e.g. "observation", "joint_pos"
                new_key = f"{prefix}.{arm_prefix}_{short}"      # "observation.left_joint_pos"
                new_names = [f"{arm_prefix}_{n}" for n in spec["names"]]
                result[new_key] = {**spec, "names": new_names}
        return result

    @property
    def is_connected(self) -> bool:
        cams_ok = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        return self.left_arm.is_connected and self.right_arm.is_connected and cams_ok

    def connect(self, calibrate: bool = True) -> None:
        if self.config.damiao_dual_gripper is not None:
            from leapcollect.robots.dobot.sdk.damiao_dual_gripper_dobot import (
                DamiaoDualGripperDobot,
            )

            self._damiao_dual_gripper = DamiaoDualGripperDobot.from_config(
                self.config.damiao_dual_gripper
            )
            self.left_arm.connect(calibrate, gripper_dobot=self._damiao_dual_gripper.left)
            self.right_arm.connect(calibrate, gripper_dobot=self._damiao_dual_gripper.right)
        else:
            self.left_arm.connect(calibrate)
            self.right_arm.connect(calibrate)
        for cam in self.cameras.values():
            cam.connect()

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        if self._damiao_dual_gripper is not None:
            self._damiao_dual_gripper.close()
            self._damiao_dual_gripper = None

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        left_obs = self.left_arm.get_observation()
        right_obs = self.right_arm.get_observation()

        obs: dict = {}
        obs.update({f"left_{k}": v for k, v in left_obs.items()})
        obs.update({f"right_{k}": v for k, v in right_obs.items()})
        obs.update(read_observation_cameras(self.cameras))

        return obs

    def get_ee_force(self) -> dict[str, object] | None:
        """Return available left/right TCP wrenches for dual-leader haptic feedback."""
        if not self.is_connected:
            return None
        out: dict[str, object] = {}
        left_force = self.left_arm.get_ee_force()
        if left_force is not None:
            out["left_ee_force"] = left_force
        right_force = self.right_arm.get_ee_force()
        if right_force is not None:
            out["right_ee_force"] = right_force
        return out or None

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        left_action = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right_action = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}
        out: dict = {}
        # Partial teleop (one LeapGence stream ON): command only that side; snapshot
        # the other so returned keys still match dual action_features / dataset.
        if left_action:
            sent_left = self.left_arm.send_action(left_action)
        else:
            sent_left = self.left_arm.held_action_snapshot()
        out.update({f"left_{k}": v for k, v in sent_left.items()})
        if right_action:
            sent_right = self.right_arm.send_action(right_action)
        else:
            sent_right = self.right_arm.held_action_snapshot()
        out.update({f"right_{k}": v for k, v in sent_right.items()})

        return out

    def go_home(self) -> None:
        """Move both arms to their configured fixed home poses."""
        self.left_arm.go_home()
        self.right_arm.go_home()
