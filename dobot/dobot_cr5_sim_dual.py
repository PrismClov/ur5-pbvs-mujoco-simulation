"""Dual Dobot CR5 simulation: two :class:`DobotCR5SimRobot` arms, shared cameras."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.types import RobotAction, RobotObservation

from .._observation_cameras import read_observation_cameras
from ..gripper.damiao import DamiaoDualGripperConfig
from .dobot_cr5_dual import DobotCR5ArmSubConfig
from .dobot_cr5_sim import DobotCR5SimRobot, DobotCR5SimRobotConfig


def _arm_sub_to_sim_robot_config(arm: DobotCR5ArmSubConfig) -> DobotCR5SimRobotConfig:
    kwargs = asdict(arm)
    if arm.damiao_gripper is not None:
        kwargs["damiao_gripper"] = arm.damiao_gripper
    return DobotCR5SimRobotConfig(**kwargs, cameras={})


@RobotConfig.register_subclass("dobot_cr5_sim_dual")
@dataclass(kw_only=True)
class DobotCR5SimDualArmRobotConfig(RobotConfig):
    """Two simulated CR5 arms; cameras only at the top-level ``cameras`` key."""

    left_arm_config: DobotCR5ArmSubConfig
    right_arm_config: DobotCR5ArmSubConfig
    damiao_dual_gripper: DamiaoDualGripperConfig | None = None
    """When set, both arms share one DM-USB2FDCAN bus (``gripper_type: damiao`` on each arm)."""
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class DobotCR5SimDualArmRobot(Robot):
    """Two :class:`DobotCR5SimRobot` instances; observations/actions use ``left_`` / ``right_`` prefixes."""

    config_class = DobotCR5SimDualArmRobotConfig
    name = "dobot_cr5_sim_dual"

    def __init__(self, config: DobotCR5SimDualArmRobotConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = DobotCR5SimRobot(_arm_sub_to_sim_robot_config(config.left_arm_config))
        self.right_arm = DobotCR5SimRobot(_arm_sub_to_sim_robot_config(config.right_arm_config))
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
        result: dict = {}
        for arm_prefix, arm in (("left", self.left_arm), ("right", self.right_arm)):
            for feat_key, spec in arm.extra_dataset_features.items():
                prefix, short = feat_key.split(".", 1)
                new_key = f"{prefix}.{arm_prefix}_{short}"
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

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        left_action = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right_action = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}
        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)
        out: dict = {f"left_{k}": v for k, v in sent_left.items()}
        out.update({f"right_{k}": v for k, v in sent_right.items()})

        return out
