"""Dobot 实现：LeRobot ``Robot`` 封装（``dobot_cr5*``）与厂商 SDK（``sdk``）。

LeRobot ``make_device_from_device_class`` 通过配置类的父包查找
``<ConfigClassName 去掉 ``Config``>`` 设备类（见
``lerobot/utils/import_utils.py``），因此本包 re-export 三个 ``Robot`` 类。
"""

from __future__ import annotations

from .dobot_cr5 import DobotCR5Robot, DobotCR5RobotConfig
from .dobot_cr5_dual import (
    DobotCR5ArmSubConfig,
    DobotCR5DualArmRobot,
    DobotCR5DualArmRobotConfig,
)
from .dobot_cr5_sim import DobotCR5SimRobot, DobotCR5SimRobotConfig
from .dobot_cr5_sim_dual import (
    DobotCR5SimDualArmRobot,
    DobotCR5SimDualArmRobotConfig,
)

__all__ = [
    "DobotCR5Robot",
    "DobotCR5RobotConfig",
    "DobotCR5ArmSubConfig",
    "DobotCR5DualArmRobot",
    "DobotCR5DualArmRobotConfig",
    "DobotCR5SimRobot",
    "DobotCR5SimRobotConfig",
    "DobotCR5SimDualArmRobot",
    "DobotCR5SimDualArmRobotConfig",
]
