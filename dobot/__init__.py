"""Dobot SDK and optional LeRobot wrappers.

The vendor SDK under :mod:`dobot.sdk` must remain importable on machines that
do not have the optional LeRobot dependencies installed. Robot wrappers are
therefore loaded lazily on first attribute access.
"""

from __future__ import annotations

from importlib import import_module

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

_LAZY_EXPORTS = {
    "DobotCR5Robot": (".dobot_cr5", "DobotCR5Robot"),
    "DobotCR5RobotConfig": (".dobot_cr5", "DobotCR5RobotConfig"),
    "DobotCR5ArmSubConfig": (".dobot_cr5_dual", "DobotCR5ArmSubConfig"),
    "DobotCR5DualArmRobot": (".dobot_cr5_dual", "DobotCR5DualArmRobot"),
    "DobotCR5DualArmRobotConfig": (
        ".dobot_cr5_dual", "DobotCR5DualArmRobotConfig"),
    "DobotCR5SimRobot": (".dobot_cr5_sim", "DobotCR5SimRobot"),
    "DobotCR5SimRobotConfig": (".dobot_cr5_sim", "DobotCR5SimRobotConfig"),
    "DobotCR5SimDualArmRobot": (
        ".dobot_cr5_sim_dual", "DobotCR5SimDualArmRobot"),
    "DobotCR5SimDualArmRobotConfig": (
        ".dobot_cr5_sim_dual", "DobotCR5SimDualArmRobotConfig"),
}


def __getattr__(name: str):
    try:
        module_name, symbol = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value
