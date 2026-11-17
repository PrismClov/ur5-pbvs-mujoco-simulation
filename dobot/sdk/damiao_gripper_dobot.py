# -*- coding: utf-8 -*-
"""Damiao gripper adapter for :class:`~leapcollect.robots.dobot.sdk.interface.DobotInterface`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leapcollect.robots.gripper.damiao.config import DamiaoGripperConfig

from leapcollect.robots.gripper.damiao import DamiaoGripper
from leapcollect.robots.gripper.damiao.damiao_gripper_axis import DamiaoGripperAxis


class DamiaoGripperDobot:
    """Damiao adapter: continuous normalized position in ``[0, 1]`` (closed → open).

    ``set_pos`` also accepts legacy int ``0–1000`` for Modbus-style callers.
    """

    POS_MIN = 0
    POS_MAX = 1000

    def __init__(self, gripper: DamiaoGripper | DamiaoGripperAxis, *, owns_bus: bool = True) -> None:
        self._gripper = gripper
        self._owns_bus = owns_bus

    @classmethod
    def from_config(cls, config: DamiaoGripperConfig) -> DamiaoGripperDobot:
        return cls(config.build())

    @classmethod
    def from_axis(cls, axis: DamiaoGripperAxis, *, owns_bus: bool = False) -> DamiaoGripperDobot:
        return cls(axis, owns_bus=owns_bus)

    def set_pos(self, val: int | float, blocking: bool = False) -> None:
        del blocking
        if isinstance(val, float):
            normalized = max(0.0, min(1.0, val))
        else:
            hw = int(max(self.POS_MIN, min(self.POS_MAX, val)))
            normalized = hw / 1000.0
        self._gripper.set_gripper(normalized)

    def get_normalized_position(self) -> float:
        return float(self._gripper.get_gripper_position())

    def close(self) -> None:
        if self._owns_bus and hasattr(self._gripper, "close"):
            self._gripper.close()
