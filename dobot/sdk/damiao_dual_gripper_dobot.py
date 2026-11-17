# -*- coding: utf-8 -*-
"""Dual Damiao gripper adapter for Dobot dual-arm setups."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leapcollect.robots.gripper.damiao.config import DamiaoDualGripperConfig

from leapcollect.robots.gripper.damiao import DamiaoDualGripper
from .damiao_gripper_dobot import DamiaoGripperDobot


class DamiaoDualGripperDobot:
    """Two :class:`DamiaoGripperDobot` views on one shared USB FDCAN connection."""

    def __init__(self, dual: DamiaoDualGripper) -> None:
        self._dual = dual
        self.left = DamiaoGripperDobot.from_axis(dual.left, owns_bus=False)
        self.right = DamiaoGripperDobot.from_axis(dual.right, owns_bus=False)

    @classmethod
    def from_config(cls, config: DamiaoDualGripperConfig) -> DamiaoDualGripperDobot:
        return cls(config.build())

    def close(self) -> None:
        self._dual.close()
