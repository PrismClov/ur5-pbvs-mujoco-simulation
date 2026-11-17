"""Kinematic MuJoCo display for the Dobot CR5 URDF.

MuJoCo is a viewer only: callers write arm joint positions and call ``mj_forward``.
There is no ``mj_step`` / actuator / contact loop.

Vendor visual meshes are Collada (``.dae``); MuJoCo has no DAE decoder, so the
viewer keeps collision ``.stl`` geometry. Placeholder gripper STLs (134-byte
``LeapCollect-placeholder`` files) are stripped so compile can succeed.
"""

from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CR5_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 7))
_PLACEHOLDER_PREFIX = b"LeapCollect-placeholder"
_MIN_MESH_BYTES = 256


def _absolute_mesh_filename(urdf_dir: Path, filename: str) -> str:
    if filename.startswith("package://"):
        rest = filename[len("package://") :]
        filename = rest.split("/", 1)[1] if "/" in rest else rest
    path = Path(filename)
    if not path.is_absolute():
        path = (urdf_dir / path).resolve()
    return str(path)


def rewrite_urdf_mesh_paths(urdf_text: str, urdf_path: Path) -> str:
    """Rewrite mesh filenames to absolute paths so a temp URDF still loads."""
    urdf_dir = urdf_path.resolve().parent

    def _replace(match: re.Match[str]) -> str:
        return f'filename="{_absolute_mesh_filename(urdf_dir, match.group(1))}"'

    return re.sub(r'filename="([^"]+)"', _replace, urdf_text)


def _usable_mesh(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size < _MIN_MESH_BYTES:
        return False
    return not path.read_bytes()[: len(_PLACEHOLDER_PREFIX)].startswith(_PLACEHOLDER_PREFIX)


def _strip_unusable_meshes(urdf_text: str) -> str:
    root = ET.fromstring(urdf_text)
    parent = {child: node for node in root.iter() for child in list(node)}
    drop: list[tuple[ET.Element, ET.Element]] = []
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename") or ""
        if _usable_mesh(Path(filename)):
            continue
        node = mesh
        while node is not None and node.tag not in ("visual", "collision"):
            node = parent.get(node)
        if node is not None and node in parent:
            drop.append((parent[node], node))
    for owner, node in drop:
        owner.remove(node)
    return ET.tostring(root, encoding="unicode")


def build_mujoco_model(urdf_path: str | Path):
    """Compile the CR5 URDF into a MuJoCo model with a floor and key light."""
    import mujoco

    urdf_path = Path(urdf_path).resolve()
    text = _strip_unusable_meshes(rewrite_urdf_mesh_paths(urdf_path.read_text(), urdf_path))

    with tempfile.TemporaryDirectory(prefix="leapcollect_cr5_mujoco_") as tmp:
        tmp_urdf = Path(tmp) / "cr5.urdf"
        tmp_urdf.write_text(text)
        spec = mujoco.MjSpec.from_file(str(tmp_urdf))

        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size[:] = [2.0, 2.0, 0.05]
        floor.rgba[:] = [0.75, 0.75, 0.78, 1.0]

        light = spec.worldbody.add_light()
        light.name = "key_light"
        light.pos[:] = [1.5, -1.5, 2.5]
        light.dir[:] = [-0.4, 0.4, -1.0]
        light.castshadow = True

        model = spec.compile()

    joint_qposadr: dict[str, int] = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is not None:
            joint_qposadr[name] = int(model.jnt_qposadr[j])

    missing = [n for n in CR5_ARM_JOINT_NAMES if n not in joint_qposadr]
    if missing:
        raise RuntimeError(
            f"MuJoCo model missing CR5 joints {missing}; got {sorted(joint_qposadr)}"
        )
    if model.ngeom < 8:
        raise RuntimeError(
            f"MuJoCo model has ngeom={model.ngeom}; expected CR5 collision meshes."
        )
    return model, joint_qposadr


def set_arm_qpos(data, joint_qposadr: dict[str, int], joints_rad: np.ndarray) -> None:
    joints = np.asarray(joints_rad, dtype=np.float64).reshape(-1)
    n = len(CR5_ARM_JOINT_NAMES)
    if joints.size < n:
        raise ValueError(f"Expected {n} arm joints, got {joints.size}")
    for name, value in zip(CR5_ARM_JOINT_NAMES, joints[:n], strict=True):
        data.qpos[joint_qposadr[name]] = float(value)


class DobotMujocoViewer:
    """Passive MuJoCo window that tracks CR5 joint commands."""

    def __init__(self, urdf_path: str | Path, joints_rad=None):
        try:
            import mujoco
            from mujoco import viewer
        except ImportError as exc:
            raise ImportError(
                "dobot_cr5_sim render=true requires mujoco. Install with: pip install mujoco"
            ) from exc

        self._mujoco = mujoco
        self.model, self.joint_qposadr = build_mujoco_model(urdf_path)
        self.data = mujoco.MjData(self.model)
        q0 = (
            np.zeros(len(CR5_ARM_JOINT_NAMES))
            if joints_rad is None
            else np.asarray(joints_rad, dtype=np.float64)
        )
        set_arm_qpos(self.data, self.joint_qposadr, q0)
        mujoco.mj_forward(self.model, self.data)

        self._viewer = viewer.launch_passive(self.model, self.data)
        self._viewer.cam.lookat[:] = np.array([0.25, -0.15, 0.55], dtype=np.float64)
        self._viewer.cam.distance = 1.8
        self._viewer.cam.azimuth = 140.0
        self._viewer.cam.elevation = -20.0
        self._viewer.sync()

    def set_joints(self, joints_rad) -> None:
        set_arm_qpos(self.data, self.joint_qposadr, joints_rad)
        self._mujoco.mj_forward(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
